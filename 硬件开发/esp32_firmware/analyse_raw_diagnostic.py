"""Offline diagnostics only; no quality acceptance, labels or model input.

Never join FIFO segments or estimate HRV across a missing sample. PPG timing
is nominal, and agreement between optical channels is not an ECG validation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
import numpy as np
from scipy import signal


def split_ppg(rows):
    """Split at segment changes, motion, nonmonotonic times and missing samples."""
    pieces, current = [], []
    previous = None
    for row in rows:
        if len(row) != 5:
            raise ValueError('Invalid PPG row')
        if row[4]:
            if current:
                pieces.append(np.asarray(current, dtype=float))
            current, previous = [], None
            continue
        if previous is not None and (row[0] != previous[0] or
                                      not 9.5 <= row[1] - previous[1] <= 10.5):
            if current:
                pieces.append(np.asarray(current, dtype=float))
            current = []
        current.append(row)
        previous = row
    if current:
        pieces.append(np.asarray(current, dtype=float))
    return pieces


def spectrum(values, hz, low, high):
    f, power = signal.welch(values, hz, nperseg=min(len(values), int(hz * 8)))
    mask = (f >= low) & (f <= high)
    if not np.any(mask) or not np.any(power[mask] > 0):
        return None
    return float(f[mask][np.argmax(power[mask])])


def optical_window(piece):
    result = {'span_s': float((piece[-1, 1] - piece[0, 1]) / 1000),
              'segment': int(piece[0, 0]), 'samples': len(piece)}
    filtered = []
    peaks_by_channel = []
    for name, col in [('red', 2), ('ir', 3)]:
        values = piece[:, col]
        band = signal.sosfiltfilt(signal.butter(3, [.5, 4], fs=100, btype='bandpass', output='sos'), values)
        # Trim filter edges before pulse comparisons.
        band = band[200:-200]
        dc = float(np.median(values))
        amplitude = float(np.percentile(band, 95) - np.percentile(band, 5))
        prominence = max(amplitude * .2, 1)
        peaks, _ = signal.find_peaks(band, distance=30, prominence=prominence)
        valleys, _ = signal.find_peaks(-band, distance=30, prominence=prominence)
        def candidate_summary(indices):
            intervals = np.diff(indices) * 10.0
            return {'peaks': len(indices),
                    'median_bpm': float(60000 / np.median(intervals)) if len(intervals) else None,
                    'intervals_outside_300_2000ms': int(np.count_nonzero((intervals < 300) | (intervals > 2000)))}
        result[name] = {'dc': dc, 'raw_std': float(np.std(values)),
                        'clipped_fraction': float(np.mean((values <= 0) | (values >= 262143))),
                        'band_p95_p5_counts': amplitude,
                        'band_p95_p5_over_dc_pct': amplitude / dc * 100 if dc > 0 else None,
                        'dominant_hz': spectrum(band, 100, .5, 4),
                        'positive_peaks': candidate_summary(peaks),
                        'negative_peaks': candidate_summary(valleys)}
        filtered.append(band)
        peaks_by_channel.append(peaks)
    if min(np.std(v) for v in filtered) > 0:
        result['red_ir_zero_lag_correlation'] = float(np.corrcoef(filtered)[0, 1])
    else:
        result['red_ir_zero_lag_correlation'] = None
    # Candidate coincidence, not a pass threshold or independent reference.
    a, b = peaks_by_channel
    result['positive_peak_coincidence_within_80ms'] = float(np.mean([
        np.min(np.abs(b - p)) <= 8 for p in a])) if len(a) and len(b) else None
    return result


def eda_summary(rows):
    if len(rows) < 20:
        return {'samples': len(rows), 'analysis': 'insufficient'}
    data = np.asarray(rows, dtype=float)
    dt = np.diff(data[:, 0]) / 1000
    if np.any(dt <= 0):
        return {'samples': len(rows), 'analysis': 'nonmonotonic timestamps'}
    hz = 1 / float(np.median(dt))
    result = {'samples': len(rows), 'median_hz': hz,
              'max_gap_ms': float(np.max(dt) * 1000),
              'raw_std_v': float(np.std(data[:, 2])),
              'raw_step_rms_v': float(np.sqrt(np.mean(np.diff(data[:, 2]) ** 2))),
              'voltage_min': float(np.min(data[:, 2])), 'voltage_max': float(np.max(data[:, 2]))}
    if np.max(dt) > .15 or hz <= 5:
        result['analysis'] = 'gaps/rate unsuitable for spectral comparison'
        return result
    grid = np.arange(data[0, 0], data[-1, 0], 1000 / hz)
    values = np.interp(grid, data[:, 0], data[:, 2])
    result['dominant_raw_hz'] = spectrum(values, hz, .2, hz / 2)
    low = signal.sosfiltfilt(signal.butter(3, 2, fs=hz, output='sos'), values)
    result['lowpass_2hz_std_v'] = float(np.std(low))
    result['lowpass_2hz_step_rms_v'] = float(np.sqrt(np.mean(np.diff(low) ** 2)))
    result['note'] = 'Filtered values are exploratory; voltage is not calibrated conductance.'
    return result


def analyse(path):
    stages = {}
    complete = reset_sent = False
    production_restored = False
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError('Invalid JSON at line %d' % line_number) from exc
        if row.get('type') == 'complete':
            complete = True
        if row.get('type') == 'host_result':
            reset_sent = bool(row.get('reset_sent'))
            production_restored = bool(row.get('startup_seen') and row.get('production_packet_seen'))
        if row.get('type') in ('stage', 'raw', 'result', 'adc_config'):
            stage = stages.setdefault(row['stage'], {'ppg': [], 'eda': [], 'seq': [], 'meta': {}, 'result': {}})
            if row['type'] == 'raw':
                stage['ppg'].extend(row['ppg'])
                stage['eda'].extend(row['eda'])
                stage['seq'].append(row['seq'])
            elif row['type'] == 'adc_config':
                stage['adc_config'] = row
            else:
                stage['meta' if row['type'] == 'stage' else 'result'] = row
    report = {'source': str(path.resolve()), 'capture_complete': complete,
              'reset_command_sent': reset_sent, 'classification_allowed': False,
              'production_restart_observed': production_restored,
              'accuracy_validated': False, 'stages': {}}
    for name, stage in stages.items():
        pieces = split_ppg(stage['ppg'])
        # Fixed 10 s nonoverlapping diagnostics; never merge discontinuities.
        windows = [optical_window(p[i:i + 1000]) for p in pieces
                   for i in range(0, len(p) - 999, 1000)]
        seq = stage['seq']
        report['stages'][name] = {'settings': stage['meta'], 'adc_config': stage.get('adc_config'), 'device_result': stage['result'],
            'packet_sequence_contiguous': bool(seq) and seq == list(range(1, len(seq) + 1)),
            'raw_ppg_samples': len(stage['ppg']), 'unblocked_segments': len(pieces),
            'longest_unblocked_span_s': max(((p[-1, 1] - p[0, 1]) / 1000 for p in pieces), default=0),
            'optical_windows': windows, 'eda': eda_summary(stage['eda'])}
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('evidence', type=Path)
    args = ap.parse_args()
    result = analyse(args.evidence)
    output = args.evidence.with_suffix('.analysis.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)
