"""Analyze a saved, actual five-minute acquisition; no model/API calls."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hrv import HRVAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument('log', type=Path)
args = parser.parse_args()
rows = []
for line in args.log.read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line.strip('\x04\r'))
    except ValueError:
        continue
    if isinstance(row, dict):
        rows.append(row)
samples = [r for r in rows if r.get('type') == 'sample']
results = [r for r in rows if r.get('type') == 'result']
if not results:
    raise SystemExit('No completed real device result in log')
result = results[-1]
raw = np.asarray([v for r in samples + [result] for v in r.get('raw_ir', [])], dtype=float)
adc = np.asarray([v for r in samples + [result] for v in r.get('adc_a0_codes', [])], dtype=float) * 4.096 / 32768
rate = 100.0
time_axis = np.arange(raw.size) / rate
has_device_metrics = 'metrics' in result
algorithm_reports = []
if has_device_metrics:
    metrics = result['metrics']
    rr = np.asarray(result.get('rr_ms', []), dtype=float)
    algorithm_reports = [r['metrics'] for r in samples]
else:
    analyzer = HRVAnalyzer(fs=100, window_sec=0, min_peak_ms=450,
                           max_peak_ms=2000, peak_ratio=0.5)
    for row in rows:
        if row.get('type') == 'warmup':
            for ir in row.get('raw_ir', []):
                analyzer.add_sample(ir)
    dc, lp, envelope = analyzer.dc, analyzer.lp, analyzer.amp_env
    analyzer.reset()
    analyzer.dc, analyzer.lp, analyzer.amp_env = dc, lp, envelope
    for row in samples + [result]:
        labels = row.get('raw_motion', [0]*len(row.get('raw_ir', [])))
        if len(labels) != len(row.get('raw_ir', [])):
            raise SystemExit('Motion/PPG label length mismatch')
        for ir, moving in zip(row.get('raw_ir', []), labels):
            analyzer.add_sample(ir, motion=bool(moving))
        m = analyzer.metrics()
        if row.get('type') == 'sample':
            algorithm_reports.append(m)
    metrics = analyzer.metrics()
    rr = np.asarray(analyzer._clean_rr(), dtype=float)
quality = np.asarray([r['sig_quality'] for r in algorithm_reports])
sampling_screen = (raw.size == result['counts']['ppg'] and
                   99.5 <= result['ppg_samples_per_s'] <= 100.5 and
                   result['errors']['ppg'] == 0 and result['imu_unknown_reads'] == 0)
summary = {
    'source_log': str(args.log.resolve()),
    'scope': 'Real PPG-derived PRV and internal waveform cross-check; no ECG or clinical accuracy validation.',
    'duration_s': result['duration_s'], 'counts': result['counts'], 'errors': result['errors'],
    'ppg_samples_per_s': result['ppg_samples_per_s'],
    'saved_raw_samples': int(raw.size), 'raw_count_matches_device': raw.size == result['counts']['ppg'],
    'gain_led_code': result['gain_led_code'], 'gain_dc': result['gain_dc'],
    'algorithm_metrics': metrics,
    'metrics_location': 'ESP32 cumulative diagnostic' if has_device_metrics else 'PC replay of existing project algorithm',
    'saturated_samples': result['saturated_samples'],
    'dark_samples': result['dark_samples'],
    'quality_report_count': int(quality.size),
    'quality_ge_0_5_reports': int(np.sum(quality >= 0.5)),
    'tail_60_quality_ge_0_5': int(np.sum(quality[-60:] >= 0.5)),
    'accepted_interval_count': int(rr.size),
    'accepted_interval_sum_s': float(np.sum(rr) / 1000),
    'accepted_interval_sum_over_duration': float(np.sum(rr)/1000/result['duration_s']),
    'interval_coverage_note': 'Approximation from interval sum, not an exact contiguous valid-time map.',
    'imu_motion_reads': result['imu_motion_reads'], 'imu_unknown_reads': result['imu_unknown_reads'],
    'sampling_screen_pass': sampling_screen,
    'sampling_screen_source': 'Proposed engineering screen: 99.5-100.5 samples/s, matching saved count, no PPG read errors or unknown IMU state; not a proposal or medical accuracy criterion.',
    'interpretation': 'Diagnostic values only; sample timing/communications failed screening.' if not sampling_screen else 'Sampling passed engineering screen; physiological accuracy still requires an independent reference.',
}
if adc.size:
    summary['gsr_a0'] = {'reads': int(adc.size), 'min_v': float(adc.min()),
                         'max_v': float(adc.max()), 'mean_v': float(adc.mean()),
                         'std_v': float(adc.std()), 'low_le_5mv': int(np.sum(adc <= 0.005)),
                         'note': 'Voltage only; not converted to calibrated conductance or emotional state.'}
filtered = np.zeros_like(raw)
peaks = np.asarray([], dtype=int)
if raw.size >= 1000:
    filtered = sosfiltfilt(butter(3, [0.5, 4.0], btype='bandpass', fs=rate, output='sos'), raw)
    robust = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))
    peaks, _ = find_peaks(filtered, distance=45, prominence=max(1.0, robust * 0.2))
    intervals = np.diff(peaks) / rate * 1000
    valid = intervals[(intervals >= 450) & (intervals <= 2000)]
    summary['offline_waveform_crosscheck'] = {
        'method': '0.5-4Hz zero-phase bandpass; minimum peak spacing 450ms, prominence 20% of whole-run 5-95 percentile amplitude.',
        'detected_peaks': int(peaks.size), 'range_filtered_intervals': int(valid.size),
        'hr_median_interval_bpm': float(60000/np.median(valid)) if valid.size else None,
        'rmssd_ms': float(np.sqrt(np.mean(np.diff(valid)**2))) if valid.size > 1 else None,
        'sdnn_ms': float(np.std(valid, ddof=1)) if valid.size > 1 else None,
        'note': 'A different detector on the same PPG; agreement does not establish truth or ECG accuracy.'}
fig, axes = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
end = min(raw.size, 60*100)
start = max(0, end-15*100)
axes[0].plot(time_axis[start:end], raw[start:end], lw=0.8)
axes[0].set(title='Actual resting PPG: 15-second excerpt', ylabel='IR ADC code')
axes[1].plot(time_axis[start:end], filtered[start:end], lw=0.8)
shown = peaks[(peaks >= start) & (peaks < end)]
axes[1].scatter(time_axis[shown], filtered[shown], s=18, color='tab:red', label='Offline detector')
axes[1].set(ylabel='Bandpassed PPG', xlabel='Nominal sample time (s)')
axes[1].legend(loc='upper right')
axes[2].plot([r['elapsed_s'] for r in samples], quality, lw=1)
axes[2].axhline(0.5, linestyle='--', color='tab:red', label='Software heuristic gate')
axes[2].set(xlabel='Host-reported device elapsed time (s)', ylabel='Quality score', ylim=(0, 1.05))
axes[2].legend(loc='upper right')
if adc.size:
    # Read count does not expose individual ADC timestamps, so show ordinal.
    axes[3].plot(np.arange(adc.size), adc, lw=0.7)
axes[3].set(xlabel='Successful ADC read index', ylabel='A0 voltage (V)')
image = args.log.with_suffix('.waveform.png')
fig.savefig(image, dpi=160)
plt.close(fig)
analysis = args.log.with_suffix('.analysis.json')
analysis.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=2))
print('PLOT', image)
print('ANALYSIS', analysis)
