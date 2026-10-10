"""Replay production detector and visualize raw diagnostics; no accuracy claims."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
import numpy as np
from scipy import signal
from analyse_raw_diagnostic import split_ppg
from hrv import HRVAnalyzer


def replay(piece):
    detector = HRVAnalyzer()
    summaries = []
    detected = []
    next_report = piece[0, 1] + 1000
    for row in piece:
        if detector.add_sample(float(row[3]), float(row[1])):
            detected.append(float(detector.last_peak_t))
        if row[1] >= next_report:
            summaries.append(detector.metrics())
            next_report = row[1] + 1000
    summaries.append(detector.metrics())
    return {'samples': len(piece), 'span_s': float((piece[-1, 1] - piece[0, 1]) / 1000),
            'all_detected_peaks': len(detected), 'internal_continuity_breaks': detector.segment_id,
            'max_quality_score': max(m['sig_quality'] for m in summaries),
            'max_pi_pct': max(m['pi'] for m in summaries),
            'max_nn': max(m['nn'] for m in summaries), 'final_metrics': summaries[-1],
            'peak_times_ms': detected}


def run(path, plot=True):
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    stages = {}
    complete = False
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('type') == 'complete':
            complete = True
        if row.get('type') == 'raw':
            s = stages.setdefault(row['stage'], {'ppg': [], 'eda': []})
            s['ppg'].extend(row['ppg'])
            s['eda'].extend(row['eda'])
    if not complete:
        raise ValueError('Capture incomplete; wait for completion before replay')
    report = {'source': str(path.resolve()), 'accuracy_validated': False,
              'classification_allowed': False, 'note': 'Same-source algorithm comparison with nominal FIFO time only.',
              'stages': {}}
    if plot:
        fig, axes = plt.subplots(len(stages), 3, figsize=(15, 4 * len(stages)), squeeze=False)
    for index, (name, s) in enumerate(stages.items()):
        pieces = split_ppg(s['ppg'])
        good = [p for p in pieces if len(p) >= 1000]
        entries = [replay(p) for p in good]
        report['stages'][name] = entries
        if not plot:
            continue
        if good:
            piece = max(good, key=len)
            # Example starts 5 s after continuity begins, avoiding filter startup.
            start = min(500, max(0, len(piece) - 1000))
            clip = piece[start:start + 1000]
            t = (clip[:, 1] - clip[0, 1]) / 1000
            for col, channel, color in [(2, 'RED', 'firebrick'), (3, 'IR', 'navy')]:
                axes[index, 0].plot(t, clip[:, col] - np.median(clip[:, col]), label=channel, color=color, lw=1)
                band = signal.sosfiltfilt(signal.butter(3, [.5, 4], fs=100, btype='bandpass', output='sos'), piece[:, col])
                axes[index, 1].plot(t, band[start:start + 1000], label=channel, color=color, lw=1)
            axes[index, 0].set_title(name + ': raw, median removed')
            axes[index, 1].set_title('0.5-4 Hz offline bandpass (not ground truth)')
            for ax in axes[index, :2]:
                ax.set_xlabel('Seconds in example'); ax.set_ylabel('ADC counts'); ax.legend()
        if s['eda']:
            eda = np.asarray(s['eda'], dtype=float)
            axes[index, 2].plot(eda[:, 0] / 1000, eda[:, 2], color='darkgreen', lw=.7)
            axes[index, 2].set_title('GSR raw voltage, whole stage')
            axes[index, 2].set_xlabel('Stage seconds'); axes[index, 2].set_ylabel('Volts')
    if plot:
        fig.suptitle('Worn diagnostic evidence — nominal PPG time, no emotion labels')
        fig.tight_layout()
        plot_path = path.with_suffix('.comparison.png')
        if plot_path.exists():
            raise FileExistsError(plot_path)
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    output = path.with_suffix('.detectors.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)
    if plot:
        print(plot_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--no-plot', action='store_true', help='Replay without optional matplotlib dependency')
    args = parser.parse_args()
    run(args.evidence, plot=not args.no_plot)
