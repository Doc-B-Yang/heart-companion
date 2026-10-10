"""Shadow replay of fixed PPG polarities; no labels or accuracy acceptance.

The same real IR samples and nominal timestamps feed both candidates. Never
pick polarity per window to maximize quality, stitch gaps, or repair intervals.
"""
import argparse
import json
from pathlib import Path

from analyse_raw_diagnostic import split_ppg
from hrv import HRVAnalyzer


def replay(piece, polarity):
    detector = HRVAnalyzer(polarity=polarity)
    rows, peaks, breaks = [], [], []
    next_report = piece[0, 1] + 1000
    previous_segment = detector.segment_id
    for sample in piece:
        if detector.add_sample(float(sample[3]), float(sample[1])):
            peaks.append(float(detector.last_peak_t))
        if detector.segment_id != previous_segment:
            breaks.append(float(sample[1]))
            previous_segment = detector.segment_id
        if sample[1] >= next_report:
            metrics = detector.metrics()
            rows.append({'time_ms': float(sample[1]), **metrics})
            next_report = sample[1] + 1000
    rows.append({'time_ms': float(piece[-1, 1]), **detector.metrics()})
    return {'polarity': polarity, 'samples': len(piece),
            'start_ms': float(piece[0, 1]), 'end_ms': float(piece[-1, 1]),
            'max_quality': max(r['sig_quality'] for r in rows),
            'max_pi_pct': max(r['pi'] for r in rows),
            'max_nn': max(r['nn'] for r in rows),
            'internal_continuity_breaks': detector.segment_id,
            'break_times_ms': breaks, 'peak_times_ms': peaks, 'metrics': rows}


def compare(path):
    stages = {}
    complete = False
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('type') == 'complete':
            complete = True
        if row.get('type') == 'raw':
            stages.setdefault(row['stage'], []).extend(row['ppg'])
    if not complete:
        raise ValueError('Incomplete capture; polarity replay is not accepted')
    return {'source': str(path.resolve()), 'classification_allowed': False,
            'accuracy_validated': False, 'deployed': False,
            'timing_reference': 'nominal_fifo_clock',
            'note': 'Fixed-polarity shadow candidates, same-source comparison only; no automatic selection.',
            'stages': {name: [{'positive': replay(p, 1), 'negative': replay(p, -1)}
                              for p in split_ppg(samples) if len(p) >= 1000]
                       for name, samples in stages.items()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    path = parser.parse_args().evidence
    result = compare(path)
    output = path.with_suffix('.polarity.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    for name, pieces in result['stages'].items():
        for piece in pieces:
            print(name, json.dumps({key: {field: entry[field] for field in
                ('max_quality', 'max_pi_pct', 'max_nn', 'internal_continuity_breaks')}
                for key, entry in piece.items()}))
    print(output)
