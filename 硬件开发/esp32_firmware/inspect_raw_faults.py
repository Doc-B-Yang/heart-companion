"""Summarize faults without repairing raw data or accepting classification."""
import argparse
import json
from pathlib import Path
from collections import defaultdict


def inspect(path):
    stages = defaultdict(lambda: {'eda': [], 'fifo': [], 'imu': []})
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row.get('type') == 'raw':
            stage = stages[row['stage']]
            stage['eda'].extend(row['eda'])
            stage['fifo'].extend(row.get('fifo_events', []))
            stage['imu'].extend(row.get('imu', []))
    report = {'source': str(path.resolve()), 'classification_allowed': False, 'stages': {}}
    for name, stage in stages.items():
        eda = stage['eda']
        low = [r for r in eda if r[2] < .02]
        runs, current = [], []
        for row in eda:
            if row[2] < .02:
                current.append(row)
            elif current:
                runs.append(current); current = []
        if current:
            runs.append(current)
        report['stages'][name] = {
            'eda_samples': len(eda), 'below_20mv_samples': len(low),
            'below_20mv_fraction': len(low) / len(eda) if eda else None,
            'low_runs': [{'start_ms': r[0][0], 'end_ms': r[-1][0], 'samples': len(r)} for r in runs],
            'low_codes': sorted(set(r[1] for r in low)),
            'large_steps_over_30mv': sum(abs(b[2]-a[2]) > .03 for a,b in zip(eda, eda[1:])),
            'fifo_event_count': len(stage['fifo']),
            'fifo_last_ms': max((r[0] for r in stage['fifo']), default=None),
            'fifo_events': stage['fifo'],
            'imu_blocked_reports': sum(bool(r[-1]) for r in stage['imu']),
            'note': 'Zero crossings are retained; causes need controlled physical checks.'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    path = parser.parse_args().evidence
    report = inspect(path)
    output = path.with_suffix('.faults.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    for name, stage in report['stages'].items():
        print(name, json.dumps({k:v for k,v in stage.items() if k not in ('low_runs','fifo_events')}, ensure_ascii=False))
    print(output)
