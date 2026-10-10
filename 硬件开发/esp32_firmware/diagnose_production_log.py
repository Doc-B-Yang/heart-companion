"""Replay saved schema=2 packets through quality checks only, never inference."""
import argparse
import json
import sys
from pathlib import Path
from collections import Counter
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / '心理助手Agent-1' / 'hrv_companion_agent(1)'))
from hrv_companion_agent.core.sensor_pipeline import SensorPipeline


def run(path):
    pipeline = SensorPipeline()
    reasons = Counter()
    packets = quality_windows = 0
    channels = {k: {'pass': 0, 'fail': 0, 'reasons': Counter()} for k in ('ppg','gsr','imu')}
    last = None
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get('schema') != 2 or row.get('seq') is None:
            continue
        flow = pipeline.process(row)
        last = flow
        packets += 1
        quality_windows += int(flow['quality']['quality_ok'])
        reasons.update(flow['quality']['reasons'])
        for k,c in channels.items():
            current = flow['quality']['channels'].get(k)
            if current:
                c['pass' if current['packet_ok'] else 'fail'] += 1
                c['reasons'].update(current['reasons'])
    report = {'source': str(path.resolve()), 'packets': packets,
              'quality_pass_packets': quality_windows, 'reasons': dict(reasons),
              'channels': channels, 'last_quality': last['quality'] if last else None,
              'duplicates': pipeline.duplicates, 'gaps': pipeline.gaps,
              'classification_performed': False, 'labels_written': False, 'baseline_written': False}
    output = path.with_suffix('.channels.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k != 'last_quality'}, ensure_ascii=False))
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    run(parser.parse_args().evidence)
