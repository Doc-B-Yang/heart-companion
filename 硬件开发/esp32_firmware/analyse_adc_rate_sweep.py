"""Describe temporary ADS rate exploration; never declare physiological accuracy."""
import argparse
import json
from pathlib import Path
from analyse_raw_diagnostic import eda_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    groups = {}
    complete = False
    host = None
    for line in args.source.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        kind = row.get('type')
        if kind == 'adc_rate_start':
            groups[row['stage']] = {'sps': row['sps'], 'rows': []}
        elif kind == 'adc_rate_samples':
            groups[row['stage']]['rows'].extend(row['samples'])
        elif kind == 'adc_rate_result':
            groups[row['stage']]['device_result'] = row
        elif kind == 'complete':
            complete = True
        elif kind == 'host_result':
            host = row
    result = {'source': str(args.source.resolve()), 'complete': complete,
              'host_result': host, 'stages': [], 'production_rate_changed': False,
              'note': 'Production multitasking suspended; rate and settling confounded with time. No conductance, label or acceptance.'}
    for stage, group in groups.items():
        settled = [r for r in group['rows'] if r[0] >= 1000]
        result['stages'].append({'stage': stage, 'configured_sps': group['sps'],
                                 'device_result': group.get('device_result'),
                                 'all_samples': eda_summary(group['rows']),
                                 'after_first_1s': eda_summary(settled)})
    with args.source.with_suffix('.analysis.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
