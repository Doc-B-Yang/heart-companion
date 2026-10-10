"""Summarize same-time PPG gates without combining independent maxima."""
import argparse
import json
from pathlib import Path


def assess(candidate):
    runs, current = [], []
    breaks = candidate['break_times_ms']
    for row in candidate['metrics']:
        ok = row['sig_quality'] >= .5 and row['pi'] >= .3 and row['nn'] >= 30
        if current and (row['segment_id'] != current[-1]['segment_id']
                        or row['time_ms'] - current[-1]['time_ms'] > 1100
                        or any(current[-1]['time_ms'] < t <= row['time_ms'] for t in breaks)):
            runs.append(current)
            current = []
        if ok:
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    summaries = [{'first_report_ms': run[0]['time_ms'], 'last_report_ms': run[-1]['time_ms'],
                  'report_span_s': (run[-1]['time_ms'] - run[0]['time_ms']) / 1000,
                  'reports': len(run)} for run in runs]
    return {'runs': summaries, 'simultaneously_passing_reports': sum(r['reports'] for r in summaries),
            'longest_passing_report_span_s': max((r['report_span_s'] for r in summaries), default=0),
            'qualified_physiological_window': False,
            'note': 'Reported metric runs only; not backend acceptance or reference accuracy.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('polarity', type=Path, nargs='+')
    paths = parser.parse_args().polarity
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        result = {'source': str(path.resolve()), 'classification_allowed': False,
                  'accuracy_validated': False,
                  'stages': {name: [{key: assess(candidate) for key, candidate in piece.items()}
                                   for piece in pieces] for name, pieces in data['stages'].items()}}
        output = path.with_name(path.stem + '.windows.json')
        with output.open('x', encoding='utf-8') as out:
            json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        print(output)
        print(json.dumps(result['stages'], ensure_ascii=False))
