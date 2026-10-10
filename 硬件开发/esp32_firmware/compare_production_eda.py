"""Offline voltage comparison; never infer emotion or qualify a baseline."""
import argparse
import json
from pathlib import Path
from analyse_raw_diagnostic import eda_summary, np


def summarize(path):
    groups = {}
    packets = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get('schema') != 2:
            continue
        packets.append(row)
        group = groups.setdefault(row['boot_id'], [])
        group.extend([[t, v / .000125, v] for t, v in row.get('eda_samples', [])])
    result = {'source': str(path.resolve()), 'packets': len(packets), 'boots': {}}
    for boot, rows in groups.items():
        summary = eda_summary(rows)
        if rows:
            values = np.asarray([r[2] for r in rows])
            steps = np.diff(values)
            summary.update(mean_v=float(values.mean()),
                           steps_over_30mv=int(np.sum(np.abs(steps) > .03)),
                           adjacent_pairs=len(steps), below_20mv=int(np.sum(values < .02)))
        result['boots'][boot] = summary
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = {'before_electrode_lead_connected_unworn': summarize(args.before),
              'after_electrode_lead_disconnected_unworn': summarize(args.after),
              'classification_performed': False,
              'note': 'Sequential open-circuit comparison, not a calibrated skin-conductance test.'}
    with args.out.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
