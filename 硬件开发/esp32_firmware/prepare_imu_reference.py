"""Inject a freshly measured reference into a local RAM test script."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('reference_log', type=Path)
args = parser.parse_args()
reference = None
for line in args.reference_log.read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get('type') == 'imu_reference':
        reference = row['reference']
if reference is None:
    raise SystemExit('No fresh IMU reference found')
root = Path(__file__).resolve().parent
source = (root / 'acceptance_imu_phase.py').read_text(encoding='utf-8')
target = root / 'diagnostics' / 'acceptance-2026-10-03' / 'imu_with_measured_reference.py'
target.write_text('TEST_REFERENCE = ' + repr(reference) + '\n' + source, encoding='utf-8')
print(target)
