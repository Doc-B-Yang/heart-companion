"""Save a real device session and report progress without dumping waveforms."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
out = root / 'diagnostics' / 'hrv-test-2026-10-01'
out.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
command = [sys.executable, '-m', 'mpremote', 'connect', 'COM6', 'run', str(root / 'hrv_session_device.py')]
result = None
last_progress = -10
with (out / 'session.jsonl').open('w', encoding='utf-8') as log:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding='utf-8', errors='replace', env=env)
    for line in process.stdout:
        log.write(line)
        log.flush()
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(line.rstrip(), flush=True)
            continue
        if row.get('type') == 'sample':
            elapsed = row['elapsed_s']
            if elapsed - last_progress >= 10:
                m = row['metrics']
                print('PROGRESS', round(elapsed, 1),
                      {k: m[k] for k in ('hr', 'sdnn', 'rmssd', 'nn', 'pi', 'jitter', 'sig_quality')},
                      'READ_ERRORS', row['read_errors'], flush=True)
                last_progress = elapsed
        elif row.get('type') == 'result':
            result = row
            (out / 'result.json').write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            brief = {k: v for k, v in row.items() if k not in ('rr_ms', 'raw_ir')}
            print('RESULT', json.dumps(brief, ensure_ascii=False), flush=True)
        else:
            print('STATUS', json.dumps(row, ensure_ascii=False), flush=True)
    code = process.wait()
print('OUTPUT', str(out), flush=True)
if code or result is None:
    sys.exit(code or 1)
