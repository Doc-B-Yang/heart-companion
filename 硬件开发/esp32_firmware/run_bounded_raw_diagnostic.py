"""Supervise RAM diagnostics or --deploy, including Windows driver open hangs.

Pass the same arguments as run_raw_diagnostic.py. A killed capture is incomplete;
the supervisor never claims that the board has been restored.
"""
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


def main():
    args = sys.argv[1:]
    deploying = bool(args and args[0] == '--deploy')
    verifying = bool(args and args[0] == '--verify')
    if deploying or verifying:
        args = args[1:]
    runner = Path(__file__).with_name('deploy_firmware.py' if deploying else
        'verify_production_restore.py' if verifying else 'run_raw_diagnostic.py')
    proc = subprocess.Popen([sys.executable, '-u', str(runner), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', errors='replace')
    events = queue.Queue()

    def read_output():
        for line in proc.stdout:
            events.put(line)
        events.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    start = time.monotonic()
    open_deadline = start + 25
    total_deadline = start + (280 if deploying or verifying or '--worn' in args or '--ble-pacing' in args else 130)
    if '--duration' in args:
        duration = int(args[args.index('--duration') + 1])
        total_deadline = start + min(600, max(30, duration)) + 130
    opened = False
    evidence = None
    reason = None
    output_finished = False
    while not output_finished:
        now = time.monotonic()
        if not opened and now > open_deadline:
            reason = 'serial_open_exceeded_25s'
            break
        if now > total_deadline:
            reason = 'diagnostic_and_restore_deadline_exceeded'
            break
        try:
            line = events.get(timeout=.2)
        except queue.Empty:
            continue
        if line is None:
            output_finished = True
            continue
        print(line, end='', flush=True)
        if line.startswith('EVIDENCE '):
            evidence = Path(line.removeprefix('EVIDENCE ').strip())
        if deploying and line.startswith('BOARD_BACKUP '):
            evidence = Path(line.removeprefix('BOARD_BACKUP ').strip()) / 'deployment.jsonl'
        if line.strip() == 'PORT_OPEN':
            opened = True
    if reason:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass  # Some driver waits require a physical USB disconnect.
        result = {'supervisor_timeout': reason, 'port_open_observed': opened,
                  'child_pid': proc.pid, 'child_exit_observed': proc.poll() is not None,
                  'capture_complete': False, 'board_restore_verified': False,
                  'deployment_mode': deploying,
                  'note': 'Inspect backup and board before further changes.' if deploying else
                          'Inspect original evidence; never accept this run as a baseline.'}
        if evidence is not None:
            with evidence.with_suffix('.supervisor.json').open('x', encoding='utf-8') as out:
                json.dump(result, out, ensure_ascii=False, indent=2)
        print('SUPERVISOR_ABORT', json.dumps(result), flush=True)
        return 1
    return proc.wait(timeout=5)


if __name__ == '__main__':
    sys.exit(main())
