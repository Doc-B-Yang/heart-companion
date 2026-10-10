"""Run one diagnostic in RAM, save evidence, and resume the existing main.py."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

from mpremote.transport_serial import SerialTransport

parser = argparse.ArgumentParser()
parser.add_argument('script', type=Path)
parser.add_argument('--label', required=True)
parser.add_argument('--port', default='COM6')
parser.add_argument('--brief-json', action='store_true', help='Save all samples but display one progress update per ten seconds')
args = parser.parse_args()
out = Path(__file__).resolve().parent / 'diagnostics' / 'acceptance-2026-10-03'
out.mkdir(parents=True, exist_ok=True)
stamp = datetime.now().strftime('%H%M%S')
path = out / (args.label + '-' + stamp + '.txt')
transport = None
entered = False
exit_code = 0
with path.open('w', encoding='utf-8') as log:
    def emit(text):
        log.write(text)
        log.flush()
        print(text, end='', flush=True)

    pending = bytearray()
    last_progress = -10

    def consume(data):
        global last_progress
        if data == b'\x04':
            return
        pending.extend(data)
        if b'\n' not in pending:
            return
        line = bytes(pending).decode('utf-8', 'replace').replace('\r', '')
        pending.clear()
        log.write(line)
        log.flush()
        if args.brief_json:
            try:
                row = json.loads(line)
            except ValueError:
                print(line, end='', flush=True)
                return
            if row.get('type') == 'sample':
                elapsed = row.get('elapsed_s', 0)
                if elapsed - last_progress < 10:
                    return
                last_progress = elapsed
            if row.get('type') == 'result':
                result_path = path.with_suffix('.result.json')
                result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
            brief = {k: v for k, v in row.items() if k not in ('raw_ir', 'raw_motion', 'adc_a0_codes', 'rr_ms', 'vectors', 'codes')}
            print(json.dumps(brief, ensure_ascii=False), flush=True)
        else:
            print(line, end='', flush=True)

    emit('HOST_START ' + datetime.now().isoformat() + '\n')
    emit('SCRIPT ' + str(args.script.resolve()) + '\n')
    try:
        transport = SerialTransport(args.port)
        # Opening CH340 serial may reset ESP32. Let startup finish before Ctrl-C.
        time.sleep(5)
        transport.enter_raw_repl(soft_reset=True)
        entered = True
        transport.exec_raw_no_follow(args.script.read_bytes())
        _, error = transport.follow(30, data_consumer=consume)
        if pending:
            emit(bytes(pending).decode('utf-8', 'replace'))
        if error:
            emit('\nDEVICE_ERROR ' + error.decode('utf-8', 'replace') + '\n')
            exit_code = 1
    except Exception as error:
        emit('\nHOST_ERROR ' + repr(error) + '\n')
        exit_code = 1
    finally:
        if transport is not None:
            if entered:
                try:
                    transport.exit_raw_repl()
                    time.sleep(0.2)
                    transport.serial.write(b'\x04')
                    time.sleep(4)
                    boot = transport.serial.read(transport.serial.in_waiting)
                    emit('\nRESTORED_MAIN_STARTUP\n' + boot.decode('utf-8', 'replace') + '\n')
                except Exception as error:
                    emit('\nRESTORE_ERROR ' + repr(error) + '\n')
                    exit_code = 1
            transport.close()
    emit('HOST_END ' + datetime.now().isoformat() + '\n')
print('EVIDENCE', path, flush=True)
sys.exit(exit_code)
