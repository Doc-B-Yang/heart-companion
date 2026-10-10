"""Bounded passive USB verification; no labels, baseline or device file writes."""
import argparse
import json
import sys
import subprocess
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
import serial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', default='COM6')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=35)
    parser.add_argument('--min-packets', type=int, default=3)
    parser.add_argument('--ble-address', help='Record BLE simultaneously without API writes')
    args = parser.parse_args()
    if not 1 <= args.seconds <= 240 or args.min_packets < 1:
        parser.error('seconds must be 1..240 and min-packets positive')
    port = serial.Serial()
    port.port, port.baudrate, port.timeout = args.port, 115200, .5
    port.dtr = port.rts = False
    packets, startup = [], None
    brownouts = 0
    receiver = receiver_log = None
    receiver_exit = None
    stop_file = args.out.with_suffix('.ble-stop')
    print('EVIDENCE', args.out, flush=True)
    with args.out.open('x', encoding='utf-8') as out:
        port.open()
        print('PORT_OPEN', flush=True)
        try:
            if args.ble_address:
                receiver_log = args.out.with_suffix('.ble.console.log').open('x', encoding='utf-8')
                receiver_path = args.out.with_suffix('.ble.jsonl')
                receiver = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('record_ble_diagnostic.py')),
                    '--address', args.ble_address, '--seconds', str(args.seconds + 20),
                    '--record', str(receiver_path), '--stop-file', str(stop_file)],
                    stdout=receiver_log, stderr=subprocess.STDOUT)
                print('BLE_EVIDENCE', receiver_path, flush=True)
            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                line = port.readline().decode('utf-8', 'replace').strip()
                if not line:
                    continue
                out.write(line + '\n'); out.flush()
                if 'Brownout detector was triggered' in line:
                    brownouts += 1
                try:
                    row = json.loads(line)
                except ValueError:
                    print(line[:160], flush=True)
                    continue
                if not isinstance(row, dict):
                    continue
                if row.get('type') == 'startup':
                    startup = row
                if row.get('schema') == 2 and row.get('seq') is not None:
                    packets.append(row)
                    if len(packets) % 20 == 0:
                        print('PASSIVE_PROGRESS', len(packets), row['seq'], row.get('transport'), flush=True)
                    if len(packets) >= args.min_packets:
                        break
        finally:
            port.close()
            if receiver is not None:
                stop_file.open('x', encoding='utf-8').close()
                try:
                    receiver_exit = receiver.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    receiver.kill()
                    receiver.wait(timeout=5)
                    receiver_exit = 'forced_stop'
                receiver_log.close()
    result = {'startup': startup, 'production_packets': len(packets),
              'observed_brownouts': brownouts,
              'boot_ids': list(dict.fromkeys(r.get('boot_id') for r in packets)),
              'last_seq': packets[-1].get('seq') if packets else None,
              'device_online': {k: packets[-1].get(k) for k in ('ppg','imu','ads')} if packets else {},
              'classification_allowed': False, 'files_modified': False,
              'ble_receiver_exit': receiver_exit}
    with args.out.with_suffix('.summary.json').open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if packets else 1


if __name__ == '__main__':
    sys.exit(main())
