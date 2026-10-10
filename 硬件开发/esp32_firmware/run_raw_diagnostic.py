"""Save bounded RAM diagnostic evidence, then reset to unchanged main.py."""
import argparse
import json
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
from mpremote.transport_serial import SerialTransport


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', default='COM6')
    ap.add_argument('--worn', action='store_true')
    ap.add_argument('--single-range', action='store_true', help='75 s worn check at 16384 nA only')
    ap.add_argument('--duration', type=int, help='Bounded single-range display capture, 30..600 seconds')
    ap.add_argument('--adc-rate-sweep', action='store_true', help='RAM-only unworn A0 rate exploration')
    ap.add_argument('--adc-multisensor-crossover', action='store_true', help='RAM-only unworn 128/32/128 SPS with other sensors')
    ap.add_argument('--ads-sps', type=int, choices=(32, 128), help='Temporary RAM-only ADS setting for raw capture')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out-dir', type=Path, help='Evidence folder; default retains the original diagnostic folder')
    ap.add_argument('--ble-pacing', action='store_true', help='RAM-only production sampler BLE pacing crossover')
    ap.add_argument('--ble-address', default='20:50:0D:16:5A:A6', help='Confirmed device for simultaneous BLE diagnostic receiver')
    args = ap.parse_args()
    if args.duration is not None and (not args.single_range or not 30 <= args.duration <= 600):
        ap.error('--duration requires --single-range and 30..600 seconds')
    if args.ble_pacing and (args.worn or args.single_range or args.adc_rate_sweep or args.adc_multisensor_crossover or args.ads_sps is not None):
        ap.error('--ble-pacing cannot be combined with raw sensor modes')
    if args.single_range and not args.worn:
        ap.error('--single-range requires --worn')
    if (args.adc_rate_sweep or args.adc_multisensor_crossover) and (args.worn or args.single_range):
        ap.error('ADC exploration requires unworn state')
    if args.adc_rate_sweep and args.adc_multisensor_crossover:
        ap.error('Select one ADC exploration')
    if args.ads_sps is not None and (args.adc_rate_sweep or args.adc_multisensor_crossover):
        ap.error('--ads-sps cannot override an ADC crossover')
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label):
        ap.error('label must be lowercase ASCII letters, digits, hyphen or underscore')
    root = Path(__file__).resolve().parent
    folder = args.out_dir or root / 'diagnostics' / 'raw-validation-2026-10-05'
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    path = folder / (args.label + '-' + stamp + '.jsonl')
    console = path.with_suffix('.console.log')
    transport = None
    complete = restored = False
    startup_seen = False
    production_packet_seen = False
    status = 0
    pending = bytearray()
    last_progress = {}
    receiver = receiver_log = None
    receiver_stop = path.with_suffix('.ble-stop')
    deadline = time.monotonic() + (args.duration + 100 if args.duration else (230 if args.worn or args.ble_pacing else 80))
    with path.open('x', encoding='utf-8') as evidence, console.open('x', encoding='utf-8') as log:
        def consume(data):
            nonlocal complete
            if time.monotonic() > deadline:
                raise TimeoutError('bounded diagnostic host deadline exceeded')
            pending.extend(data.replace(b'\x04', b''))
            while b'\n' in pending:
                line, _, rest = pending.partition(b'\n')
                pending[:] = rest
                decoded = line.decode('utf-8', 'replace').strip()
                try:
                    row = json.loads(decoded)
                except ValueError:
                    log.write(decoded + '\n'); log.flush()
                    if decoded: print(decoded, flush=True)
                    continue
                if not isinstance(row, dict):
                    raise ValueError('Diagnostic JSON record must be an object')
                evidence.write(json.dumps(row, ensure_ascii=False) + '\n'); evidence.flush()
                kind = row.get('type')
                if kind == 'complete':
                    complete = True
                if args.ble_pacing and row.get('schema') == 2 and row.get('seq') is not None:
                    if row['device_ms'] - last_progress.get('production', -10000) >= 10000:
                        last_progress['production'] = row['device_ms']
                        print(json.dumps({'seq': row['seq'], 'device_ms': row['device_ms'],
                            'transport': row.get('transport')}), flush=True)
                    continue
                if kind == 'raw':
                    stage = row['stage']
                    elapsed = row['elapsed_ms']
                    if elapsed - last_progress.get(stage, -10000) < 10000:
                        continue
                    last_progress[stage] = elapsed
                    print(json.dumps({'stage': stage, 'elapsed_ms': elapsed, 'seq': row['seq']}), flush=True)
                elif kind == 'adc_rate_samples':
                    continue  # Full data is saved; only stage summaries go to stdout.
                else:
                    print(json.dumps(row, ensure_ascii=False), flush=True)
        print('EVIDENCE', str(path), flush=True)
        try:
            if args.ble_pacing:
                receiver_log = path.with_suffix('.ble.console.log').open('x', encoding='utf-8')
                receiver_path = path.with_suffix('.ble.jsonl')
                receiver = subprocess.Popen([sys.executable, '-u', str(root / 'record_ble_diagnostic.py'),
                    '--address', args.ble_address, '--seconds', '250', '--record', str(receiver_path),
                    '--stop-file', str(receiver_stop)], stdout=receiver_log, stderr=subprocess.STDOUT)
                print('BLE_EVIDENCE', receiver_path, flush=True)
            transport = SerialTransport(args.port, timeout=5)
            transport.serial.write_timeout = 5
            transport.use_raw_paste = False
            print('PORT_OPEN', flush=True)
            time.sleep(5)
            transport.serial.write(b'\r\x03\x03\x02')
            time.sleep(1)
            transport.serial.read(transport.serial.in_waiting)
            print('ENTER_RAW_REPL', flush=True)
            transport.enter_raw_repl(soft_reset=False)
            transport.enter_raw_repl(soft_reset=True)
            print('UPLOAD_RAM_DIAGNOSTIC', flush=True)
            if args.ble_pacing:
                source = (root / 'ble_pacing_diagnostic.py').read_bytes()
            elif args.adc_rate_sweep:
                source = (root / 'adc_rate_diagnostic.py').read_bytes()
            else:
                source = ('WORN = %r\nSINGLE_RANGE = %r\nCAPTURE_SECONDS = %r\nADC_MULTISENSOR_CROSSOVER = %r\nADS_SPS_OVERRIDE = %r\n' % (args.worn, args.single_range, args.duration or 75, args.adc_multisensor_crossover, args.ads_sps)).encode() + (root / 'raw_signal_diagnostic.py').read_bytes()
            transport.exec_raw_no_follow(source)
            _, error = transport.follow(15, data_consumer=consume)
            if error:
                log.write(error.decode('utf-8', 'replace')); log.flush()
                raise RuntimeError(error.decode('utf-8', 'replace'))
            if not complete:
                raise RuntimeError('No completion marker: incomplete evidence')
        except Exception as exc:
            status = 1
            log.write('HOST_ERROR ' + repr(exc) + '\n'); log.flush()
            print('HOST_ERROR', repr(exc), flush=True)
        finally:
            if transport is not None:
                try:
                    transport.serial.write(b'\r\x03\x03')
                    time.sleep(.3)
                    transport.exit_raw_repl()
                    transport.serial.write(b'import machine; machine.reset()\r\n')
                    restored = True
                    log.write('RESET_TO_EXISTING_MAIN\n')
                    boot_buffer = bytearray()
                    restore_deadline = time.monotonic() + 20
                    while time.monotonic() < restore_deadline:
                        if not transport.serial.in_waiting:
                            time.sleep(.05)
                            continue
                        boot = transport.serial.read(transport.serial.in_waiting)
                        log.write(boot.decode('utf-8', 'replace')); log.flush()
                        boot_buffer.extend(boot)
                        while b'\n' in boot_buffer:
                            line, _, rest = boot_buffer.partition(b'\n')
                            boot_buffer[:] = rest
                            try:
                                row = json.loads(line)
                            except (ValueError, UnicodeDecodeError):
                                continue
                            if isinstance(row, dict):
                                if row.get('type') == 'startup':
                                    startup_seen = True
                                    print('RESTORED_STARTUP', json.dumps(row), flush=True)
                                if row.get('schema') == 2 and row.get('seq') is not None:
                                    production_packet_seen = True
                        if startup_seen and production_packet_seen:
                            break
                    if not startup_seen or not production_packet_seen:
                        print('RESTORE_UNVERIFIED: missing startup or production packet', flush=True)
                        status = 1
                except Exception as exc:
                    log.write('RESTORE_ERROR ' + repr(exc) + '\n')
                    print('RESTORE_ERROR', repr(exc), flush=True)
                    status = 1
                transport.close()
            if receiver is not None:
                receiver_stop.open('x', encoding='utf-8').close()
                try:
                    receiver_status = receiver.wait(timeout=15)
                    print('BLE_RECEIVER_EXIT', receiver_status, flush=True)
                    if receiver_status != 0:
                        status = 1
                except subprocess.TimeoutExpired:
                    receiver.kill()
                    receiver.wait(timeout=5)
                    print('BLE_RECEIVER_FORCED_STOP: incomplete receiver summary', flush=True)
                    status = 1
                receiver_log.close()
            evidence.write(json.dumps({'type': 'host_result', 'complete': complete,
                                      'reset_sent': restored, 'startup_seen': startup_seen,
                                      'production_packet_seen': production_packet_seen,
                                      'files_modified': False}) + '\n')
    print('DIAGNOSTIC_END', 'complete=' + str(complete), 'reset_sent=' + str(restored), flush=True)
    return status


if __name__ == '__main__':
    sys.exit(main())
