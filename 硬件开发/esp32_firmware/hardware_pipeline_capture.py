"""Capture actual firmware output and validate the local PC bridge contract."""
import argparse
import json
import math
import time
from pathlib import Path
import serial
import serial.tools.list_ports
from bridge import Bridge, parse_args

parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=90)
parser.add_argument('--port', default='COM6')
parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'diagnostics' / 'fullcheck-2026-10-01')
args = parser.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)
print('PORTS', [(p.device, p.description) for p in serial.tools.list_ports.comports()], flush=True)
bridge = Bridge(parse_args(['--no-agent', '--baseline-file', str(out / 'unused-baseline.json')]))
rows = []
contracts = []
invalid = 0
contract_errors = 0
boot = []
processing_ms = []
arrival_seconds = []
empty_increment_fallback_packets = 0
fallback_interval_count = 0
s = serial.Serial()
s.port, s.baudrate, s.timeout = args.port, 115200, 0.5
s.dtr = False
s.rts = False
s.open()
try:
    s.write(b'\r\x03\x03\x02')
    time.sleep(0.5)
    s.reset_input_buffer()
    s.write(b'\x04')
    started = time.monotonic()
    last_progress = started
    with (out / 'serial.txt').open('w', encoding='utf-8') as log, \
         (out / 'raw.jsonl').open('w', encoding='utf-8') as rawlog, \
         (out / 'contracts.jsonl').open('w', encoding='utf-8') as contractlog:
        while time.monotonic() - started < args.seconds:
            text = s.readline().decode('utf-8', 'replace').strip()
            if not text:
                continue
            log.write(text + '\n')
            if text.startswith('{'):
                try:
                    raw = json.loads(text)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                rows.append(raw)
                arrival_seconds.append(time.monotonic() - started)
                if raw.get('ibi_new') == [] and raw.get('ibi'):
                    empty_increment_fallback_packets += 1
                    fallback_interval_count += len(raw['ibi'])
                rawlog.write(json.dumps(raw, ensure_ascii=False) + '\n')
                try:
                    processing_start = time.perf_counter()
                    packet, meta = bridge.build_packet(raw)
                    processing_ms.append((time.perf_counter() - processing_start)*1000)
                    expected = {'t', 'hr', 'ibi', 'sdnn', 'rmssd', 'pnn50', 'lf_hf', 'rr', 'sig_quality'}
                    valid = set(packet) == expected and all(
                        isinstance(packet[k], (int, float)) and math.isfinite(packet[k])
                        for k in expected - {'ibi'})
                    valid = valid and isinstance(packet['ibi'], list) and all(
                        isinstance(x, (int, float)) and math.isfinite(x) for x in packet['ibi'])
                    if not valid:
                        contract_errors += 1
                    contracts.append(packet)
                    contractlog.write(json.dumps({'packet': packet, 'resp_conf': meta['resp_conf']}, ensure_ascii=False) + '\n')
                except Exception as exc:
                    contract_errors += 1
                    print('CONTRACT_ERROR', repr(exc), flush=True)
            else:
                boot.append(text)
                print('BOOT', text, flush=True)
            now = time.monotonic()
            if now - last_progress >= 15:
                r = rows[-1] if rows else {}
                print('LIVE', round(now-started, 1), 'packets', len(rows),
                      {k: r.get(k) for k in ('ppg', 'imu', 'hr', 'nn', 'sig_quality', 'pi', 'moving', 'sat')}, flush=True)
                last_progress = now
finally:
    s.close()

tail = rows[-30:]
gaps = [b-a for a,b in zip(arrival_seconds, arrival_seconds[1:])]
summary = {
    'duration_s': round(time.monotonic() - started, 2),
    'packets': len(rows), 'json_parse_errors': invalid,
    'bridge_contracts': len(contracts), 'bridge_errors': contract_errors,
    'ppg_online_packets': sum(bool(r.get('ppg')) for r in rows),
    'imu_online_packets': sum(bool(r.get('imu')) for r in rows),
    'tail_quality_ge_0_5': sum(float(r.get('sig_quality') or 0) >= 0.5 for r in tail),
    'tail_motion_packets': sum(bool(r.get('moving')) for r in tail),
    'tail_hr_min_max': [min((r.get('hr', 0) for r in tail), default=0), max((r.get('hr', 0) for r in tail), default=0)],
    'last_firmware': rows[-1] if rows else {},
    'last_contract': contracts[-1] if contracts else {},
    'boot_messages': boot,
    'packet_arrival_gap_min_max_s': [min(gaps), max(gaps)] if gaps else None,
    'bridge_processing_mean_max_ms': [sum(processing_ms)/len(processing_ms), max(processing_ms)] if processing_ms else None,
    'empty_increment_fallback_packets': empty_increment_fallback_packets,
    'fallback_interval_count': fallback_interval_count,
    'latency_scope': 'Local packet conversion duration and USB arrival spacing; no source-time synchronization, Agent/model/TTS/APP latency or BLE measured.',
    'scope': 'Actual USB firmware output and local bridge parsing only; no Agent tick or model/API request.'
}
(out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print('SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
print('OUTPUT', str(out), flush=True)
