"""Compare RAM pacing stages against independently received BLE packets."""
import argparse
import json
from pathlib import Path
from ble_transport import PACKET_FIELDS


def identity(packet):
    return packet['device_id'], packet['boot_id'], packet['seq']


def analyse(board_rows, receiver_rows):
    usb = [row for row in board_rows if row.get('schema') == 2 and row.get('seq') is not None]
    received = [row['packet'] for row in receiver_rows if row.get('type') == 'received']
    receiver_result = next((row for row in reversed(receiver_rows)
                            if row.get('type') == 'receiver_result'), None)
    host = next((row for row in reversed(board_rows) if row.get('type') == 'host_result'), {})
    complete = any(row.get('type') == 'complete' for row in board_rows) and host.get('complete') is True
    restored = all(host.get(key) is True for key in ('reset_sent', 'startup_seen', 'production_packet_seen'))
    ble_map = {}
    duplicate_packets = 0
    for packet in received:
        key = identity(packet)
        duplicate_packets += key in ble_map
        ble_map[key] = packet
    stages = []
    for row in board_rows:
        if row.get('type') != 'ble_stage_result':
            continue
        # Board first-service clock and acquisition clock differ slightly.
        # Exclude a full second at stage boundaries, including crossing frames.
        packets = [p for p in usb if row['start_ms'] + 1000 <= p['device_ms'] <= row['end_ms'] - 1000]
        missing, mismatches = [], []
        for packet in packets:
            other = ble_map.get(identity(packet))
            if other is None:
                missing.append(packet['seq'])
            else:
                differing = [key for key in PACKET_FIELDS if packet.get(key) != other.get(key)]
                if differing:
                    mismatches.append({'seq': packet['seq'], 'fields': differing})
        fraction = row['connected_service_calls'] / max(1, row['service_calls'])
        gaps = sum(b['seq'] != a['seq'] + 1 for a, b in zip(packets, packets[1:]))
        sampler_errors = {key: sum(p.get('errors', {}).get(key, 0) for p in packets)
                          for key in ('ppg', 'fifo', 'imu', 'ads', 'clock')}
        rate_values = [p['sample_rate'] for p in packets]
        usb_dropped = [p['transport']['usb_dropped'] for p in packets]
        result = {**row, 'usb_packets_interior': len(packets), 'usb_sequence_gaps': gaps,
            'matching_ble_packets': len(packets) - len(missing) - len(mismatches),
            'missing_ble_sequences': missing, 'payload_mismatches': mismatches,
            'connected_service_fraction': fraction, 'sampler_errors_interior': sampler_errors,
            'sample_rate_min_hz': min(rate_values) if rate_values else None,
            'sample_rate_max_hz': max(rate_values) if rate_values else None,
            'usb_dropped_interior_change': max(usb_dropped)-min(usb_dropped) if usb_dropped else None}
        result['short_transport_check_passed'] = bool(complete and restored and receiver_result
            and len(packets) >= 20 and fraction >= .98 and not missing and not mismatches
            and not gaps and row['errors'] == 0 and row['dropped'] == 0
            and not any(sampler_errors.values()) and result['usb_dropped_interior_change'] == 0)
        stages.append(result)
    return {'capture_complete': complete, 'production_restored': restored,
        'receiver_completed': receiver_result is not None,
        'receiver_result': receiver_result, 'duplicate_received_packets': duplicate_packets,
        'boundary_exclusion_ms': 1000, 'stages': stages,
        'classification_allowed': False, 'accuracy_validated': False,
        'note': 'Short transport checks only; not physiological or long-term acceptance.'}


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board', type=Path)
    parser.add_argument('receiver', type=Path)
    args = parser.parse_args()
    result = analyse(read(args.board), read(args.receiver))
    result.update(board_source=str(args.board.resolve()), receiver_source=str(args.receiver.resolve()))
    output = args.board.with_suffix('.pacing.json')
    with output.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    for stage in result['stages']:
        print(json.dumps({key: stage[key] for key in ('stage', 'interval_ms', 'burst', 'errors',
            'errno_counts', 'connected_service_fraction', 'usb_packets_interior',
            'matching_ble_packets', 'missing_ble_sequences', 'short_transport_check_passed')}))
    print(output)
