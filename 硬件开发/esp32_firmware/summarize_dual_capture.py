"""Compare saved live USB/BLE captures without classification or data repair."""
import argparse
import json
import statistics
from pathlib import Path
from ble_receiver import PACKET_FIELDS


def key(p):
    return (p.get('device_id'), p.get('boot_id'), p.get('seq'))


def load(path):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    received = [r for r in rows if r.get('phase') == 'received']
    posted = [r for r in rows if r.get('phase') == 'posted']
    packets = {key(r['packet']): r['packet'] for r in received}
    times = {key(r['packet']): r['received_unix'] for r in received}
    delays = [r['posted_unix'] - times[key(r)] for r in posted if key(r) in times]
    ordered = [r['packet'] for r in received]
    report = {
        'source': str(path.resolve()), 'received': len(received), 'posted': len(posted),
        'unique_packets': len(packets), 'boots': sorted({p['boot_id'] for p in ordered}),
        'sequence_gaps': sum(max(0, b['seq']-a['seq']-1) for a,b in zip(ordered, ordered[1:])
                             if a['boot_id'] == b['boot_id']),
        'api_accepted': sum(r.get('accepted') is True for r in posted),
        'quality_pass': sum(r.get('state', {}).get('quality', {}).get('quality_ok') is True for r in posted),
        'emotions': sorted({r.get('state', {}).get('emotion', '') for r in posted}),
        'receive_to_api_response_ms': {
            'median': statistics.median(delays)*1000 if delays else None,
            'max': max(delays)*1000 if delays else None,
            'scope': 'host receipt to API response; excludes sensor and radio latency'},
        'last_transport': ordered[-1].get('transport') if ordered else None,
    }
    return report, packets, posted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('usb', type=Path)
    parser.add_argument('ble', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    u, usb, up = load(args.usb)
    b, ble, bp = load(args.ble)
    shared = usb.keys() & ble.keys()
    mismatches = {}
    for k in shared:
        fields = [f for f in PACKET_FIELDS if usb[k].get(f) != ble[k].get(f)]
        if fields:
            mismatches[str(k)] = fields
    accepted = [key(r) for r in up+bp if r.get('accepted') is True]
    result = {'usb': u, 'ble': b, 'shared_packets': len(shared),
              'shared_field_mismatches': mismatches,
              'accepted_total': len(accepted), 'accepted_unique': len(set(accepted)),
              'duplicate_acceptances': len(accepted)-len(set(accepted)),
              'unique_received_union': len(usb.keys() | ble.keys()),
              'physiological_accuracy_test': False}
    with args.out.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
