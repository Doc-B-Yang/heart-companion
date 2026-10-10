"""Bounded BLE evidence capture; does not post packets to API or classify."""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
from bleak import BleakClient
from bleak.exc import BleakError
from ble_receiver import DATA_UUID, FrameAssembler


async def record(args):
    folder = Path(__file__).resolve().parent / 'diagnostics' / 'raw-validation-2026-10-05'
    path = args.record or folder / ('ble-pacing-receiver-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.jsonl')
    deadline = time.monotonic() + args.seconds
    counts = {'packets': 0, 'notifications': 0, 'reconnects': 0,
              'incomplete_frames': 0, 'connections': 0}
    print('BLE_EVIDENCE', path, flush=True)
    with path.open('x', encoding='utf-8') as evidence:
        def emit(row):
            evidence.write(json.dumps(row, ensure_ascii=False) + '\n')
            evidence.flush()
        def running():
            return time.monotonic() < deadline and not (args.stop_file and args.stop_file.exists())
        while running():
            assembler = FrameAssembler()
            disconnected = asyncio.Event()
            def notified(_sender, data):
                counts['notifications'] += 1
                packet = assembler.add(data)
                if packet is not None:
                    counts['packets'] += 1
                    emit({'type': 'received', 'received_unix': time.time(), 'packet': packet})
                    if counts['packets'] % 10 == 0:
                        print('BLE_PROGRESS', counts['packets'], packet.get('seq'), flush=True)
            try:
                async with BleakClient(args.address, timeout=10,
                        disconnected_callback=lambda _: disconnected.set()) as client:
                    await client.start_notify(DATA_UUID, notified)
                    counts['connections'] += 1
                    emit({'type': 'connected', 'unix': time.time(), 'mtu': client.mtu_size})
                    print('BLE_CONNECTED', client.mtu_size, flush=True)
                    while running() and not disconnected.is_set():
                        try:
                            await asyncio.wait_for(disconnected.wait(), timeout=.5)
                        except asyncio.TimeoutError:
                            pass
            except (BleakError, OSError, asyncio.TimeoutError) as exc:
                emit({'type': 'connection_error', 'unix': time.time(), 'error': repr(exc)})
                print('BLE_CONNECTION_ERROR', repr(exc), flush=True)
            finally:
                counts['incomplete_frames'] += assembler.dropped + bool(assembler.parts)
            if running():
                counts['reconnects'] += 1
                await asyncio.sleep(min(2, max(0, deadline - time.monotonic())))
        emit({'type': 'receiver_result', **counts, 'classification_allowed': False})
    print('BLE_END', json.dumps(counts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', required=True)
    parser.add_argument('--seconds', type=float, default=190)
    parser.add_argument('--record', type=Path)
    parser.add_argument('--stop-file', type=Path)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 300:
        parser.error('seconds must be 1..300')
    asyncio.run(record(args))
