"""USB/BLE -> local webpage API. No inference duplicates or automatic LLM calls.

python gateway.py --port COM6
python gateway.py --ble --scan
python gateway.py --ble --address <address printed by --scan>
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '.runtime'))
from ble_receiver import FrameAssembler, SERVICE_UUID, DATA_UUID


class PacketQueue:
    """Keep serial/BLE reception independent of slow HTTP; never replay backlog."""
    def __init__(self, maxsize=3, max_age=2):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.max_age = max_age
        self.dropped = self.expired = 0

    def put(self, packet, now=None):
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait((time.monotonic() if now is None else now, packet))

    async def get(self):
        while True:
            received, packet = await self.queue.get()
            if time.monotonic() - received <= self.max_age:
                return packet
            self.expired += 1


def record_received(args, packet):
    if getattr(args, 'record', None) and isinstance(packet, dict) and packet.get('schema') == 2:
        with args.record.open('a', encoding='utf-8') as log:
            log.write(json.dumps({'phase':'received', 'received_unix':time.time(),
                'transport':'ble' if args.ble else 'usb', 'packet':packet}, ensure_ascii=False)+'\n')


async def post_packet(client, args, packet):
    if not isinstance(packet, dict) or packet.get('schema') != 2:
        return
    try:
        response = await client.post(args.url.rstrip('/')+'/api/sensor/raw',
                                     json={'user_id':args.user, 'packet':packet})
        response.raise_for_status()
        result = response.json()
        state = result.get('state') or {}
        print(json.dumps({'seq':packet.get('seq'), 'accepted':result.get('accepted'),
                          'emotion':state.get('emotion'), 'quality':state.get('quality')}, ensure_ascii=False), flush=True)
        if getattr(args, 'record', None):
            with args.record.open('a', encoding='utf-8') as log:
                log.write(json.dumps({'phase':'posted', 'posted_unix':time.time(),
                    'transport':'ble' if args.ble else 'usb', 'device_id':packet.get('device_id'),
                    'boot_id':packet.get('boot_id'), 'seq':packet.get('seq'),
                    'accepted':result.get('accepted'), 'state':state}, ensure_ascii=False)+'\n')
    except Exception as exc:
        print('Gateway API unavailable:', str(exc), file=sys.stderr)


async def run_ble(args):
    from bleak import BleakClient, BleakScanner
    from bleak.exc import BleakError
    if args.scan:
        found = await BleakScanner.discover(timeout=8, return_adv=True)
        for device, adv in found.values():
            if SERVICE_UUID in [u.lower() for u in adv.service_uuids] or 'HeartCompanion' in (adv.local_name or ''):
                print(device.address, adv.local_name or device.name)
        return
    if not args.address:
        raise SystemExit('Use --scan, then --address to select the intended device')
    import httpx
    async with httpx.AsyncClient(timeout=2) as http:
        while True:
            assembler = FrameAssembler()
            queue = PacketQueue()
            disconnected = asyncio.Event()
            def notified(_sender, data):
                packet = assembler.add(data)
                if packet is not None:
                    record_received(args, packet)
                    queue.put(packet)
            try:
                async with BleakClient(args.address, disconnected_callback=lambda _:disconnected.set()) as client:
                    await client.start_notify(DATA_UUID, notified)
                    print('BLE connected; receiving schema v2', flush=True)
                    while not disconnected.is_set():
                        try:
                            packet = await asyncio.wait_for(queue.get(), timeout=1)
                            await post_packet(http,args,packet)
                        except asyncio.TimeoutError:
                            continue
            except (BleakError, OSError, asyncio.TimeoutError) as exc:
                print('BLE reconnect:', exc, file=sys.stderr)
            finally:
                print('BLE transport:', json.dumps({'queue_dropped':queue.dropped,
                      'expired':queue.expired, 'incomplete_frames':assembler.dropped}), flush=True)
            await asyncio.sleep(3)


async def run_usb(args):
    import serial
    import httpx
    async with httpx.AsyncClient(timeout=2) as http:
        while True:
            try:
                # Opening this serial port may reset common ESP32 dev boards.
                with serial.Serial(port=None, baudrate=args.baud, timeout=1) as port:
                    port.port = args.port
                    port.dtr = port.rts = False
                    port.open()
                    port.reset_input_buffer()
                    print('USB connected:', args.port, flush=True)
                    queue = PacketQueue()
                    async def receive():
                        oversized = False
                        while True:
                            line = await asyncio.to_thread(port.readline, 8192)
                            if len(line) >= 8192 and not line.endswith(b'\n'):
                                oversized = True
                                continue
                            if oversized:
                                oversized = not line.endswith(b'\n')
                                continue
                            if not line.strip().startswith(b'{'):
                                if line.strip():
                                    print('Board console:', line.decode('utf-8', errors='replace').strip()[:512], flush=True)
                                continue
                            try:
                                packet = json.loads(line)
                            except ValueError:
                                continue
                            record_received(args, packet)
                            queue.put(packet)
                    async def send():
                        while True:
                            await post_packet(http, args, await queue.get())
                    reader = asyncio.create_task(receive())
                    sender = asyncio.create_task(send())
                    try:
                        done, _ = await asyncio.wait((reader, sender), return_when=asyncio.FIRST_EXCEPTION)
                        for task in done:
                            task.result()
                    finally:
                        reader.cancel(); sender.cancel()
                        await asyncio.gather(reader, sender, return_exceptions=True)
                        print('USB transport:', json.dumps({'queue_dropped':queue.dropped,
                              'expired':queue.expired}), flush=True)
            except serial.SerialException as exc:
                print('USB reconnect:', exc, file=sys.stderr)
                await asyncio.sleep(3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ble', action='store_true')
    ap.add_argument('--scan', action='store_true')
    ap.add_argument('--address')
    ap.add_argument('--port',default='COM6')
    ap.add_argument('--baud',type=int,default=115200)
    ap.add_argument('--url',default='http://127.0.0.1:8000')
    ap.add_argument('--user',default='user001')
    ap.add_argument('--seconds',type=float,help='Stop after a bounded diagnostic capture')
    ap.add_argument('--record',type=Path,help='Append raw packets and API results to JSONL')
    args=ap.parse_args(argv)
    if args.seconds is not None and args.seconds <= 0:ap.error('--seconds must be positive')
    if args.record:args.record.parent.mkdir(parents=True,exist_ok=True)
    async def run():
        task = run_ble(args) if args.ble else run_usb(args)
        if args.seconds is not None:
            await asyncio.wait_for(task,timeout=args.seconds)
        else:
            await task
    try:
        asyncio.run(run())
    except asyncio.TimeoutError:
        print('Diagnostic capture completed', flush=True)
    except KeyboardInterrupt:
        print('Gateway stopped')

if __name__ == '__main__':
    main()
