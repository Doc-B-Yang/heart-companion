"""BLE peripheral with bounded, negotiated-MTU fragmented JSON notifications.

Frame: b'HC' + little endian uint16 message_id + uint8 index + uint8 count
       + payload fitting negotiated MTU (14 bytes before negotiation).
"""
import struct
import time
import json

# HC compact JSON v1 carries the exact schema2 inference fields. USB retains
# all legacy diagnostics. Receivers restore names before API ingestion.
PACKET_FIELDS = ('schema', 'device_id', 'boot_id', 'seq', 'device_ms',
    'segment_id', 'continuity', 'ppg', 'imu', 'motion_known', 'moving',
    'motion', 'sig_quality', 'pi', 'sample_rate', 'expected_rate',
    'ibi_events', 'ads', 'eda_samples', 'errors', 'nn', 'hr')


def encode_packet(packet):
    values = [packet.get(key) for key in PACKET_FIELDS]
    values[18] = [[t - packet['device_ms'], voltage]
                  for t, voltage in packet['eda_samples']]
    # Compact display-only IR; segment breaks remain explicit.
    if 'ppg_samples' in packet:
        values.append([[p[1] - packet['device_ms'], p[3], p[0]]
                      for p in packet['ppg_samples']])
    return json.dumps({'hc': 1, 'v': values}, separators=(',', ':'))

SERVICE_UUID = '8e7c0001-7f8a-4a2c-9f81-4b9e764d0001'
DATA_UUID = '8e7c0002-7f8a-4a2c-9f81-4b9e764d0001'


def frames(payload, message_id, payload_size=14):
    if not isinstance(payload_size, int) or not 1 <= payload_size <= 14:
        raise ValueError('HC fragments must fit MTU23')
    data = payload.encode('utf-8') if isinstance(payload, str) else payload
    count = (len(data) + payload_size - 1) // payload_size
    if not 1 <= count <= 255:
        raise ValueError('BLE packet exceeds bounded frame size')
    return [b'HC' + struct.pack('<HBB', message_id & 65535, i, count)
            + data[i * payload_size:(i + 1) * payload_size] for i in range(count)]


class BLETransport:
    def __init__(self, name='HeartCompanion'):
        import bluetooth
        self.ble = bluetooth.BLE()
        self.ble.active(True)
        self.ble.config(mtu=247)
        self.connections = set()
        self.mtus = {}
        self._exchange_pending = set()
        self._advertise_pending = False
        self.name = name
        self.ble.irq(self._irq)
        ((self.handle,),) = self.ble.gatts_register_services(((
            bluetooth.UUID(SERVICE_UUID),
            ((bluetooth.UUID(DATA_UUID), bluetooth.FLAG_NOTIFY),)),))
        self.ble.gatts_set_buffer(self.handle, 244)
        self.queue = b''
        self.index = 0
        self.msg_id = 0
        self.dropped = 0
        self.errors = 0
        self._last_send = time.ticks_ms()
        self.advertise()

    def advertise(self):
        import bluetooth
        service = bytes(bluetooth.UUID(SERVICE_UUID))
        adv = b'\x02\x01\x06' + bytes((17, 7)) + service
        name = self.name.encode()[:26]
        response = bytes((len(name) + 1, 9)) + name
        self.ble.gap_advertise(100000, adv_data=adv, resp_data=response)

    def _irq(self, event, data):
        if event == 1:  # central connected
            self.connections.add(data[0])
            self.mtus[data[0]] = 23
            self._exchange_pending.add(data[0])
        elif event == 2:
            self.connections.discard(data[0])
            self.mtus.pop(data[0], None)
            self._exchange_pending.discard(data[0])
            if not self.connections:
                self.queue = b''
            self._advertise_pending = True
        elif event == 21:  # MTU exchanged
            self.mtus[data[0]] = max(23, min(247, data[1]))

    def enqueue(self, payload):
        if not self.connections:
            return False
        if self.queue:
            self.dropped += 1
            return False
        self.msg_id = (self.msg_id + 1) & 65535
        data = payload.encode('utf-8') if isinstance(payload, str) else payload
        mtu = min(getattr(self, 'mtus', {}).get(c, 23) for c in self.connections)
        self.payload_size = mtu - 9
        count = (len(data) + self.payload_size - 1) // self.payload_size
        if not 1 <= count <= 255:
            self.dropped += 1
            return False
        # Keep one payload, rather than allocating ~100 fragment objects in
        # the report path. Materialize each frame only when it is sent.
        self.queue = data
        self.count = count
        self.index = 0
        return True

    def service(self):
        for conn in tuple(getattr(self, '_exchange_pending', ())):
            self._exchange_pending.discard(conn)
            try:
                self.ble.gattc_exchange_mtu(conn)
            except (OSError, AttributeError):
                pass  # MTU23 remains usable on stacks without peripheral exchange.
        if self._advertise_pending:
            self._advertise_pending = False
            self.advertise()
        now = time.ticks_ms()
        if not self.queue or time.ticks_diff(now, self._last_send) < 8:
            return
        self._last_send = now
        # Frame size is fixed for the entire message, even if MTU changes.
        # MTU23 hardware crossover: two notifications per 2 ms caused errno12
        # retries; one per 8 ms delivered identical packets without retries.
        # Keep each service call bounded and leave time for sensor sampling.
        for _ in range(1):
            if not self.queue:
                break
            frame = b'HC' + struct.pack('<HBB', self.msg_id, self.index, self.count) + self.queue[self.index * self.payload_size:(self.index + 1) * self.payload_size]
            try:
                for conn in tuple(self.connections):
                    self.ble.gatts_notify(conn, self.handle, frame)
                self.index += 1
                if self.index == self.count:
                    self.queue = b''
            except OSError:
                # Retry the same fragment; the receiver tolerates duplicates.
                self.errors += 1
                break

    def close(self):
        self.ble.active(False)
