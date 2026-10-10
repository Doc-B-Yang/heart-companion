"""Bounded BLE reassembly for firmware and Web-compatible HC frames."""
import json
import struct
import time

SERVICE_UUID = '8e7c0001-7f8a-4a2c-9f81-4b9e764d0001'
DATA_UUID = '8e7c0002-7f8a-4a2c-9f81-4b9e764d0001'
PACKET_FIELDS = ('schema', 'device_id', 'boot_id', 'seq', 'device_ms',
    'segment_id', 'continuity', 'ppg', 'imu', 'motion_known', 'moving',
    'motion', 'sig_quality', 'pi', 'sample_rate', 'expected_rate',
    'ibi_events', 'ads', 'eda_samples', 'errors', 'nn', 'hr')


class FrameAssembler:
    def __init__(self, timeout=2.5):
        self.timeout = timeout
        self.reset()
        self.dropped = 0

    def reset(self):
        self.message = None
        self.count = self.started = None
        self.parts = {}

    def add(self, data, now=None):
        now = time.monotonic() if now is None else now
        if not 7 <= len(data) <= 244 or data[:2] != b'HC':
            return None
        mid, index, count = struct.unpack('<HBB', bytes(data[2:6]))
        if not count or index >= count:
            return None
        if self.message != mid or (self.started is not None and now-self.started>self.timeout):
            if self.parts:
                self.dropped += 1
            self.reset()
            self.message, self.count, self.started = mid, count, now
        if count != self.count:
            self.reset()
            self.dropped += 1
            return None
        payload = bytes(data[6:])
        if index in self.parts and self.parts[index] != payload:
            self.reset()
            self.dropped += 1
            return None
        self.parts[index] = payload
        if sum(map(len,self.parts.values())) > 8192:
            self.reset()
            self.dropped += 1
            return None
        if len(self.parts) != count:
            return None
        full = b''.join(self.parts[i] for i in range(count))
        self.reset()
        try:
            result = json.loads(full.decode('utf-8'))
            if isinstance(result, dict) and result.get('hc') == 1:
                values = result.get('v')
                if not isinstance(values, list) or len(values) != len(PACKET_FIELDS):
                    raise ValueError('Invalid HC compact packet')
                result = dict(zip(PACKET_FIELDS, values))
                result['eda_samples'] = [[t + result['device_ms'], voltage]
                                          for t, voltage in result['eda_samples']]
            return result if isinstance(result, dict) else None
        except (ValueError, UnicodeError, TypeError):
            self.dropped += 1
            return None
