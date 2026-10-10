"""RAM-only BLE pacing crossover around the unchanged production sampler.

No API writes, labels, baseline or inference. Stage summaries are held in RAM
until sampling stops so console output cannot interrupt live acquisition.
"""
import time
import json
import struct
import ble_transport
from serial_transport import USBTransport
import acquisition


class DiagnosticFinished(Exception):
    pass


class MeasuredBLE(ble_transport.BLETransport):
    instance = None
    stages = ((2, 2), (8, 1), (2, 2), (12, 1))
    stage_ms = 30000

    def __init__(self, name='HeartCompanion'):
        super().__init__(name)
        MeasuredBLE.instance = self
        self.first_service = None
        self.started = None
        self.stage_index = 0
        self.summaries = []
        self.deadline_ms = 190000
        self._new_counts()

    def _new_counts(self):
        self.attempts = self.sent = self.completed = 0
        self.errnos = {}
        self.connection_samples = self.service_calls = 0
        self.max_queue_age_ms = 0
        self.enqueued = self.rejected = 0
        self.queue_started = None
        self.before_errors = self.errors
        self.before_dropped = self.dropped

    def enqueue(self, payload):
        accepted = super().enqueue(payload)
        if accepted:
            self.enqueued += 1
            self.queue_started = time.ticks_ms()
        else:
            self.rejected += 1
        return accepted

    def _snapshot(self, elapsed):
        delay, burst = self.stages[self.stage_index]
        self.summaries.append({'type': 'ble_stage_result',
            'stage': self.stage_index, 'interval_ms': delay, 'burst': burst,
            'start_ms': self.start_offset_ms + self.stage_index * self.stage_ms,
            'end_ms': self.start_offset_ms + elapsed,
            'clock_alignment': 'relative_first_service_approx_device_ms',
            'attempts': self.attempts, 'frames_sent': self.sent,
            'messages_completed': self.completed, 'enqueued': self.enqueued,
            'enqueue_rejected': self.rejected,
            'errors': self.errors - self.before_errors,
            'dropped': self.dropped - self.before_dropped,
            'errno_counts': self.errnos, 'max_queue_age_ms': self.max_queue_age_ms,
            'service_calls': self.service_calls,
            'connected_service_calls': self.connection_samples,
            'mtus': list(self.mtus.values()), 'classification_allowed': False})

    def service(self):
        now = time.ticks_ms()
        if self.first_service is None:
            self.first_service = now
        if time.ticks_diff(now, self.first_service) >= self.deadline_ms:
            raise RuntimeError('BLE diagnostic connection/total deadline exceeded')
        if self.started is None and self.connections:
            self.started = now
            self.start_offset_ms = time.ticks_diff(now, self.first_service)
            self._new_counts()
        if self.started is not None:
            elapsed = time.ticks_diff(now, self.started)
            if elapsed >= (self.stage_index + 1) * self.stage_ms:
                self._snapshot(elapsed)
                self.stage_index += 1
                if self.stage_index == len(self.stages):
                    raise DiagnosticFinished()
                # Messages crossing stage boundaries remain visible; host
                # comparisons exclude 1 second at either side of each stage.
                queued_since = self.queue_started
                self._new_counts()
                self.queue_started = queued_since
        self.service_calls += 1
        self.connection_samples += bool(self.connections)
        for conn in tuple(self._exchange_pending):
            self._exchange_pending.discard(conn)
            try:
                self.ble.gattc_exchange_mtu(conn)
            except (OSError, AttributeError):
                pass
        if self._advertise_pending:
            self._advertise_pending = False
            self.advertise()
        delay, burst = self.stages[self.stage_index]
        if not self.queue or time.ticks_diff(now, self._last_send) < delay:
            return
        self._last_send = now
        if self.queue_started is not None:
            self.max_queue_age_ms = max(self.max_queue_age_ms,
                time.ticks_diff(now, self.queue_started))
        for _ in range(burst):
            if not self.queue:
                break
            frame = b'HC' + struct.pack('<HBB', self.msg_id, self.index, self.count) + self.queue[self.index * self.payload_size:(self.index + 1) * self.payload_size]
            try:
                for conn in tuple(self.connections):
                    self.attempts += 1
                    self.ble.gatts_notify(conn, self.handle, frame)
                    self.sent += 1
                self.index += 1
                if self.index == self.count:
                    self.queue = b''
                    self.completed += 1
            except OSError as exc:
                self.errors += 1
                key = str(exc.args[0]) if exc.args else 'missing_errno'
                if key not in self.errnos and len(self.errnos) >= 8:
                    key = 'other'
                self.errnos[key] = self.errnos.get(key, 0) + 1
                break


usb_instances = []
original_usb_init = USBTransport.__init__


def track_usb(self, *args, **kwargs):
    original_usb_init(self, *args, **kwargs)
    usb_instances.append(self)


USBTransport.__init__ = track_usb
ble_transport.BLETransport = MeasuredBLE
print(json.dumps({'type': 'ble_diagnostic', 'ram_only': True,
    'stages': MeasuredBLE.stages, 'stage_ms': MeasuredBLE.stage_ms,
    'production_sampler': True, 'classification_allowed': False}))
completed = False
try:
    acquisition.main()
except DiagnosticFinished:
    completed = True
finally:
    # Finish outstanding USB evidence before printing the summaries. This is
    # after sampling stops; no live continuity claim covers this drain period.
    drain_start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), drain_start) < 1000:
        for usb in usb_instances:
            usb.service()
        time.sleep_ms(2)
    instance = MeasuredBLE.instance
    if instance is not None:
        instance.close()
        for row in instance.summaries:
            print(json.dumps(row))
    if completed:
        print(json.dumps({'type': 'complete', 'classification_allowed': False,
            'accuracy_validated': False, 'files_modified': False}))
