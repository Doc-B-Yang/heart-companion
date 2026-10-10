"""Bounded RAM-only raw capture. No baseline, labels, or emotion predictions.

PPG rows: segment, approximate device_ms, red, ir, motion_blocked.
ADS rows: conversion midpoint device_ms, signed code, volts.
IMU rows: device_ms, accel deviation g, gyro magnitude dps, tilt deg, moving.
PPG timestamps use FIFO order/nominal rate anchored after each discontinuity;
they are not an independent clock or ECG reference. Output overhead is logged.
"""
import time
import json
import gc
from machine import I2C, Pin
import config
from max30102 import MAX30102
from ads1115 import ADS1115
from acquisition import initialize_imu
from serial_transport import USBTransport


def emit(row):
    print(json.dumps(row))


def capture(bus, imu, adc, stage, duration, adc_range, led, gain):
    ppg = MAX30102(bus, config.MAX30102_ADDR)
    if not ppg.begin(sample_rate=100, avg=1, adc_range=adc_range,
                     led_current=led, pulse_width=411):
        raise OSError('PPG identity/initialization failed')
    if gain:
        ppg.auto_gain()
    ppg.flush_fifo()
    emit({'type': 'stage', 'stage': stage, 'duration_s': duration,
          'adc_range_na': adc_range, 'led_code': ppg.led_current,
          'gain_dc': ppg.gain_dc, 'gain_ok': ppg.gain_ok,
          'part_id': ppg.part_id(), 'rev_id': ppg.rev_id(),
          'registers': {str(r): ppg._rreg(r) for r in (8, 9, 10, 12, 13)},
          'worn': bool(globals().get('WORN', False)),
          'nominal_hz': 100, 'time_reference': 'nominal_fifo_clock'})
    emit({'type': 'adc_config', 'stage': stage, 'config_word': adc.cfg,
          'read_schedule_hz': 20, 'full_scale_v': adc.scale * 32768})
    start = previous = report = time.ticks_ms()
    elapsed = 0
    sample_t = None
    segment = total = seq = 0
    last_imu = last_ads = -10000
    last_imu_good = -10000
    moving, deviations = True, (0, 0, 0)
    errors = {'ppg': 0, 'imu': 0, 'ads': 0, 'fifo': 0, 'clock': 0}
    fifo_events = []
    usb = USBTransport(8192)
    flush_pending = False
    recovery_errors = 0
    max_loop = max_output = max_raw_rows = max_payload_bytes = 0
    raw, eda, motion_rows = [], [], []
    adc.pending = False
    while elapsed < int(duration * 1000):
        usb.service()
        # Raw two-channel rows are larger than production feature packets.
        # Drain two bounded 32-byte chunks per loop to avoid producer backlog;
        # each write remains bounded, and loop/FIFO/motion checks stay active.
        usb.service()
        now = time.ticks_ms()
        dt = time.ticks_diff(now, previous)
        previous = now
        elapsed = time.ticks_diff(now, start)
        max_loop = max(max_loop, dt)
        if elapsed - last_imu >= 10:
            last_imu = elapsed
            try:
                moving, deviations = imu.is_motion()
                last_imu_good = elapsed
            except OSError:
                moving = True
                errors['imu'] += 1
        blocked = moving or elapsed - last_imu_good > config.IMU_MAX_AGE_MS
        # A delayed diagnostic output leaves motion during queued samples unknown.
        blocked = blocked or dt > config.IMU_MAX_AGE_MS
        try:
            if flush_pending:
                ppg.flush_fifo()
                flush_pending = False
            batch = ppg.read_fifo()
            if ppg.last_overflow:
                errors['fifo'] += 1
                # Preserve actual counter/pointers; do not infer lost sample count.
                fifo_events.append([elapsed, ppg.last_overflow,
                                    ppg._rreg(4), ppg._rreg(6), dt])
                raise OSError('FIFO overflow')
            if batch:
                latest = sample_t + 10 * len(batch) if sample_t is not None else elapsed
                if sample_t is not None and abs(latest - elapsed) > config.PPG_CLOCK_DRIFT_MS:
                    errors['clock'] += 1
                    raise OSError('FIFO clock drift')
                if sample_t is None:
                    sample_t = elapsed - 10 * len(batch)
                for red, ir in batch:
                    sample_t += 10
                    raw.append([segment, round(sample_t, 1), red, ir, int(blocked)])
                total += len(batch)
        except OSError:
            errors['ppg'] += 1
            segment += 1
            sample_t = None
            flush_pending = True
            try:
                ppg.flush_fifo()
                flush_pending = False
            except OSError:
                recovery_errors += 1
        try:
            if adc.pending:
                value = adc.poll()
                if value is not None:
                    stamp = time.ticks_diff(adc.started, start)
                    mid = (stamp + elapsed) / 2.0
                    eda.append([round(mid, 1), value[0], round(value[1], 6)])
            if not adc.pending and elapsed - last_ads >= 50:
                adc.start()
                last_ads = elapsed
        except OSError:
            errors['ads'] += 1
            adc.pending = False
        if time.ticks_diff(now, report) >= 200 and not usb.pending:
            motion_rows.append([elapsed] + [round(v, 4) for v in deviations] + [int(blocked)])
            seq += 1
            out_start = time.ticks_ms()
            payload = json.dumps({'type': 'raw', 'stage': stage, 'seq': seq, 'elapsed_ms': elapsed,
                  'ppg': raw, 'eda': eda, 'imu': motion_rows, 'fifo_events': fifo_events}, separators=(',', ':'))
            max_raw_rows = max(max_raw_rows, len(raw))
            max_payload_bytes = max(max_payload_bytes, len(payload.encode('utf-8')) + 1)
            if not usb.send(payload):
                raise RuntimeError('Diagnostic raw output exceeded bounded buffer')
            max_output = max(max_output, time.ticks_diff(time.ticks_ms(), out_start))
            raw, eda, motion_rows = [], [], []
            fifo_events = []
            report = now
            gc.collect()
        time.sleep_ms(2)
    # Flush queued text after sampling stops, with an explicit host-output bound.
    drain_started = time.ticks_ms()
    while usb.pending:
        if time.ticks_diff(time.ticks_ms(), drain_started) > 3000:
            raise RuntimeError('Diagnostic output did not drain')
        usb.service()
        time.sleep_ms(1)
    if raw or eda:
        seq += 1
        emit({'type': 'raw', 'stage': stage, 'seq': seq, 'elapsed_ms': elapsed,
              'ppg': raw, 'eda': eda, 'imu': motion_rows, 'fifo_events': fifo_events})
    emit({'type': 'result', 'stage': stage, 'elapsed_ms': elapsed,
          'samples': total, 'observed_hz': total / (elapsed / 1000.0),
          'errors': errors, 'max_loop_gap_ms': max_loop,
          'max_output_ms': max_output, 'free_heap': gc.mem_free(),
          'max_raw_buffer_rows': max_raw_rows, 'max_payload_bytes': max_payload_bytes,
          'usb_service_calls_per_loop': 2,
          'fifo_recovery_errors': recovery_errors,
          'output_mode': 'bounded_chunks', 'usb_errors': usb.dropped,
          'classification_allowed': False})
    ppg.shutdown()
    gc.collect()


bus = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL), sda=Pin(config.I2C_SDA),
          freq=config.I2C_FREQ)
emit({'type': 'devices', 'addresses': bus.scan(), 'mode': 'raw_diagnostic',
      'radar_enabled': False})
imu = initialize_imu(bus)
adc = ADS1115(bus, config.ADS1115_ADDR, config.ADS1115_PGA,
              globals().get('ADS_SPS_OVERRIDE') or config.ADS1115_SPS, config.GSR_ADC_CHANNEL)
if globals().get('ADC_MULTISENSOR_CROSSOVER', False):
    stages = [('unworn-ads128-before', 12, 16384, 31, False),
              ('unworn-ads32', 12, 16384, 31, False),
              ('unworn-ads128-after', 12, 16384, 31, False)]
elif globals().get('WORN', False):
    stages = [('worn-16384-contact-check', globals().get('CAPTURE_SECONDS', 75), 16384, 31, True)] if globals().get('SINGLE_RANGE', False) else [('worn-16384', 60, 16384, 31, True),
              ('worn-8192', 30, 8192, 31, True),
              ('worn-4096', 30, 4096, 31, True)]
else:
    stages = [('unworn-dark', 10, 16384, 0, False),
              ('unworn-lit', 20, 16384, 31, False)]
for index, stage in enumerate(stages):
    if globals().get('ADC_MULTISENSOR_CROSSOVER', False):
        adc = ADS1115(bus, config.ADS1115_ADDR, config.ADS1115_PGA,
                     (128, 32, 128)[index], config.GSR_ADC_CHANNEL)
    capture(bus, imu, adc, *stage)
emit({'type': 'complete', 'stages': len(stages), 'classification_allowed': False})
