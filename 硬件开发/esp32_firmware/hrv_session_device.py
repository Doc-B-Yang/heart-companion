"""One 120-second resting finger PPG session, executed in RAM."""
import time
import json
from machine import I2C, Pin
from max30102 import MAX30102
from hrv import HRVAnalyzer

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
sensor = MAX30102(bus)
print(json.dumps({'type': 'devices', 'addresses': bus.scan()}))
if not sensor.present():
    raise RuntimeError('MAX30102 does not respond')
sensor.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384)
try:
    sensor.auto_gain()
    print(json.dumps({'type': 'gain', 'led': sensor.gain_code,
                      'ir_dc': sensor.gain_dc, 'ok': sensor.gain_ok}))
    if not sensor.gain_ok:
        raise RuntimeError('Finger contact/gain validation failed')
    hrv = HRVAnalyzer(fs=100, window_sec=0, min_peak_ms=450,
                      max_peak_ms=2000, peak_ratio=0.5)
    settle_start = time.ticks_ms()
    settle_errors = 0
    while time.ticks_diff(time.ticks_ms(), settle_start) < 10000:
        try:
            for red, ir in sensor.read_fifo():
                hrv.add_sample(ir)
        except OSError:
            settle_errors += 1
        time.sleep_ms(5)
    dc, lp, envelope = hrv.dc, hrv.lp, hrv.amp_env
    hrv.reset()
    hrv.dc, hrv.lp, hrv.amp_env = dc, lp, envelope
    print(json.dumps({'type': 'start', 'duration_s': 120, 'settle_errors': settle_errors}))
    start = time.ticks_ms()
    report = start
    samples = 0
    errors = 0
    reads = 0
    saturated = 0
    dark = 0
    ir_min, ir_max = 262143, 0
    ir_sum = 0
    raw_block = []
    while time.ticks_diff(time.ticks_ms(), start) < 120000:
        try:
            rows = sensor.read_fifo()
            reads += 1
            for red, ir in rows:
                samples += 1
                ir_min = min(ir_min, ir)
                ir_max = max(ir_max, ir)
                ir_sum += ir
                saturated += int(ir >= 262100)
                dark += int(ir <= 100)
                raw_block.append(ir)
                hrv.add_sample(ir)
        except OSError:
            errors += 1
        now = time.ticks_ms()
        if time.ticks_diff(now, report) >= 1000:
            print(json.dumps({'type': 'sample', 'elapsed_s': time.ticks_diff(now, start)/1000,
                              'metrics': hrv.metrics(), 'samples': samples,
                              'read_errors': errors, 'raw_ir': raw_block}))
            raw_block = []
            report = now
        time.sleep_ms(5)
    elapsed = time.ticks_diff(time.ticks_ms(), start)/1000
    rr = hrv._clean_rr()
    print(json.dumps({'type': 'result', 'duration_s': elapsed,
                      'samples': samples, 'sample_rate_observed': samples/elapsed,
                      'read_errors': errors, 'read_attempts': reads + errors,
                      'saturated_samples': saturated, 'dark_samples': dark,
                      'ir_min': ir_min, 'ir_max': ir_max, 'ir_mean': ir_sum/max(1, samples),
                      'metrics': hrv.metrics(), 'rr_ms': rr, 'raw_ir': raw_block,
                      'gain_led': sensor.gain_code, 'gain_dc': sensor.gain_dc,
                      'settle_errors': settle_errors, 'imu_gating': False}))
finally:
    sensor.shutdown()
