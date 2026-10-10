"""Five-minute real resting PPG + IMU + ADC acquisition, RAM only.

Requires user confirmation of stable finger/electrode placement. Existing
firmware stays unchanged. Unknown IMU state is gated in this diagnostic.
"""
import json
import time
from machine import I2C, Pin
from max30102 import MAX30102
from mpu6050 import MPU6050
from hrv import HRVAnalyzer

DURATION_S = 300
bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
ppg = MAX30102(bus)
imu = MPU6050(bus)
saved_adc = None

def emit(row):
    print(json.dumps(row))

try:
    emit({'type': 'devices', 'addresses': bus.scan(), 'duration_s': DURATION_S})
    imu.begin(rate_hz=100)
    emit({'type': 'imu_calibration', 'note': 'Keep all modules still'})
    bias = imu.calibrate_gyro(2)
    grav = imu.init_gravity(1)
    emit({'type': 'imu_calibrated', 'bias_dps': bias, 'gravity_ref': grav})
    if not ppg.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384):
        raise RuntimeError('PPG begin failed')
    ppg.auto_gain()
    emit({'type': 'gain', 'led_code': ppg.gain_code, 'ir_dc': ppg.gain_dc, 'ok': ppg.gain_ok})
    if not ppg.gain_ok:
        raise RuntimeError('Finger contact/gain validation failed; session not started')
    saved_adc = bus.readfrom_mem(0x48, 1, 2)
    bus.writeto_mem(0x48, 1, b'\x42\x83')
    hrv = HRVAnalyzer(fs=100, window_sec=0, min_peak_ms=450,
                      max_peak_ms=2000, peak_ratio=0.5)
    settle = time.ticks_ms()
    settle_errors = 0
    while time.ticks_diff(time.ticks_ms(), settle) < 10000:
        try:
            for red, ir in ppg.read_fifo():
                hrv.add_sample(ir)
        except OSError:
            settle_errors += 1
        time.sleep_ms(5)
    dc, lp, envelope = hrv.dc, hrv.lp, hrv.amp_env
    hrv.reset()
    hrv.dc, hrv.lp, hrv.amp_env = dc, lp, envelope
    counts = {'ppg': 0, 'imu': 0, 'adc': 0}
    errors = {'ppg': 0, 'imu': 0, 'adc': 0}
    raw_block = []
    adc_block = []
    ir_min, ir_max, ir_sum = 262143, 0, 0
    saturated, dark, imu_motion, imu_unknown = 0, 0, 0, 0
    start = time.ticks_ms()
    report = start
    adc_tick = start
    emit({'type': 'start', 'duration_s': DURATION_S, 'settle_errors': settle_errors})
    while time.ticks_diff(time.ticks_ms(), start) < DURATION_S * 1000:
        moving = True
        try:
            moving, deviations = imu.is_motion()
            counts['imu'] += 1
            imu_motion += int(moving)
        except OSError:
            errors['imu'] += 1
            imu_unknown += 1
        try:
            for red, ir in ppg.read_fifo():
                counts['ppg'] += 1
                raw_block.append(ir)
                ir_min = min(ir_min, ir)
                ir_max = max(ir_max, ir)
                ir_sum += ir
                saturated += int(ir >= 262100)
                dark += int(ir <= 100)
                hrv.add_sample(ir, motion=moving)
        except OSError:
            errors['ppg'] += 1
        now = time.ticks_ms()
        if time.ticks_diff(now, adc_tick) >= 40:
            adc_tick = now
            try:
                raw = bus.readfrom_mem(0x48, 0, 2)
                code = (raw[0] << 8) | raw[1]
                code = code - 65536 if code >= 32768 else code
                adc_block.append(code)
                counts['adc'] += 1
            except OSError:
                errors['adc'] += 1
        if time.ticks_diff(now, report) >= 1000:
            emit({'type': 'sample', 'elapsed_s': time.ticks_diff(now, start)/1000,
                  'metrics': hrv.metrics(), 'counts': counts, 'errors': errors,
                  'imu_motion_reads': imu_motion, 'imu_unknown_reads': imu_unknown,
                  'raw_ir': raw_block, 'adc_a0_codes': adc_block})
            raw_block = []
            adc_block = []
            report = now
        time.sleep_ms(5)
    elapsed = time.ticks_diff(time.ticks_ms(), start)/1000
    clean = hrv._clean_rr()
    emit({'type': 'result', 'duration_s': elapsed, 'counts': counts,
          'errors': errors, 'ppg_samples_per_s': counts['ppg']/elapsed,
          'settle_errors': settle_errors, 'saturated_samples': saturated,
          'dark_samples': dark, 'ir_min': ir_min, 'ir_max': ir_max,
          'ir_mean': ir_sum/max(1, counts['ppg']), 'rr_ms': clean,
          'accepted_interval_sum_s': sum(clean)/1000,
          'metrics': hrv.metrics(), 'raw_ir': raw_block, 'adc_a0_codes': adc_block,
          'gain_led_code': ppg.gain_code, 'gain_dc': ppg.gain_dc,
          'imu_motion_reads': imu_motion, 'imu_unknown_reads': imu_unknown,
          'scope': 'RAM test harness; PPG-derived PRV, no ECG reference. IMU and ADC are polled. Raw samples use nominal 100Hz; transport gaps must be reviewed.'})
finally:
    try:
        ppg.shutdown()
    except OSError:
        pass
    if saved_adc is not None:
        try:
            bus.writeto_mem(0x48, 1, saved_adc)
        except OSError:
            pass
    bus.deinit()
