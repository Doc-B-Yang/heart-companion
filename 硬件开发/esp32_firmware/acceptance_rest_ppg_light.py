"""Five-minute raw acquisition; cumulative HRV computation stays on the PC.

RAM only. No board files are changed. Records PPG motion labels and real
read counts; does not use the driver's overflow accumulator as lost samples.
"""
import json
import time
from machine import I2C, Pin
from max30102 import MAX30102
from mpu6050 import MPU6050

DURATION_S = 300
bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
ppg = MAX30102(bus)
imu = MPU6050(bus)
saved_adc = None

def emit(row):
    print(json.dumps(row))

try:
    emit({'type': 'devices', 'addresses': bus.scan(), 'duration_s': DURATION_S,
          'method': 'Raw acquisition; cumulative HRV computed on host'})
    emit({'type': 'imu_calibration', 'note': 'Keep all modules still'})
    init_errors = []
    for attempt in range(3):
        try:
            imu.begin(rate_hz=100)
            bias = imu.calibrate_gyro(2)
            grav = imu.init_gravity(1)
            break
        except OSError as error:
            init_errors.append(repr(error))
            emit({'type': 'imu_init_retry', 'attempt': attempt+1, 'error': repr(error)})
            if attempt == 2:
                raise
            time.sleep_ms(100)
    emit({'type': 'imu_calibrated', 'bias_dps': bias, 'gravity_ref': grav})
    if not ppg.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384):
        raise RuntimeError('PPG begin failed')
    ppg.auto_gain()
    emit({'type': 'gain', 'led_code': ppg.gain_code, 'ir_dc': ppg.gain_dc, 'ok': ppg.gain_ok})
    if not ppg.gain_ok:
        raise RuntimeError('Finger contact/gain validation failed; session not started')
    saved_adc = bus.readfrom_mem(0x48, 1, 2)
    bus.writeto_mem(0x48, 1, b'\x42\x83')
    settle = time.ticks_ms()
    report = settle
    warm = []
    settle_errors = 0
    while time.ticks_diff(time.ticks_ms(), settle) < 10000:
        try:
            warm.extend(ir for red, ir in ppg.read_fifo())
        except OSError:
            settle_errors += 1
        now = time.ticks_ms()
        if time.ticks_diff(now, report) >= 1000:
            emit({'type': 'warmup', 'elapsed_s': time.ticks_diff(now,settle)/1000, 'raw_ir': warm})
            warm = []
            report = now
        time.sleep_ms(5)
    if warm:
        emit({'type': 'warmup', 'elapsed_s': 10, 'raw_ir': warm})
    # The warmup transmission may leave a few samples queued; drain before timing.
    ppg.read_fifo()
    counts = {'ppg': 0, 'imu': 0, 'adc': 0}
    errors = {'ppg': 0, 'imu': 0, 'adc': 0}
    raw_block, motion_block, adc_block = [], [], []
    ir_min, ir_max, ir_sum = 262143, 0, 0.0
    saturated, dark, imu_motion, imu_unknown = 0, 0, 0, 0
    last_loop = time.ticks_ms()
    max_loop_gap_ms = 0
    max_report_ms = 0
    start = last_loop
    report = start
    adc_tick = start
    emit({'type': 'start', 'duration_s': DURATION_S, 'settle_errors': settle_errors, 'init_errors': init_errors})
    while time.ticks_diff(time.ticks_ms(), start) < DURATION_S * 1000:
        loop_now = time.ticks_ms()
        max_loop_gap_ms = max(max_loop_gap_ms, time.ticks_diff(loop_now,last_loop))
        last_loop = loop_now
        moving = True  # Failed IMU read means unknown movement, conservatively gated.
        try:
            moving, deviations = imu.is_motion()
            counts['imu'] += 1
            imu_motion += int(moving)
        except OSError:
            errors['imu'] += 1
            imu_unknown += 1
        try:
            rows = ppg.read_fifo()
            counts['ppg'] += len(rows)
            for red, ir in rows:
                raw_block.append(ir)
                motion_block.append(int(moving))
                ir_min = min(ir_min, ir)
                ir_max = max(ir_max, ir)
                ir_sum += ir  # Float prevents growing big-integer accumulation.
                saturated += int(ir >= 262100)
                dark += int(ir <= 100)
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
            reporting_start = time.ticks_ms()
            emit({'type': 'sample', 'elapsed_s': time.ticks_diff(now,start)/1000,
                  'counts': counts, 'errors': errors, 'raw_ir': raw_block,
                  'raw_motion': motion_block, 'adc_a0_codes': adc_block,
                  'imu_motion_reads': imu_motion, 'imu_unknown_reads': imu_unknown,
                  'max_loop_gap_ms': max_loop_gap_ms, 'max_report_ms': max_report_ms})
            max_report_ms = max(max_report_ms,time.ticks_diff(time.ticks_ms(),reporting_start))
            raw_block, motion_block, adc_block = [], [], []
            report = now
        time.sleep_ms(5)
    elapsed = time.ticks_diff(time.ticks_ms(),start)/1000
    emit({'type': 'result', 'duration_s': elapsed, 'counts': counts, 'errors': errors,
          'ppg_samples_per_s': counts['ppg']/elapsed, 'settle_errors': settle_errors,
          'init_errors': init_errors,
          'saturated_samples': saturated, 'dark_samples': dark, 'ir_min': ir_min,
          'ir_max': ir_max, 'ir_mean': ir_sum/max(1,counts['ppg']),
          'raw_ir': raw_block, 'raw_motion': motion_block, 'adc_a0_codes': adc_block,
          'gain_led_code': ppg.gain_code, 'gain_dc': ppg.gain_dc,
          'imu_motion_reads': imu_motion, 'imu_unknown_reads': imu_unknown,
          'max_loop_gap_ms': max_loop_gap_ms, 'max_report_ms': max_report_ms,
          'scope': 'Raw RAM acquisition. Per-batch IMU labels, not independently synchronized sample timestamps. No clinical reference.'})
finally:
    try:
        ppg.shutdown()
    except OSError:
        pass
    if saved_adc is not None:
        try:
            bus.writeto_mem(0x48,1,saved_adc)
        except OSError:
            pass
    bus.deinit()
