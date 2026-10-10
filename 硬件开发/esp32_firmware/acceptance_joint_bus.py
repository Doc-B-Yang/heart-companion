"""Real PPG/IMU/ADC joint traffic at two bus speeds; RAM only."""
import json
import time
from machine import I2C, Pin
from max30102 import MAX30102
from mpu6050 import MPU6050

def emit(row):
    print(json.dumps(row))

for freq in (400000, 100000):
    bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=freq)
    ppg = MAX30102(bus)
    imu = MPU6050(bus)
    saved = None
    try:
        found = bus.scan()
        emit({'type': 'init', 'freq': freq, 'addresses': found})
        if not all(a in found for a in (0x48, 0x57, 0x68)):
            emit({'type': 'blocked', 'freq': freq, 'reason': 'Missing joint-test device'})
            continue
        if not ppg.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384):
            raise RuntimeError('PPG begin failed')
        imu.begin(rate_hz=100)
        saved = bus.readfrom_mem(0x48, 1, 2)
        bus.writeto_mem(0x48, 1, b'\x42\x83')  # A0, +/-4.096 V, continuous 128 SPS
        time.sleep_ms(20)
        ppg.read_fifo()
        errors = {'ppg': 0, 'imu': 0, 'adc': 0}
        first = {}
        counts = {'ppg': 0, 'imu': 0, 'adc': 0}
        adc = []
        ir_min, ir_max, sat = 262143, 0, 0
        start = time.ticks_ms()
        report = start
        next_adc = start
        while time.ticks_diff(time.ticks_ms(), start) < 20000:
            try:
                samples = ppg.read_fifo()
                counts['ppg'] += len(samples)
                for red, ir in samples:
                    ir_min = min(ir_min, ir)
                    ir_max = max(ir_max, ir)
                    sat += int(ir >= 262100)
            except OSError as e:
                errors['ppg'] += 1
                first.setdefault('ppg', repr(e))
            try:
                imu.read_raw()
                counts['imu'] += 1
            except OSError as e:
                errors['imu'] += 1
                first.setdefault('imu', repr(e))
            now = time.ticks_ms()
            if time.ticks_diff(now, next_adc) >= 40:
                next_adc = now
                try:
                    raw = bus.readfrom_mem(0x48, 0, 2)
                    code = (raw[0] << 8) | raw[1]
                    code = code - 65536 if code >= 32768 else code
                    adc.append(code)
                    counts['adc'] += 1
                except OSError as e:
                    errors['adc'] += 1
                    first.setdefault('adc', repr(e))
            if time.ticks_diff(now, report) >= 5000:
                emit({'type': 'progress', 'freq': freq, 'elapsed_ms': time.ticks_diff(now, start),
                      'counts': counts, 'errors': errors})
                report = now
            time.sleep_ms(5)
        elapsed = time.ticks_diff(time.ticks_ms(), start) / 1000
        mean = sum(adc) / len(adc) if adc else None
        adc_stats = None
        if adc:
            scale = 4.096 / 32768
            adc_stats = {'min_v': min(adc) * scale, 'max_v': max(adc) * scale,
                         'mean_v': mean * scale,
                         'std_v': (sum((x - mean)**2 for x in adc) / len(adc))**0.5 * scale,
                         'at_rail': sum(abs(x) >= 32760 for x in adc)}
        emit({'type': 'result', 'freq': freq, 'duration_s': elapsed,
              'counts': counts, 'errors': errors, 'first_error': first,
              'ppg_samples_per_s': counts['ppg']/elapsed,
              'imu_successful_reads_per_s': counts['imu']/elapsed,
              'adc_successful_reads_per_s': counts['adc']/elapsed,
              'adc_configured_sps': 128, 'adc_stats': adc_stats,
              'ir_min': ir_min if counts['ppg'] else None, 'ir_max': ir_max,
              'saturated_samples': sat,
              'note': 'Transport-read rates for IMU/ADC, not unique sample rates. FIFO overflow counter not used as loss proof.'})
    except Exception as e:
        emit({'type': 'test_error', 'freq': freq, 'error': repr(e)})
    finally:
        try:
            ppg.shutdown()
        except OSError as e:
            emit({'type': 'cleanup_error', 'device': 'ppg', 'error': repr(e)})
        if saved is not None:
            try:
                bus.writeto_mem(0x48, 1, saved)
            except OSError as e:
                emit({'type': 'cleanup_error', 'device': 'adc', 'error': repr(e)})
        bus.deinit()
        time.sleep_ms(100)
