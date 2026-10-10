"""Compare real reads at bus rates, not just address acknowledgements."""
import time
from machine import Pin, I2C, SoftI2C
from max30102 import MAX30102
from mpu6050 import MPU6050

for kind, rate in (('hardware', 400000), ('hardware', 100000),
                   ('hardware', 50000), ('hardware', 10000), ('software', 10000)):
    bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=rate) if kind == 'hardware' else SoftI2C(scl=Pin(22), sda=Pin(21), freq=rate)
    ppg = MAX30102(bus)
    imu = MPU6050(bus)
    good_ppg = 0
    good_imu = 0
    errors = 0
    first_error = None
    sat = 0
    print('RATE', kind, rate, 'SCAN', [hex(x) for x in bus.scan()])
    try:
        ppg.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384)
        imu.begin()
        started = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), started) < 2000:
            try:
                rows = ppg.read_fifo()
                good_ppg += len(rows)
                sat += sum(1 for _, ir in rows if ir >= 262143)
                imu.read_raw()
                good_imu += 1
            except Exception as e:
                errors += 1
                first_error = first_error or repr(e)
            time.sleep_ms(10)
    except Exception as e:
        errors += 1
        first_error = first_error or repr(e)
    finally:
        try:
            ppg.shutdown()
        except Exception:
            pass
        if kind == 'hardware':
            bus.deinit()
    print('RESULT', kind, rate, 'PPG_SAMPLES', good_ppg, 'IMU_READS', good_imu,
          'ERRORS', errors, 'FIRST_ERROR', first_error, 'SAT', sat)
