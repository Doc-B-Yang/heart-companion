"""30-second simultaneous PPG/IMU bus reliability check in RAM."""
import time
from machine import I2C, Pin
from max30102 import MAX30102
from mpu6050 import MPU6050

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
print('SCAN', [hex(x) for x in bus.scan()])
ppg = MAX30102(bus)
imu = MPU6050(bus)
print('PPG_INIT', ppg.begin(sample_rate=100, led_current=31, avg=1, adc_range=16384))
print('PPG_TEMP', ppg.read_temp())
imu.begin()
print('IMU_REGISTERS', imu.verify_registers()[0])
# Discard initial PPG samples before the timed test.
ppg.read_fifo()
total = 0
reads = 0
errors_ppg = 0
errors_imu = 0
first_error = None
start = time.ticks_ms()
last = start
try:
    while time.ticks_diff(time.ticks_ms(), start) < 30000:
        try:
            total += len(ppg.read_fifo())
        except OSError as e:
            errors_ppg += 1
            first_error = first_error or repr(e)
        try:
            imu.read_raw()
            reads += 1
        except OSError as e:
            errors_imu += 1
            first_error = first_error or repr(e)
        now = time.ticks_ms()
        if time.ticks_diff(now, last) >= 5000:
            print('PROGRESS', time.ticks_diff(now, start), 'PPG', total, 'IMU', reads,
                  'PPG_ERRORS', errors_ppg, 'IMU_ERRORS', errors_imu)
            last = now
        time.sleep_ms(5)
finally:
    ppg.shutdown()
print('FINAL', 'MS', time.ticks_diff(time.ticks_ms(), start), 'PPG', total,
      'IMU', reads, 'PPG_ERRORS', errors_ppg, 'IMU_ERRORS', errors_imu,
      'FIRST_ERROR', first_error, 'OVERFLOW', ppg.ovf_count)
