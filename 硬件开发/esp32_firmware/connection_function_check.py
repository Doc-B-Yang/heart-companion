"""RAM-only functional check after bus wiring has passed."""
import time
import sys
from machine import I2C, Pin

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
found = bus.scan()
print('DEVICES', [hex(x) for x in found])

if 0x57 in found:
    from max30102 import MAX30102
    ppg = MAX30102(bus)
    try:
        print('PPG_BEGIN', ppg.begin(sample_rate=100, led_current=0x1f, avg=1, adc_range=16384))
        print('PPG_TEMP_C', ppg.read_temp())
        total = 0
        low, high, sat = 262143, 0, 0
        start = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), start) < 2000:
            for red, ir in ppg.read_fifo():
                total += 1
                low = min(low, ir)
                high = max(high, ir)
                sat += int(ir >= 262143)
            time.sleep_ms(10)
        print('PPG_FIFO', total, 'IR_MIN_MAX', low, high, 'SAT', sat, 'OVERFLOW', ppg.ovf_count)
        print('PPG_TEMP_C', ppg.read_temp())
    except Exception as e:
        print('PPG_ERROR', repr(e))
        sys.print_exception(e)
    finally:
        ppg.shutdown()

if 0x68 in found:
    from mpu6050 import MPU6050
    imu = MPU6050(bus)
    try:
        print('IMU_ID', hex(imu.who_am_i()))
        imu.begin(accel_range=2, gyro_range=250, dlpf=3, rate_hz=100)
        ok, rows = imu.verify_registers()
        print('IMU_REGISTER_CHECK', ok, rows)
        vals = []
        for _ in range(100):
            vals.append(imu.read())
            time.sleep_ms(10)
        print('IMU_FIRST', vals[0])
        print('IMU_LAST', vals[-1])
        mags = [(v['ax']**2 + v['ay']**2 + v['az']**2)**0.5 for v in vals]
        print('IMU_ACCEL_MAG', min(mags), sum(mags)/len(mags), max(mags))
    except Exception as e:
        print('IMU_ERROR', repr(e))

if 0x48 in found:
    saved = bus.readfrom_mem(0x48, 1, 2)
    try:
        cfg = b'\xc3\x83'  # OS=1, A0 single-ended, +/-4.096V, single shot, 128 SPS.
        values = []
        config_ok = True
        for _ in range(10):
            bus.writeto_mem(0x48, 1, cfg)
            time.sleep_ms(20)
            rb = bus.readfrom_mem(0x48, 1, 2)
            got = (rb[0] << 8) | rb[1]
            config_ok = config_ok and ((got & 0x7fff) == 0x4383) and bool(got & 0x8000)
            raw = bus.readfrom_mem(0x48, 0, 2)
            value = (raw[0] << 8) | raw[1]
            if value >= 32768:
                value -= 65536
            values.append(value)
        print('ADC_CONFIG_AND_READY', config_ok)
        print('ADC_A0_CODES', values)
        print('ADC_A0_VOLTS', [round(x * 4.096 / 32768, 5) for x in values])
        print('ADC_SATURATED', any(abs(x) >= 32760 for x in values))
    except Exception as e:
        print('ADC_ERROR', repr(e))
    finally:
        bus.writeto_mem(0x48, 1, saved)

print('FUNCTION_CHECK_DONE')
