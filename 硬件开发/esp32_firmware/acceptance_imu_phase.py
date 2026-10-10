"""One controlled ten-second IMU phase; injected fresh reference optional."""
import json
import time
from machine import I2C, Pin
from mpu6050 import MPU6050

try:
    TEST_REFERENCE
except NameError:
    TEST_REFERENCE = None
bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
imu = MPU6050(bus)
try:
    imu.begin(rate_hz=100)
    if TEST_REFERENCE is None:
        bias = imu.calibrate_gyro(2)
        gravity = imu.init_gravity(1)
        baseline, noise = imu.calibrate_still(0.8)
        profile = {'gyro_bias': bias, 'gravity': gravity, 'accel_mag_ref': baseline,
                   'static_mag_std_g': noise}
        print(json.dumps({'type': 'imu_reference', 'reference': profile}))
    else:
        imu.set_gyro_bias(tuple(TEST_REFERENCE['gyro_bias']))
        imu._grav = tuple(TEST_REFERENCE['gravity'])
        imu._g_ref = TEST_REFERENCE['accel_mag_ref']
        profile = TEST_REFERENCE
    errors, count, hits = 0, 0, 0
    peaks = [0, 0, 0]
    sums = [0, 0, 0]
    vectors = []
    start = time.ticks_ms()
    report = start
    print(json.dumps({'type': 'imu_phase_start', 'duration_s': 10,
                      'reference_reused': TEST_REFERENCE is not None}))
    while time.ticks_diff(time.ticks_ms(), start) < 10000:
        try:
            moving, values = imu.is_motion(update_ref=False)
            count += 1
            hits += int(moving)
            for j in range(3):
                peaks[j] = max(peaks[j], values[j])
                sums[j] += values[j]
            if count % 20 == 0:
                vectors.append(imu.read())
        except OSError:
            errors += 1
        now = time.ticks_ms()
        if time.ticks_diff(now, report) >= 2000:
            print(json.dumps({'type': 'imu_phase_progress', 'elapsed_s': time.ticks_diff(now,start)/1000,
                              'read_count': count, 'motion_hits': hits, 'errors': errors}))
            report = now
        time.sleep_ms(10)
    print(json.dumps({'type': 'result', 'kind': 'imu_phase', 'duration_s': time.ticks_diff(time.ticks_ms(),start)/1000,
                      'count': count, 'errors': errors, 'motion_hits': hits,
                      'motion_fraction': hits/max(1,count),
                      'max_dev_g_omega_dps_tilt_deg': peaks,
                      'mean_dev_g_omega_dps_tilt_deg': [v/max(1,count) for v in sums],
                      'vectors': vectors, 'reference': profile,
                      'note': 'One user-controlled phase with fixed reference; reaction and action timing are not independently instrumented.'}))
finally:
    bus.deinit()
