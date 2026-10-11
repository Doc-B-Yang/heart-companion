"""PPG + IMU + ADS/GSR production sampler; bounded work, USB/BLE schema v2.

Errors/unknown motion break the PPG segment. No on-device emotion diagnosis.
Initialization retries require a restart rather than blocking live acquisition.
"""
import time
import json
import gc
import os
import ubinascii
from machine import I2C, Pin, unique_id
import config
from max30102 import MAX30102
from mpu6050 import MPU6050
from ads1115 import ADS1115
from hrv import HRVAnalyzer


def initialize_ppg(bus):
    ppg = MAX30102(bus, config.MAX30102_ADDR)
    if not ppg.present() or not ppg.begin(
            sample_rate=config.MAX30102_SAMPLE_RATE,
            led_current=config.LED_IR_CURRENT, avg=config.MAX30102_FIFO_AVG,
            adc_range=config.MAX30102_ADC_RANGE):
        raise OSError('MAX30102 not found')
    ppg.auto_gain()
    ppg.flush_fifo()
    return ppg


def initialize_imu(bus):
    imu = MPU6050(bus, config.MPU6050_ADDR, config.MPU6050_INT_PIN)
    identity = imu.who_am_i()
    if identity not in config.MPU6050_ALLOWED_IDS:
        raise ValueError('Unverified MPU identity: 0x%02x' % identity)
    imu.begin(accel_range=2, gyro_range=250, dlpf=config.MPU6050_DLPF,
              rate_hz=config.MPU6050_RATE_HZ)
    imu.motion_threshold_g = config.MPU6050_MOTION_G
    imu.gyro_threshold_dps = config.MPU6050_GYRO_DPS
    imu.tilt_threshold_deg = config.MPU6050_TILT_DEG
    print('MPU static calibration; leave modules stationary')
    imu.calibrate_gyro(config.MPU6050_CALIB_SEC)
    imu.init_gravity(config.MPU6050_GRAVITY_SEC)
    gravity, noise = imu.calibrate_still(0.8)
    if not .8 <= gravity <= 1.2 or noise > .03:
        raise ValueError('MPU calibration is not stationary')
    return imu


def main():
    bus = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL), sda=Pin(config.I2C_SDA),
              freq=config.I2C_FREQ)
    devices = {'ppg': None, 'imu': None, 'ads': None}
    init_errors = {}
    for name, init in (
        ('ppg', initialize_ppg), ('imu', initialize_imu),
        ('ads', lambda b: ADS1115(b, config.ADS1115_ADDR, config.ADS1115_PGA,
                                 config.ADS1115_SPS, config.GSR_ADC_CHANNEL))):
        if (name == 'imu' and not config.MPU6050_ENABLED) or (name == 'ads' and not config.GSR_ENABLED):
            init_errors[name] = 'disabled by configuration'
            continue
        try:
            devices[name] = init(bus)
        except (OSError, ValueError) as exc:
            init_errors[name] = str(exc)
    ble = None
    ble_error = None
    usb = None
    usb_error = None
    if config.USB_REPORT_ENABLED:
        try:
            from serial_transport import USBTransport
            usb = USBTransport(config.USB_TX_BUFFER)
        except (ImportError, OSError, ValueError) as exc:
            usb_error = str(exc)
    if config.BLE_ENABLED:
        try:
            from ble_transport import BLETransport
            ble = BLETransport(config.BLE_NAME)
        except (ImportError, OSError, ValueError) as exc:
            ble_error = str(exc)
    print(json.dumps({'type': 'startup', 'schema': 2, 'init_errors': init_errors,
                      'ble_error': ble_error, 'usb_error': usb_error, 'radar_removed': True}))
    device_id = ubinascii.hexlify(unique_id()).decode()
    boot_id = ubinascii.hexlify(os.urandom(8)).decode()
    hrv = HRVAnalyzer(config.MAX30102_SAMPLE_RATE / config.MAX30102_FIFO_AVG, config.HRV_WINDOW_SEC,
                      config.PPG_MIN_PEAK_MS, config.PPG_MAX_PEAK_MS,
                      config.PPG_PEAK_RATIO)
    led = Pin(config.LED_PIN, Pin.OUT)
    tick = time.ticks_ms()
    uptime = last_report = last_imu = last_ads = seq = samples = 0
    sample_t = None
    last_ppg = last_imu_good = last_adc = -100000
    moving, motion_known = True, False
    devs = (0, 0, 0)
    blocked = True
    last_unsafe = 0
    errors = {'ppg': 0, 'imu': 0, 'ads': 0, 'fifo': 0, 'clock': 0}
    totals = dict(errors)
    eda = []
    ppg_display = []
    display_due = 0
    healthy = {'ppg': False, 'imu': False, 'ads': False}
    expected = config.MAX30102_SAMPLE_RATE / config.MAX30102_FIFO_AVG
    period_ms = 1000.0 / expected
    report_work_ms = 0
    gc_pending = False
    # MPU calibration runs for seconds after MAX starts sampling.
    # Discard that startup backlog before assigning live sample timestamps.
    if devices['ppg'] is not None:
        devices['ppg'].flush_fifo()
    while True:
        current = time.ticks_ms()
        uptime += time.ticks_diff(current, tick)
        tick = current
        if usb:
            usb.service()
        if ble:
            ble.service()
        if uptime - last_imu >= 10:
            if last_imu and uptime - last_imu > config.IMU_MAX_AGE_MS:
                errors['imu'] += 1
                last_unsafe = uptime
                if not blocked:
                    hrv.invalidate()
                    sample_t = None
                blocked = True
            last_imu = uptime
            try:
                if devices['imu'] is None:
                    raise OSError('IMU unavailable')
                moving, devs = devices['imu'].is_motion()
                motion_known = healthy['imu'] = True
                last_imu_good = uptime
            except OSError:
                errors['imu'] += 1
                healthy['imu'] = motion_known = False
                moving = True
        unsafe = moving or not motion_known or uptime - last_imu_good > config.IMU_MAX_AGE_MS
        if unsafe:
            last_unsafe = uptime
            if not blocked:
                hrv.invalidate()
                sample_t = None
            blocked = True
        elif blocked and uptime - last_unsafe >= config.PPG_RECOVERY_MS:
            hrv.invalidate()
            sample_t = None
            blocked = False
        try:
            ppg = devices['ppg']
            if ppg is not None:
                batch = ppg.read_fifo()
                if getattr(ppg, 'last_overflow', 0):
                    errors['fifo'] += 1
                    raise OSError('FIFO overflow: timestamps uncertain')
                if batch:
                    latest = sample_t + period_ms * len(batch) if sample_t is not None else uptime
                    if sample_t is not None and abs(latest - uptime) > config.PPG_CLOCK_DRIFT_MS:
                        errors['clock'] += 1
                        raise OSError('PPG clock mapping drift')
                    if sample_t is None:
                        sample_t = uptime - len(batch) * period_ms
                    for red, ir in batch:
                        sample_t += period_ms
                        # Bounded 25 Hz display samples; HRV still uses every
                        # sensor sample. Never reconstruct a waveform from HR.
                        if sample_t >= display_due:
                            ppg_display.append([hrv.segment_id, int(sample_t), red, ir, int(blocked)])
                            if len(ppg_display) > 28:
                                ppg_display.pop(0)
                            display_due = sample_t + 40
                        if ir >= 262100 or ir <= 100:
                            if not blocked:
                                hrv.invalidate()
                            blocked = True
                            last_unsafe = uptime
                        hrv.add_sample(ir, t_ms=sample_t, motion=blocked)
                    samples += len(batch)
                    last_ppg = uptime
                    healthy['ppg'] = True
        except OSError:
            errors['ppg'] += 1
            healthy['ppg'] = False
            if not blocked:
                hrv.invalidate()
            blocked = True
            last_unsafe = uptime
            sample_t = None
            try:
                devices['ppg'].flush_fifo()
            except (OSError, AttributeError):
                pass
        ads = devices['ads']
        if uptime - last_ppg > config.PPG_MAX_AGE_MS and not blocked:
            hrv.invalidate()
            sample_t = None
            blocked = True
            last_unsafe = uptime
        if ads is not None and config.GSR_ENABLED:
            try:
                value = ads.poll(current)
                if value is not None:
                    code, voltage = value
                    eda.append([uptime, round(voltage, 6)])
                    if len(eda) > 24:
                        eda.pop(0)
                    healthy['ads'] = abs(code) < 32760 and -0.01 <= voltage <= 3.35
                    last_adc = uptime
                if uptime - last_ads >= 1000 // config.GSR_READ_HZ and not ads.pending:
                    last_ads = uptime
                    ads.start(current)
            except OSError:
                errors['ads'] += 1
                healthy['ads'] = False
        if uptime - last_report >= config.REPORT_INTERVAL_MS and last_imu == uptime:
            report_started = time.ticks_ms()
            duration_ms = uptime - last_report
            last_report = uptime
            seq += 1
            m = hrv.metrics()
            fresh = uptime - last_ppg <= config.PPG_MAX_AGE_MS
            continuity = not blocked and fresh and not m.get('pending_dropped') and not any(errors[k] for k in ('ppg', 'imu', 'fifo', 'clock'))
            if not continuity:
                m['ibi_new'], m['ibi_events'] = [], []
            for key in totals:
                totals[key] += errors[key]
            out = dict(m)
            out.update({'schema': 2, 'device_id': device_id, 'boot_id': boot_id,
                'seq': seq, 'device_ms': uptime, 't': uptime / 1000.0,
                'continuity': continuity, 'ppg': healthy['ppg'] and fresh,
                'imu': healthy['imu'] and uptime - last_imu_good <= config.IMU_MAX_AGE_MS,
                'motion_known': motion_known, 'moving': moving,
                'motion_devs': [round(x, 3) for x in devs],
                'ads': healthy['ads'] and uptime - last_adc < 150,
                'eda_samples': eda, 'eda_voltage': eda[-1][1] if eda else None,
                'ppg_samples': ppg_display,
                'ppg_display_hz': 25,
                'sample_rate': round(samples * 1000.0 / duration_ms, 3),
                'expected_rate': expected, 'samples': samples,
                'report_work_ms': report_work_ms,
                'errors': errors, 'error_totals': dict(totals),
                'sig_quality': m['sig_quality'] if continuity else 0.0,
                'transport': {'ble': bool(ble), 'ble_error': ble_error,
                              'ble_dropped': ble.dropped if ble else 0,
                              'ble_errors': ble.errors if ble else 0,
                              'ble_mtu': min(ble.mtus.values()) if ble and ble.mtus else 23,
                              'usb': bool(usb), 'usb_error': usb_error,
                              'usb_chunked': usb.chunked if usb else False,
                              'usb_dropped': usb.dropped if usb else 0}})
            payload = None
            encoded = True
            try:
                payload = json.dumps(out, separators=(',', ':'))
                if usb:
                    usb.send(payload)
                if ble and ble.connections:
                    from ble_transport import encode_packet
                    ble.enqueue(encode_packet(out))
            except MemoryError:
                # A skipped sequence must break the receiving window. Keep
                # sampling alive and reclaim transient report objects below.
                encoded = False
                hrv.invalidate()
                blocked = True
                last_unsafe = uptime
                sample_t = None
                errors['ppg'] += 1
            led.value(1 if continuity and m['nn'] else 0)
            samples, eda = 0, []
            ppg_display = []
            if encoded:
                errors = {k: 0 for k in errors}
            del m, out, payload
            # Run collection on a later, freshly sampled IMU cycle rather
            # than adding its pause to metrics + two JSON encoders.
            gc_pending = True
            report_work_ms = time.ticks_diff(time.ticks_ms(), report_started)
        elif gc_pending and last_imu == uptime:
            gc.collect()
            gc_pending = False
        time.sleep_ms(2)
