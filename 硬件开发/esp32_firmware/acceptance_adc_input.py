"""Measure a user-selected A0 reference at two PGA ranges in RAM."""
import json
import time
from machine import I2C, Pin

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
addr = 0x48
saved = bus.readfrom_mem(addr, 1, 2)

def read16(reg):
    b = bus.readfrom_mem(addr, reg, 2)
    return (b[0] << 8) | b[1]

try:
    print(json.dumps({'type': 'adc_input_start', 'addresses': bus.scan(),
                      'note': 'Physical A0 source must be recorded by the host/user; no calibrated voltage reference inferred.'}))
    for config, full_scale in ((0xc383, 4.096), (0xc183, 6.144)):
        codes = []
        times = []
        errors = 0
        for _ in range(64):
            try:
                bus.writeto_mem(addr, 1, bytes((config >> 8, config & 255)))
                start = time.ticks_ms()
                while not read16(1) & 0x8000:
                    if time.ticks_diff(time.ticks_ms(), start) > 100:
                        raise RuntimeError('Conversion timeout')
                    time.sleep_ms(1)
                times.append(time.ticks_diff(time.ticks_ms(), start))
                code = read16(0)
                codes.append(code - 65536 if code >= 32768 else code)
            except Exception as e:
                errors += 1
                print(json.dumps({'type': 'adc_read_error', 'error': repr(e)}))
            time.sleep_ms(2)
        mean = sum(codes) / len(codes) if codes else None
        scale = full_scale / 32768
        result = {'type': 'adc_input_result', 'full_scale_v': full_scale,
                  'successes': len(codes), 'errors': errors, 'codes': codes,
                  'ready_ms_min_max': [min(times), max(times)] if times else None}
        if codes:
            result.update({'mean_v': mean * scale, 'min_v': min(codes) * scale,
                           'max_v': max(codes) * scale,
                           'std_v': (sum((v-mean)**2 for v in codes) / len(codes))**0.5 * scale,
                           'saturated': sum(abs(v) >= 32760 for v in codes)})
        print(json.dumps(result))
finally:
    bus.writeto_mem(addr, 1, saved)
    bus.deinit()
