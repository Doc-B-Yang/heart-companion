"""Capture A0 in continuous and single-shot modes; physical source recorded by host."""
import json
import time
from machine import I2C, Pin

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
saved = bus.readfrom_mem(0x48, 1, 2)

def read16(reg):
    b = bus.readfrom_mem(0x48, reg, 2)
    return (b[0] << 8) | b[1]

try:
    for mode in ('continuous_128sps', 'single_shot_128sps'):
        bus.writeto_mem(0x48, 1, b'\x42\x83' if mode.startswith('continuous') else b'\x43\x83')
        time.sleep_ms(20)
        codes = []
        block = []
        errors = 0
        first = None
        start = time.ticks_ms()
        report = start
        while time.ticks_diff(time.ticks_ms(), start) < 10000:
            try:
                if mode.startswith('single'):
                    bus.writeto_mem(0x48, 1, b'\xc3\x83')
                    ready_start = time.ticks_ms()
                    while not read16(1) & 0x8000:
                        if time.ticks_diff(time.ticks_ms(), ready_start) > 100:
                            raise RuntimeError('Conversion timeout')
                        time.sleep_ms(1)
                code = read16(0)
                code = code - 65536 if code >= 32768 else code
                codes.append(code)
                block.append(code)
            except Exception as e:
                errors += 1
                first = first or repr(e)
            now = time.ticks_ms()
            if time.ticks_diff(now, report) >= 1000:
                print(json.dumps({'type': 'adc_sample', 'mode': mode,
                                  'elapsed_s': time.ticks_diff(now, start)/1000,
                                  'codes': block, 'errors': errors}))
                block = []
                report = now
            time.sleep_ms(20)
        mean = sum(codes)/len(codes) if codes else 0
        scale = 4.096/32768
        print(json.dumps({'type': 'adc_capture_result', 'mode': mode,
                          'duration_s': time.ticks_diff(time.ticks_ms(), start)/1000,
                          'count': len(codes), 'errors': errors, 'first_error': first,
                          'mean_v': mean*scale,
                          'min_v': min(codes)*scale if codes else None,
                          'max_v': max(codes)*scale if codes else None,
                          'std_v': (sum((x-mean)**2 for x in codes)/max(1,len(codes)))**0.5*scale,
                          'low_le_5mv': sum(x*scale <= 0.005 for x in codes),
                          'rail_codes': sum(abs(x) >= 32760 for x in codes),
                          'codes': block}))
finally:
    bus.writeto_mem(0x48, 1, saved)
    bus.deinit()
