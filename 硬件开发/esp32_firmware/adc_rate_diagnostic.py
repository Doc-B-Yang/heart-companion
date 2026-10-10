"""RAM-only ADS conversion-rate exploration on unchanged A0 wiring."""
import json
import time
from machine import I2C, Pin
from ads1115 import ADS1115

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
saved = bus.readfrom_mem(0x48, 1, 2)
try:
    for stage, sps in enumerate((128, 32, 16, 8, 128)):
        adc = ADS1115(bus, sps=sps)
        start = time.ticks_ms()
        last_report = start
        block = []
        errors = 0
        first_error = None
        count = 0
        print(json.dumps({'type': 'adc_rate_start', 'stage': stage, 'sps': sps,
                          'physical_state': 'electrodes_connected_separated_unworn'}))
        adc.start()
        while time.ticks_diff(time.ticks_ms(), start) < 10000:
            try:
                value = adc.poll()
                if value is not None:
                    now = time.ticks_ms()
                    count += 1
                    block.append([time.ticks_diff(now, start), value[0], value[1]])
                    adc.start()
            except OSError as exc:
                errors += 1
                first_error = first_error or repr(exc)
                adc.pending = False
                adc.start()
            now = time.ticks_ms()
            if time.ticks_diff(now, last_report) >= 500:
                print(json.dumps({'type': 'adc_rate_samples', 'stage': stage,
                                  'sps': sps, 'samples': block}))
                block = []
                last_report = now
            time.sleep_ms(1)
        if block:
            print(json.dumps({'type': 'adc_rate_samples', 'stage': stage,
                              'sps': sps, 'samples': block}))
        print(json.dumps({'type': 'adc_rate_result', 'stage': stage, 'sps': sps,
                          'count': count, 'errors': errors, 'first_error': first_error}))
finally:
    bus.writeto_mem(0x48, 1, saved)
    bus.deinit()
print(json.dumps({'type': 'complete', 'production_settings_changed': False,
                  'note': 'ADC output-rate comparison, not calibrated conductance or acceptance.'}))
