"""Record two minutes after user-confirmed GSR electrode placement."""
import json
import time
from machine import I2C, Pin

bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
saved = bus.readfrom_mem(0x48, 1, 2)
windows = []
for _ in range(4):
    windows.append({'n': 0, 'sum': 0, 'sum_sq': 0, 'min': 32767, 'max': -32768,
                    'low': 0, 'rail': 0})
scale = 4.096/32768
errors = 0
first = None
block = []
try:
    bus.writeto_mem(0x48, 1, b'\x42\x83')
    time.sleep_ms(20)
    start = time.ticks_ms()
    report = start
    print(json.dumps({'type': 'start', 'duration_s': 120, 'configured_sps': 128,
                      'physical_state': 'User just placed GSR electrodes; resting arm, MAX not touched.'}))
    while time.ticks_diff(time.ticks_ms(), start) < 120000:
        now = time.ticks_ms()
        elapsed = time.ticks_diff(now, start)
        window = windows[min(3, elapsed//30000)]
        try:
            raw = bus.readfrom_mem(0x48, 0, 2)
            code = (raw[0] << 8) | raw[1]
            code = code - 65536 if code >= 32768 else code
            window['n'] += 1
            window['sum'] += code
            window['sum_sq'] += code*code
            window['min'] = min(window['min'], code)
            window['max'] = max(window['max'], code)
            window['low'] += int(code*scale <= 0.005)
            window['rail'] += int(abs(code) >= 32760)
            block.append(code)
        except OSError as e:
            errors += 1
            first = first or repr(e)
        if time.ticks_diff(now, report) >= 1000:
            print(json.dumps({'type': 'sample', 'elapsed_s': elapsed/1000,
                              'adc_a0_codes': block, 'errors': errors,
                              'last_v': block[-1]*scale if block else None}))
            block = []
            report = now
        time.sleep_ms(25)
    stats = []
    for i, w in enumerate(windows):
        mean = w['sum']/max(1,w['n'])
        variance = max(0, w['sum_sq']/max(1,w['n']) - mean*mean)
        stats.append({'start_s': i*30, 'end_s': (i+1)*30, 'count': w['n'],
                      'mean_v': mean*scale, 'std_v': variance**0.5*scale,
                      'min_v': w['min']*scale if w['n'] else None,
                      'max_v': w['max']*scale if w['n'] else None,
                      'low_le_5mv': w['low'], 'rail_codes': w['rail']})
    print(json.dumps({'type': 'result', 'kind': 'gsr_contact_settle',
                      'duration_s': time.ticks_diff(time.ticks_ms(),start)/1000,
                      'total_reads': sum(w['n'] for w in windows), 'errors': errors,
                      'first_error': first, 'windows': stats, 'adc_a0_codes': block,
                      'scope': 'Analog output and contact response only; no calibrated skin conductance or emotional inference.'}))
finally:
    bus.writeto_mem(0x48, 1, saved)
    bus.deinit()
