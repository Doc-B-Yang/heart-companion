"""ADS1115 internal register/conversion tests; run in RAM via mpremote.

An absent device is reported as BLOCKED, never as proven chip damage.
Keep A0 within the module's supply rails. Known-voltage input tests need
separate user wiring and are not inferred from an arbitrary GSR value.
"""
import time
from machine import I2C, SoftI2C, Pin

bus = None

def read16(addr, reg):
    b = bus.readfrom_mem(addr, reg, 2)
    return (b[0] << 8) | b[1]

def write16(addr, reg, value):
    bus.writeto_mem(addr, reg, bytes((value >> 8, value & 255)))

def test(addr):
    saved = {reg: read16(addr, reg) for reg in (1, 2, 3)}
    print('INITIAL_REGISTERS', hex(addr), saved)
    try:
        # Disable comparator before altering thresholds.
        write16(addr, 1, 0x4383)
        checks = []
        for config in (0x4383, 0x4583, 0x5383):
            write16(addr, 1, config)
            got = read16(addr, 1)
            checks.append((1, config, got, (got & 0x7fff) == config))
        for reg, value in ((2, 0x9000), (3, 0x6000), (2, 0x8000), (3, 0x7fff)):
            write16(addr, reg, value)
            got = read16(addr, reg)
            checks.append((reg, value, got, got == value))
        print('REGISTER_WRITE_READ', checks)
        results = []
        for _ in range(5):
            write16(addr, 1, 0xc383)  # A0, +/-4.096V, 128SPS, single shot.
            started = time.ticks_ms()
            while not (read16(addr, 1) & 0x8000):
                if time.ticks_diff(time.ticks_ms(), started) > 250:
                    raise RuntimeError('ADC conversion did not complete')
                time.sleep_ms(2)
            value = read16(addr, 0)
            if value >= 32768:
                value -= 65536
            results.append((value, value * 4.096 / 32768))
        print('CONVERSIONS_A0', results)
        print('INTERNAL_DIGITAL_CHECK', 'PASS' if all(row[3] for row in checks) else 'FAIL')
        print('ANALOG_ACCURACY', 'NOT_VERIFIED: needs known GND and voltage input tests')
    finally:
        for reg in (2, 3, 1):
            write16(addr, reg, saved[reg])

tested = False
for mode in ('hardware_100k', 'software_1k_internal_pullups'):
    if mode == 'hardware_100k':
        bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=100000)
    else:
        bus = SoftI2C(scl=Pin(22, Pin.IN, Pin.PULL_UP),
                      sda=Pin(21, Pin.IN, Pin.PULL_UP), freq=1000)
    print('MODE', mode)
    responders = []
    for addr in (0x48, 0x49, 0x4a, 0x4b):
        try:
            bus.writeto(addr, b'')
            config = read16(addr, 1)
            print('RESPONSE', hex(addr), 'CONFIG', hex(config))
            responders.append(addr)
        except OSError as exc:
            print('NO_RESPONSE', hex(addr), repr(exc))
    if len(responders) == 1:
        try:
            test(responders[0])
            tested = True
        except Exception as exc:
            print('INTERNAL_TEST_ERROR', repr(exc))
    if mode == 'hardware_100k':
        bus.deinit()
    if tested:
        break
if not tested:
    print('INTERNAL_TEST_BLOCKED', 'No completed register/conversion test; chip damage not proven.')
Pin(21, Pin.IN, pull=None)
Pin(22, Pin.IN, pull=None)
