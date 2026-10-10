"""Explicitly probe every 7-bit I2C address, including reserved addresses.

Address-only writes: no register pointers, reset commands or data bytes.
Reserved addresses are diagnostic results, not assumed usable devices.
"""
import time
from machine import I2C, SoftI2C, Pin

results = []
for mode in ('hardware_100k', 'hardware_100k_repeat', 'software_1k'):
    if mode.startswith('hardware'):
        bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=100000, timeout=10000)
    else:
        bus = SoftI2C(scl=Pin(22, Pin.IN, Pin.PULL_UP),
                      sda=Pin(21, Pin.IN, Pin.PULL_UP), freq=1000, timeout=10000)
    found = []
    unexpected = []
    started = time.ticks_ms()
    for addr in range(128):
        try:
            count = bus.writeto(addr, b'', True)
            if count == 0:
                found.append(addr)
            else:
                unexpected.append((addr, count))
        except OSError:
            pass
    results.append(found)
    print('FULL_SCAN', mode, 'RANGE', '0x00..0x7F', 'ACK',
          [hex(addr) for addr in found], 'MS', time.ticks_diff(time.ticks_ms(), started))
    if unexpected:
        print('UNEXPECTED_COUNTS', unexpected)
    if mode.startswith('hardware'):
        bus.deinit()
    time.sleep_ms(50)

stable = sorted(set(results[0]) & set(results[1]) & set(results[2]))
print('STABLE_ALL_MODES', [hex(addr) for addr in stable])
print('RESERVED_ACK', [hex(addr) for addr in stable if addr < 8 or addr >= 0x78])
print('VARIABLE_ACK', [hex(addr) for addr in sorted(set(sum(results, [])) - set(stable))])
Pin(21, Pin.IN, pull=None)
Pin(22, Pin.IN, pull=None)
print('ALL_128_ADDRESSES_CHECKED')
