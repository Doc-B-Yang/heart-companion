"""Temporary RAM-only diagnostic: inputs and open-drain I2C only."""
import time
import sys
import os
from machine import Pin, I2C

print('PLATFORM', sys.implementation, os.uname())
try:
    import config
    print('CONFIG', config.I2C_SDA, config.I2C_SCL, config.I2C_FREQ,
          'IMU', config.MPU6050_ENABLED, 'GSR', config.GSR_ENABLED,
          'RADAR', config.RADAR_ENABLED)
except Exception as e:
    print('CONFIG_ERROR', repr(e))

for n in (25, 26, 27, 32, 33, 21, 22):
    p = Pin(n, Pin.IN, Pin.PULL_UP)
    time.sleep_ms(10)
    up = p.value()
    p.init(Pin.IN, Pin.PULL_DOWN)
    time.sleep_ms(10)
    down = p.value()
    p.init(Pin.IN, pull=None)
    time.sleep_ms(10)
    print('LEVEL', n, 'UP', up, 'DOWN', down, 'FLOAT', p.value())

found = []
scans = []
for freq in (400000, 100000, 50000, 10000):
    try:
        bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=freq, timeout=10000)
        addrs = bus.scan()
        print('SCAN', freq, [hex(x) for x in addrs])
        if len(addrs) > 8:
            print('SCAN_UNTRUSTWORTHY', 'too many responses: possible floating/noisy SDA')
        else:
            scans.append(addrs)
        bus.deinit()
    except Exception as e:
        print('SCAN_ERROR', freq, repr(e))

found = [addr for addr in range(8, 120) if sum(addr in row for row in scans) >= 2]
print('REPEATED_CANDIDATES', [hex(addr) for addr in found])
bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=100000, timeout=10000)
for addr, reg, size in ((0x57, 0xff, 1), (0x68, 0x75, 1),
                        (0x69, 0x75, 1), (0x48, 1, 2)):
    if addr in found:
        try:
            print('REGISTER', hex(addr), hex(reg), list(bus.readfrom_mem(addr, reg, size)))
        except Exception as e:
            print('REGISTER_ERROR', hex(addr), repr(e))
bus.deinit()
print('DONE')
