"""Read-only ADS address/line-order check, using open-drain I2C."""
from machine import Pin, I2C, SoftI2C

for mode in ('hardware_normal', 'software_normal', 'software_reversed'):
    if mode == 'hardware_normal':
        bus = I2C(0, scl=Pin(22), sda=Pin(21), freq=400000)
    elif mode == 'software_normal':
        bus = SoftI2C(scl=Pin(22), sda=Pin(21), freq=10000)
    else:
        bus = SoftI2C(scl=Pin(21), sda=Pin(22), freq=10000)
    print('MODE', mode)
    for addr in (0x48, 0x49, 0x4a, 0x4b):
        try:
            first = bus.readfrom_mem(addr, 1, 2)
            second = bus.readfrom_mem(addr, 1, 2)
            print('CONFIG_REPLY', hex(addr), list(first), list(second))
        except OSError as e:
            print('NO_REPLY', hex(addr), repr(e))
    if mode == 'hardware_normal':
        bus.deinit()

Pin(21, Pin.IN, pull=None)
Pin(22, Pin.IN, pull=None)
print('ADS_PROBE_DONE')
