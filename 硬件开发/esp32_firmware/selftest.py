"""Current hardware self-test; no radar. This is not long-term acceptance."""
import time
import json
from machine import I2C, Pin
import config
from max30102 import MAX30102
from mpu6050 import MPU6050
from ads1115 import ADS1115

def main():
    bus = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL), sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
    result = {'addresses': bus.scan(), 'radar_required': False}
    for name, test in (('max', lambda: MAX30102(bus,config.MAX30102_ADDR).part_id()),
                       ('mpu', lambda: MPU6050(bus,config.MPU6050_ADDR).who_am_i())):
        try:
            result[name] = {'identity': test()}
        except OSError as exc:
            result[name] = {'error': str(exc)}
    try:
        adc = ADS1115(bus,config.ADS1115_ADDR)
        adc.start()
        while True:
            value = adc.poll()
            if value is not None:
                result['ads'] = {'code':value[0], 'A0_voltage':value[1]}
                break
            time.sleep_ms(2)
    except OSError as exc:
        result['ads'] = {'error':str(exc)}
    print(json.dumps(result))

if __name__ == '__main__':
    main()
