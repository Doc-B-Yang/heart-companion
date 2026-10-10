"""MicroPython ADS1115 driver: nonblocking single-shot, one read per conversion."""
import time


class ADS1115:
    PGA = {6.144: 0, 4.096: 1, 2.048: 2, 1.024: 3, 0.512: 4, 0.256: 5}
    RATES = {8: 0, 16: 1, 32: 2, 64: 3, 128: 4, 250: 5, 475: 6, 860: 7}

    def __init__(self, i2c, addr=0x48, full_scale=4.096, sps=128, channel=0):
        if full_scale not in self.PGA or sps not in self.RATES or channel not in range(4):
            raise ValueError('Unsupported ADS1115 configuration')
        self.bus, self.addr = i2c, addr
        self.scale = full_scale / 32768.0
        self.cfg = (0x8000 | ((4 + channel) << 12) | (self.PGA[full_scale] << 9)
                    | 0x0100 | (self.RATES[sps] << 5) | 3)
        self.timeout_ms = max(100, 3 * (1000 // sps + 1))
        self.pending = False

    def _read16(self, reg):
        b = self.bus.readfrom_mem(self.addr, reg, 2)
        if len(b) != 2:
            raise OSError('Short ADS1115 register read')
        return b[0] << 8 | b[1]

    def start(self, now_ms=None):
        if self.pending:
            return False
        self.bus.writeto_mem(self.addr, 1, bytes((self.cfg >> 8, self.cfg & 255)))
        self.started = time.ticks_ms() if now_ms is None else now_ms
        self.pending = True
        return True

    def poll(self, now_ms=None):
        if not self.pending:
            return None
        now = time.ticks_ms() if now_ms is None else now_ms
        try:
            if time.ticks_diff(now, self.started) > self.timeout_ms:
                raise OSError('ADS conversion timed out before polling')
            actual = self._read16(1)
            if (actual & 0x7fff) != (self.cfg & 0x7fff):
                raise OSError('ADS configuration changed')
            if not actual & 0x8000:
                if time.ticks_diff(now, self.started) > self.timeout_ms:
                    raise OSError('ADS conversion timeout')
                return None
            code = self._read16(0)
            if code & 0x8000:
                code -= 65536
            self.pending = False
            return code, code * self.scale
        except OSError:
            self.pending = False
            raise
