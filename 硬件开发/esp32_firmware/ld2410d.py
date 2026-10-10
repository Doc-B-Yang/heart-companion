# -*- coding: utf-8 -*-
"""
ld2410d.py - 24GHz 毫米波雷达 MicroPython 驱动（协议自适应）

支持的两种协议（自动识别）:
  A. HLK-LD2410 标准二进制协议
     上报帧: 头 F4 F3 F2 F1 | 长度(2,小端) | 数据 | 尾 F8 F7 F6 F5
     基础目标信息(数据[0]=0x02, 长13):
       02 AA | 状态 | 运动距离(2) | 运动能量 | 静止距离(2) | 静止能量 | 探测距离(2) | 55 00
  B. 通用 ASCII 文本协议（部分兼容/贴牌 24GHz 雷达，如 LD1125H 系列）
     形如: "distance:98\\r\\n" / "mov, dis=1.23\\r\\n" / "occ, dis=0.85\\r\\n"
     无小数 → 视为 cm; 含小数 → 视为 m

注意: LD2410D 官方默认波特率 115200 (LD2410B 为 256000)。
"""

from machine import UART, Pin


class LD2410D:
    HDR = b'\xf4\xf3\xf2\xf1'
    TLR = b'\xf8\xf7\xf6\xf5'

    S_NONE   = 0x00
    S_MOVING = 0x01
    S_STATIC = 0x02
    S_BOTH   = 0x03
    STATE_NAME = {0: 'none', 1: 'moving', 2: 'static', 3: 'moving+static'}

    def __init__(self, uart_id=2, tx=17, rx=16, baud=115200, out_pin=None):
        self.uart = UART(uart_id, baudrate=baud,
                         tx=Pin(tx), rx=Pin(rx),
                         timeout=50, rxbuf=1024)
        self.out = Pin(out_pin, Pin.IN) if out_pin is not None else None
        self._buf = bytearray()

        # 最新解析结果
        self.state       = 0
        self.mov_dist    = 0
        self.mov_energy  = 0
        self.stat_dist   = 0
        self.stat_energy = 0
        self.det_dist    = 0
        self.eng_gates   = None
        self.mode        = 0

        # 协议追踪
        self.protocol    = None     # 'binary' | 'ascii'
        self.last_line   = ''

        self.frame_count = 0
        self.raw_bytes   = 0

    # ================= 主入口 =================
    def update(self):
        """读取 UART 并解析。返回本次解析到的有效帧数。"""
        n = self.uart.any()
        if n:
            chunk = self.uart.read(n)
            if chunk:
                self._buf.extend(chunk)
                self.raw_bytes += len(chunk)
        if len(self._buf) > 4096:
            self._buf = self._buf[-512:]

        # 若缓冲区出现二进制帧头 → 走二进制解析；否则按 ASCII 行解析
        if self.HDR in bytes(self._buf):
            return self._parse_binary()
        return self._parse_ascii()

    # ================= A. 二进制协议 =================
    def _parse_binary(self):
        frames = 0
        while True:
            i = self._buf.find(self.HDR)
            if i < 0:
                if len(self._buf) > 3:
                    self._buf = self._buf[-3:]
                break
            if i > 0:
                self._buf = self._buf[i:]
            if len(self._buf) < 6:
                break
            length = self._buf[4] | (self._buf[5] << 8)
            total = 4 + 2 + length + 4
            if len(self._buf) < total:
                break
            data = self._buf[6:6 + length]
            tail = self._buf[6 + length:total]
            self._buf = self._buf[total:]
            if tail == self.TLR:
                self.frame_count += 1
                if self._parse_data(data):
                    self.protocol = 'binary'
                    frames += 1
        return frames

    def _parse_data(self, data):
        t = data[0]
        if t == 0x02 and len(data) >= 13:
            self.mode        = 0x02
            self.state       = data[2]
            self.mov_dist    = data[3] | (data[4] << 8)
            self.mov_energy  = data[5]
            self.stat_dist   = data[6] | (data[7] << 8)
            self.stat_energy = data[8]
            self.det_dist    = data[9] | (data[10] << 8)
            return True
        elif t == 0x01 and len(data) >= 13:
            self.mode        = 0x01
            self.state       = data[2]
            self.mov_dist    = data[3] | (data[4] << 8)
            self.mov_energy  = data[5]
            self.stat_dist   = data[6] | (data[7] << 8)
            self.stat_energy = data[8]
            self.det_dist    = data[9] | (data[10] << 8)
            gates = []
            idx = 13
            while idx + 1 < len(data) and len(gates) < 16:
                gates.append((data[idx], data[idx + 1]))
                idx += 2
            self.eng_gates = gates
            return True
        return False

    # ================= B. ASCII 文本协议 =================
    def _parse_ascii(self):
        frames = 0
        while True:
            j = self._buf.find(b'\n')
            if j < 0:
                if len(self._buf) > 128:
                    self._buf = self._buf[-64:]
                break
            line = bytes(self._buf[:j]).strip()
            self._buf = self._buf[j + 1:]
            if not line:
                continue
            if self._parse_line(line):
                self.protocol = 'ascii'
                self.frame_count += 1
                frames += 1
        return frames

    def _parse_line(self, line):
        s = bytes(line).decode('utf-8', 'replace').strip()
        if not s:
            return False
        self.last_line = s
        low = s.lower()

        # 1) 忽略固件报错 / 调试文本（如 "Error: flashDataLen is more than ..."）
        if 'error' in low or 'flash' in low:
            return False

        # 2) 明确的"无人"状态词
        for kw in ('off', 'none', 'clear', 'idle', 'no target', 'no_target'):
            if low == kw or low.startswith(kw):
                self.state = self.S_NONE
                self.det_dist = 0
                self.mov_dist = 0
                self.stat_dist = 0
                return True

        # 3) 明确的"有人"状态词（无距离）
        for kw in ('occupied', 'present', 'on'):
            if low == kw or low.startswith(kw):
                self.state = self.S_MOVING
                return True

        # 4) 含数值 → 提取距离
        num = self._extract_number(s)
        if num is None:
            return False
        # 含小数点 → 视为米; 纯整数 → 视为厘米
        if '.' in s:
            cm = int(round(num * 100.0))
        else:
            cm = int(num)

        # 状态判定
        if low.startswith('mov') or 'moving' in low:
            self.state = self.S_MOVING
        elif low.startswith('occ') or 'static' in low or 'still' in low:
            self.state = self.S_STATIC
        else:
            self.state = self.S_MOVING if cm > 0 else self.S_NONE

        self.det_dist = cm
        if self.state == self.S_MOVING:
            self.mov_dist = cm
            self.stat_dist = 0
        elif self.state == self.S_STATIC:
            self.stat_dist = cm
            self.mov_dist = 0
        return True

    @staticmethod
    def _extract_number(s):
        """提取字符串中最后一个数字(可含小数点)。"""
        cur = ''
        best = None
        for ch in s:
            if ch.isdigit() or ch == '.':
                cur += ch
            else:
                if cur:
                    best = cur
                    cur = ''
        if cur:
            best = cur
        if not best:
            return None
        try:
            return float(best)
        except ValueError:
            return None

    # ================= 便捷属性 =================
    @property
    def present(self):
        return self.state != 0

    def state_name(self):
        return self.STATE_NAME.get(self.state, 'unknown')

    def out_value(self):
        return self.out.value() if self.out else None

    def as_dict(self):
        return {
            'state': self.state,
            'st':    self.state_name(),
            'mov_d': self.mov_dist,
            'mov_e': self.mov_energy,
            'stat_d': self.stat_dist,
            'stat_e': self.stat_energy,
            'det_d': self.det_dist,
            'out':   self.out_value(),
            'proto': self.protocol,
        }
