# -*- coding: utf-8 -*-
"""
max30102.py - MAX30102 心率/血氧传感器 MicroPython 驱动
适配: ESP32-WROOM-32E, I2C 400kHz, 从机地址 0x57

数据手册寄存器参考 (MAX30102):
  0x00 INT_STATUS_1   0x01 INT_STATUS_2   0x02 INT_ENABLE_1  0x03 INT_ENABLE_2
  0x04 FIFO_WR_PTR    0x05 OVF_COUNTER    0x06 FIFO_RD_PTR   0x07 FIFO_DATA
  0x08 FIFO_CONFIG    0x09 MODE_CONFIG    0x0A SPO2_CONFIG
  0x0C LED1_PA(RED)   0x0D LED2_PA(IR)    0x11 MULTI_LED_1   0x12 MULTI_LED_2
  0x1F TEMP_INT       0x20 TEMP_FRAC      0x21 TEMP_EN
  0xFE REV_ID         0xFF PART_ID (MAX30102 = 0x15)
"""

import time


class MAX30102:
    # ---- 寄存器地址 ----
    REG_INT_STATUS_1 = 0x00
    REG_INT_STATUS_2 = 0x01
    REG_INT_ENABLE_1 = 0x02
    REG_INT_ENABLE_2 = 0x03
    REG_FIFO_WR_PTR  = 0x04
    REG_OVF_COUNTER  = 0x05
    REG_FIFO_RD_PTR  = 0x06
    REG_FIFO_DATA    = 0x07
    REG_FIFO_CONFIG  = 0x08
    REG_MODE_CONFIG  = 0x09
    REG_SPO2_CONFIG  = 0x0A
    REG_LED1_PA      = 0x0C
    REG_LED2_PA      = 0x0D
    REG_MULTI_LED_1  = 0x11
    REG_MULTI_LED_2  = 0x12
    REG_TEMP_INT     = 0x1F
    REG_TEMP_FRAC    = 0x20
    REG_TEMP_EN      = 0x21
    REG_REV_ID       = 0xFE
    REG_PART_ID      = 0xFF

    EXPECTED_PART_ID = 0x15

    MODE_SHUTDOWN = 0x80
    MODE_RESET    = 0x40
    MODE_HR_ONLY  = 0x02
    MODE_SPO2     = 0x03
    MODE_MULTI    = 0x07

    # SpO2_CONFIG 位域取值
    ADC_RANGE = {2048: 0x00, 4096: 0x20, 8192: 0x40, 16384: 0x60}
    SR_BITS   = {50: 0, 100: 1, 200: 2, 400: 3, 800: 4, 1000: 5, 1600: 6, 3200: 7}
    PW_BITS   = {69: 0x00, 118: 0x01, 215: 0x02, 411: 0x03}   # us
    AVG_BITS  = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}

    def __init__(self, i2c, addr=0x57):
        self.i2c = i2c
        self.addr = addr
        self._read_ptr = 0
        self.ovf_count = 0
        self.sample_count = 0
        self.sample_rate = 0
        self.led_current = 0x3F
        self.gain_code = 0x3F    # 最近一次 auto_gain 选定的电流码
        self.gain_dc = 0.0       # 定档后实测 IR 直流
        self.gain_ok = False     # 定档后直流是否落在目标区间

    # ================= 底层寄存器 =================
    def _wreg(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val & 0xFF]))

    def _rreg(self, reg):
        return self.i2c.readfrom_mem(self.addr, reg, 1)[0]

    def _rblock(self, reg, n):
        return self.i2c.readfrom_mem(self.addr, reg, n)

    # ================= 识别 =================
    def part_id(self):
        return self._rreg(self.REG_PART_ID)

    def rev_id(self):
        return self._rreg(self.REG_REV_ID)

    def present(self):
        """探测传感器是否存在且型号正确。"""
        try:
            return self.part_id() == self.EXPECTED_PART_ID
        except OSError:
            return False

    # ================= 初始化 =================
    def begin(self, sample_rate=100, led_current=0x24, avg=1,
              adc_range=16384, pulse_width=411):
        """初始化传感器。成功返回 True。

        sample_rate : 采样率 Hz
        led_current : LED 电流寄存器值(0~255)
        avg         : FIFO 平均次数
        adc_range   : ADC 满量程 nA (2048/4096/8192/16384)
        pulse_width : LED 脉宽 us (69/118/215/411)
        """
        # 1) 软复位
        self._wreg(self.REG_MODE_CONFIG, self.MODE_RESET)
        time.sleep_ms(100)
        for _ in range(10):
            if self._rreg(self.REG_MODE_CONFIG) & self.MODE_RESET == 0:
                break
            time.sleep_ms(20)

        # 2) 校验型号
        if self.part_id() != self.EXPECTED_PART_ID:
            return False

        # 3) 清空 FIFO 指针
        self._wreg(self.REG_FIFO_WR_PTR, 0x00)
        self._wreg(self.REG_OVF_COUNTER, 0x00)
        self._wreg(self.REG_FIFO_RD_PTR, 0x00)
        self._read_ptr = 0
        self.ovf_count = 0

        # 4) FIFO 配置: SMP_AVE<<5 | FIFO_ROLLOVER_EN(0x10) | A_FULL(0x0F)
        self._wreg(self.REG_FIFO_CONFIG,
                   (self.AVG_BITS[avg] << 5) | 0x10 | 0x0F)

        # 5) SpO2 配置: ADC_RANGE | SR<<2 | PW
        spo2 = self.ADC_RANGE[adc_range] | (self.SR_BITS[sample_rate] << 2) | self.PW_BITS[pulse_width]
        self._wreg(self.REG_SPO2_CONFIG, spo2)

        # 6) LED 电流 (RED + IR)
        self._wreg(self.REG_LED1_PA, led_current)
        self._wreg(self.REG_LED2_PA, led_current)
        self.led_current = led_current & 0xFF

        # 7) 模式 = SpO2 (同时点亮 RED + IR)
        self._wreg(self.REG_MODE_CONFIG, self.MODE_SPO2)
        time.sleep_ms(50)

        # 8) 清除中断状态
        self._rreg(self.REG_INT_STATUS_1)
        self._rreg(self.REG_INT_STATUS_2)

        # 9) FIFO 平均后有效输出率 = 采样率 / 平均次数
        self.sample_rate = sample_rate // avg
        return True

    # ================= 自适应增益 =================
    def set_led_current(self, code):
        """设置 LED 电流码(0~255)，同时作用于 RED/IR。返回实际写入码。"""
        c = 0x00 if code < 0 else (0xFF if code > 0xFF else int(code))
        self._wreg(self.REG_LED1_PA, c)
        self._wreg(self.REG_LED2_PA, c)
        self.led_current = c
        return c

    def _measure_dc(self, ms=150):
        """在给定窗口内读取 FIFO，返回 (IR 均值, RED 均值, 样本数)。"""
        si = 0
        sr = 0
        n = 0
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < ms:
            try:
                for (red, ir) in self.read_fifo():
                    si += ir
                    sr += red
                    n += 1
            except OSError:
                pass
            time.sleep_ms(5)
        if n == 0:
            return 0.0, 0.0, 0
        return si / n, sr / n, n

    def auto_gain(self, target=60000, lo=20000, hi=100000, code=None, tries=8):
        """自适应调整 LED 电流，使 IR 直流收敛到目标电平 target（默认 60000）。

        原理: 光电流与 LED 电流近似成正比, 故按比例反解
              code_new = code * target / dc_measured
        工程要点(实测踩坑):
          - FIFO 深 32, 改电流后旧样本最长残留 320ms → 每次改档后必须等 400ms 再测;
          - 单次 250ms 短窗定档会在信号上行期误判 → 必须定档后复测确认;
          - 手指贴合差异可达 20 倍以上 → 必须留余量, 不能贴到上限。
          - **收敛判据**: 原实现"落入 [lo,hi] 即 break"会在首次恰好落到区间
            下沿(如 24838)时过早停在低电平 → SNR 不足。改为"落在目标 ±50%
            内才停", 真正把直流拉到 3万~9万 的良信噪比区间。
        返回最终电流码; 结果存于 self.gain_code / self.gain_dc / self.gain_ok。
        """
        c = self.led_current if code is None else code
        if c < 0x08:
            c = 0x08
        dc = 0.0
        for _ in range(tries):
            self.set_led_current(c)
            time.sleep_ms(400)                 # 等 FIFO 内旧档样本全部刷出
            dc, _rd, n = self._measure_dc(250)
            if n == 0:
                break
            if dc <= 0:
                c = min(0xFF, c + 0x10)
                continue
            # 收敛判据: 目标 ±50% 内视为合适(真正收敛, 而非刚进区间就停)
            if 0.5 * target <= dc <= 1.5 * target:
                break
            new = int(c * (target / float(dc)))
            if new < 0x03:
                new = 0x03
            elif new > 0xFF:
                new = 0xFF
            if new == c:
                break
            c = new

        # 复测确认(防止信号上行期误判)
        self.set_led_current(c)
        time.sleep_ms(400)
        dc, _rd, n = self._measure_dc(250)
        self.gain_code = c
        self.gain_dc = dc
        self.gain_ok = (lo <= dc <= hi)
        return c

    # ================= 数据读取 =================
    def num_available(self):
        """返回 FIFO 中尚未读取的样本数。

        以硬件 RD_PTR/WR_PTR 为准(而非软件计数): 一旦软件计数与芯片实际读指针
        错位 3 字节, 就会把 RED/IR 通道读反, 造成无法解释的信号异常。
        """
        wr = self._rreg(self.REG_FIFO_WR_PTR)
        rd = self._rreg(self.REG_FIFO_RD_PTR)
        self._read_ptr = rd
        return (wr - rd) & 0x1F

    def check_overflow(self):
        """读取 FIFO 溢出计数并清零。

        ⚠️【实测陷阱, 2026-09-30】这枚兼容 die 的 OVF_COUNTER(0x05) **读后不
        自动清零**(正品 MAX30102 是 read-to-clear)。若只做 `ovf_count += 读值`,
        计数会随读取次数线性膨胀 —— 实测 15 秒里报出"溢出 1830 个样本"
        (换算输出率 222Hz), 而每秒样本数稳定在 100, 显然自相矛盾。
        正确做法: 读完**主动写 0 清零**(写 FIFO 指针类寄存器即清零)。

        补充: 只要读取足够勤(FIFO 深 32, 我们每 ~10ms 读一次),
        真实溢出应为 0; 一旦这个数为正且持续增长, 说明读取节奏跟不上了。
        """
        ovf = self._rreg(self.REG_OVF_COUNTER)
        if ovf:
            try:
                self._wreg(self.REG_OVF_COUNTER, 0x00)   # 主动清零
            except OSError:
                pass
            self.ovf_count += ovf
        return ovf

    def read_fifo(self):
        """读取 FIFO 内全部可用样本，返回 [(red, ir), ...]（18-bit）。"""
        n = self.num_available()
        self.last_overflow = self.check_overflow()
        if n == 0:
            return []
        raw = self._rblock(self.REG_FIFO_DATA, n * 6)
        if len(raw) != n * 6:
            raise OSError('Short MAX30102 FIFO read')
        out = []
        for i in range(n):
            j = i * 6
            red = ((raw[j] << 16) | (raw[j + 1] << 8) | raw[j + 2]) & 0x3FFFF
            ir  = ((raw[j + 3] << 16) | (raw[j + 4] << 8) | raw[j + 5]) & 0x3FFFF
            out.append((red, ir))
        self.sample_count += n
        return out

    def flush_fifo(self):
        """Explicit continuity break after read failure/overflow; discard uncertain data."""
        for reg in (self.REG_FIFO_WR_PTR, self.REG_OVF_COUNTER, self.REG_FIFO_RD_PTR):
            self._wreg(reg, 0)
        self.last_overflow = 0

    def read_fifo_single(self):
        """读取一个样本 (red, ir)；无数据返回 None。"""
        if self.num_available() == 0:
            return None
        raw = self._rblock(self.REG_FIFO_DATA, 6)
        red = ((raw[0] << 16) | (raw[1] << 8) | raw[2]) & 0x3FFFF
        ir  = ((raw[3] << 16) | (raw[4] << 8) | raw[5]) & 0x3FFFF
        self.sample_count += 1
        return red, ir

    def read_temp(self):
        """读取芯片温度(℃)。"""
        self._wreg(self.REG_TEMP_EN, 0x01)
        time.sleep_ms(30)
        t_int = self._rreg(self.REG_TEMP_INT)
        t_frac = self._rreg(self.REG_TEMP_FRAC)
        if t_int > 127:
            t_int -= 256
        return t_int + t_frac * 0.0625

    def shutdown(self):
        self._wreg(self.REG_MODE_CONFIG, self.MODE_SHUTDOWN)

    def wakeup(self):
        self._wreg(self.REG_MODE_CONFIG, self.MODE_SPO2)
