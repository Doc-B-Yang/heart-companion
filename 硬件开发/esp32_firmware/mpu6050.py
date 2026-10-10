# -*- coding: utf-8 -*-
"""
mpu6050.py - MPU6050 六轴 IMU 驱动 (MicroPython / ESP32)

用途: 为 MAX30102 的 PPG 信号提供**运动伪迹抑制**依据。
      PPG 对运动极其敏感(手一动峰位就乱), 而 IMU 能直接测出"此刻是否在动"。

接线 (GY-521 模块, 8 针):
    VCC  -> 3V3      (必须 3.3V, 不要 5V)
    GND  -> GND
    SDA  -> GPIO21   (与 MAX30102/ADS1115 共用 I2C 总线)
    SCL  -> GPIO22
    AD0  -> GND      (定址 0x68; 接 VDD 则变 0x69)
    INT  -> 任选 GPIO(可选) —— 仅在需要硬件中断节拍时接; 轮询模式不需要
    XDA/XCL -> 悬空  (仅用于外挂磁力计, 本驱动不用)
    INT  -> GPIO17   (可选, 见 config.MPU6050_INT_PIN)

⚠️ 本驱动默认用**轮询**读数据(与 PPG 主循环同频), 不依赖 INT。
   INT 只用于验证引脚/中断通路是否正常 (mpu_test.py 的 test_int)。

寄存器手册: RM-MPU-6000A-00
"""

import math
import time
import struct

# ---------------- 寄存器地址 ----------------
REG_SMPLRT_DIV   = 0x19   # 采样率分频
REG_CONFIG       = 0x1A   # 数字低通滤波 DLPF
REG_GYRO_CONFIG  = 0x1B   # 陀螺量程
REG_ACCEL_CONFIG = 0x1C   # 加速度量程
REG_INT_PIN_CFG  = 0x37   # 中断引脚电气特性
REG_INT_ENABLE   = 0x38   # 中断使能
REG_INT_STATUS   = 0x3A   # 中断状态(读后清除)
REG_ACCEL_XOUT_H = 0x3B   # 加速度起址(6B) + 温度(2B) + 陀螺(6B) = 连续 14 字节
REG_PWR_MGMT_1   = 0x6B   # 电源管理 1
REG_PWR_MGMT_2   = 0x6C   # 电源管理 2
REG_WHO_AM_I     = 0x75   # 器件 ID

# ---------------- 量程对照表 ----------------
_ACCEL_LSB = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}      # LSB/g
_GYRO_LSB  = {250: 131.0, 500: 65.5, 1000: 32.8, 2000: 16.4}     # LSB/(°/s)

# DLPF_CFG -> (陀螺带宽Hz, 陀螺延迟ms, 加速度带宽Hz, 加速度延迟ms, 陀螺输出率Hz)
_DLPF = {
    0: (260, 0.98, 260, 0, 8000),
    1: (188, 1.9, 184, 2.0, 1000),
    2: (98, 2.8, 94, 3.0, 1000),
    3: (42, 4.8, 44, 4.9, 1000),
    4: (20, 8.3, 21, 8.5, 1000),
    5: (10, 13.4, 10, 13.8, 1000),
    6: (5, 18.6, 5, 19.0, 1000),
}

# 已知的 WHO_AM_I 取值
# 正品 MPU6050 = 0x68。但 MPU6050 早已停产, 市售 GY-521 基本都是兼容替代片:
#   0x70/0x71/0x73 = MPU6500/9250/9255(寄存器映射兼容)
#   0x98/0x72      = 常见"山寨"ID(实测仍有器件返回, 功能正常)
#   0x75 = 本项目观测值。仅支持有限台架功能，不能据此确认芯片身份或全部兼容性。
KNOWN_IDS = {
    0x68: "MPU6050(正品)",
    0x70: "MPU6500(兼容)",
    0x71: "MPU9250(兼容)",
    0x73: "MPU9255(兼容)",
    0x98: "MPU6050 山寨片",
    0x72: "MPU6050 山寨片",
    0x75: "MPU6050 兼容die(实测)",
}

# 寄存器兼容即可正常工作, 不必强求 0x68
COMPATIBLE_IDS = set(KNOWN_IDS) | {0x74, 0x76, 0x78}


class MPU6050:
    """MPU6050 六轴 IMU (加速度 + 陀螺 + 片内温度)"""

    def __init__(self, i2c, addr=0x68, int_pin=None):
        self.i2c = i2c
        self.addr = addr
        self.int_pin_no = int_pin
        self.accel_range = 2
        self.gyro_range = 250
        self.dlpf = 3
        self.rate_hz = 100
        self.chip = "unknown"
        # 运动判定状态
        self._g_ref = None      # 静态基线 |a| (g)
        self._dev_ema = 0.0     # 合加速度偏差的指数平滑值
        self._still_cnt = 0
        self._motion_cnt = 0
        # 陀螺零偏(静止时的直流偏置) —— 判据②必须先扣掉它
        self._gyro_bias = (0.0, 0.0, 0.0)
        # 重力方向参考(单位向量) —— 判据③(倾角)用
        self._grav = None
        self._omega_ema = 0.0   # 角速度平滑值
        self._tilt_ema = 0.0    # 倾角平滑值

    # ------------------------------------------------ 底层读写
    def _w(self, reg, val):
        self.i2c.writeto_mem(self.addr, reg, bytes([val & 0xFF]))

    def _r(self, reg, n=1):
        return self.i2c.readfrom_mem(self.addr, reg, n)

    def _modify(self, reg, mask, val):
        """读-改-写指定位域"""
        cur = self._r(reg)[0]
        self._w(reg, (cur & ~mask) | (val & mask))

    # ------------------------------------------------ 器件识别
    def who_am_i(self):
        v = self._r(REG_WHO_AM_I)[0]
        self.chip = KNOWN_IDS.get(v, "UNKNOWN(0x%02X)" % v)
        return v

    def is_mpu6050(self):
        """是否为正品 MPU6050 (严格判据)"""
        return self.who_am_i() == 0x68

    def is_compatible(self):
        """是否为寄存器兼容片 (可用判据)

        不以 WHO_AM_I 为准 —— MPU6050 已停产, 兼容片极常见。
        真正可靠的判据是"能否读写寄存器", 见 verify_registers()。
        """
        return self.who_am_i() in COMPATIBLE_IDS

    def verify_registers(self):
        """写-回读校验: 逐项写入已知值再读回比对。

        这是比 WHO_AM_I 更可靠的可用性判据。
        返回 (全部一致?, [(寄存器名, 写入值, 读回值), ...])
        """
        checks = [
            ("SMPLRT_DIV", REG_SMPLRT_DIV, 0x09),
            ("CONFIG", REG_CONFIG, self.dlpf & 0x07),
            ("ACCEL_CONFIG", REG_ACCEL_CONFIG,
             {2: 0, 4: 1, 8: 2, 16: 3}[self.accel_range] << 3),
            ("GYRO_CONFIG", REG_GYRO_CONFIG,
             {250: 0, 500: 1, 1000: 2, 2000: 3}[self.gyro_range] << 3),
            ("PWR_MGMT_1", REG_PWR_MGMT_1, 0x01),
            ("PWR_MGMT_2", REG_PWR_MGMT_2, 0x00),
        ]
        rows = []
        all_ok = True
        for name, reg, want in checks:
            self._w(reg, want)
            time.sleep_ms(5)
            got = self._r(reg)[0]
            rows.append((name, want, got))
            if got != want:
                all_ok = False
        return all_ok, rows

    # ------------------------------------------------ 初始化
    def begin(self, accel_range=2, gyro_range=250, dlpf=3, rate_hz=100):
        """唤醒并配置。accel_range: 2/4/8/16 (g); gyro_range: 250/500/1000/2000"""
        if accel_range not in _ACCEL_LSB:
            raise ValueError("accel_range 只能是 2/4/8/16")
        if gyro_range not in _GYRO_LSB:
            raise ValueError("gyro_range 只能是 250/500/1000/2000")
        if dlpf not in _DLPF:
            raise ValueError("dlpf 只能是 0~6")
        if not 4 <= rate_hz <= 1000:
            raise ValueError('rate_hz需在4~1000范围内')

        # 1) 复位 (DEVICE_RESET = bit7), 等待完成后自动清零
        self._w(REG_PWR_MGMT_1, 0x80)
        time.sleep_ms(120)

        # 2) 唤醒 + 选时钟源 = PLL(X 陀螺) —— 比内部 8MHz 振荡器稳得多
        self._w(REG_PWR_MGMT_1, 0x01)
        time.sleep_ms(50)
        # 六轴全部启用
        self._w(REG_PWR_MGMT_2, 0x00)

        # 3) DLPF
        self.dlpf = dlpf
        self._w(REG_CONFIG, dlpf & 0x07)

        # 4) 采样率: 陀螺输出率(由 DLPF 决定) / (1 + SMPLRT_DIV)
        gyro_out = _DLPF[dlpf][4]
        div = max(0, min(255, int(round(gyro_out / float(rate_hz))) - 1))
        self._w(REG_SMPLRT_DIV, div)
        self.rate_hz = gyro_out / (1.0 + div)

        # 5) 量程
        self.accel_range = accel_range
        self.gyro_range = gyro_range
        # AFS_SEL: 0=±2g 1=±4g 2=±8g 3=±16g (bit4:3)
        afs = {2: 0, 4: 1, 8: 2, 16: 3}[accel_range]
        self._modify(REG_ACCEL_CONFIG, 0x18, afs << 3)
        # FS_SEL: 0=±250 1=±500 2=±1000 3=±2000 dps (bit4:3)
        fs = {250: 0, 500: 1, 1000: 2, 2000: 3}[gyro_range]
        self._modify(REG_GYRO_CONFIG, 0x18, fs << 3)

        # 6) 关中断(轮询模式默认不使能)
        self._w(REG_INT_ENABLE, 0x00)

        time.sleep_ms(30)
        expected = ((REG_PWR_MGMT_1, 0x01), (REG_PWR_MGMT_2, 0x00),
                    (REG_CONFIG, dlpf & 7), (REG_SMPLRT_DIV, div),
                    (REG_ACCEL_CONFIG, afs << 3), (REG_GYRO_CONFIG, fs << 3))
        for reg, value in expected:
            if self._r(reg)[0] != value:
                raise OSError('MPU configuration readback failed at 0x%02x' % reg)
        return True

    # ------------------------------------------------ 读取
    def read_raw(self):
        """一次突发读 14 字节: ax ay az temp gx gy gz (原始有符号 16 位)"""
        b = self._r(REG_ACCEL_XOUT_H, 14)
        ax, ay, az, t, gx, gy, gz = struct.unpack(">7h", b)
        return ax, ay, az, t, gx, gy, gz

    def read(self):
        """返回物理量字典 (单位: g / °/s / °C)"""
        ax, ay, az, t, gx, gy, gz = self.read_raw()
        ka = _ACCEL_LSB[self.accel_range]
        kg = _GYRO_LSB[self.gyro_range]
        return {
            "ax": ax / ka, "ay": ay / ka, "az": az / ka,
            "gx": gx / kg, "gy": gy / kg, "gz": gz / kg,
            "temp": t / 340.0 + 36.53,
        }

    def accel_mag(self):
        """合加速度模长 (g)。静止时应 ≈ 1.000, 偏差即运动幅度。"""
        ax, ay, az, _t, _gx, _gy, _gz = self.read_raw()
        ka = _ACCEL_LSB[self.accel_range]
        x = ax / ka; y = ay / ka; z = az / ka
        return (x * x + y * y + z * z) ** 0.5

    # ------------------------------------------------ 陀螺零偏
    def set_gyro_bias(self, bias):
        """注入已测得的零偏三元组 (°/s)"""
        self._gyro_bias = (float(bias[0]), float(bias[1]), float(bias[2]))

    def gyro_bias(self):
        return self._gyro_bias

    def calibrate_gyro(self, seconds=2.0):
        """静止采零偏: 平均陀螺读数即为直流偏置。返回 (bx, by, bz) °/s

        ⚠️ 必须在**完全静止**时调用。MPU6050 零偏可达 ±20 °/s(规格),
        不扣除的话静止时 |ω| 恒不为 0, 角速度判据会一直误报运动。
        """
        sx = sy = sz = squares = 0.0
        n = 0
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < int(seconds * 1000):
            d = self.read()
            sx += d["gx"]; sy += d["gy"]; sz += d["gz"]
            squares += d['gx']**2 + d['gy']**2 + d['gz']**2
            n += 1
            time.sleep_ms(10)
        if n < 20:
            raise ValueError('Gyro calibration needs at least20 samples')
        bias = (sx / n, sy / n, sz / n)
        variance = max(0, squares/n - sum(v*v for v in bias))
        if variance > 9 or max(abs(v) for v in bias) > 60:
            raise ValueError('MPU moved or has abnormal gyro during calibration')
        self.set_gyro_bias(bias)
        return self._gyro_bias

    def gyro_mag(self):
        """扣除零偏后的角速度模长 (°/s) —— 运动判据②"""
        d = self.read()
        bx, by, bz = self._gyro_bias
        x = d["gx"] - bx; y = d["gy"] - by; z = d["gz"] - bz
        return (x * x + y * y + z * z) ** 0.5

    # ------------------------------------------------ 运动判据
    def init_gravity(self, seconds=1.0):
        """静止时建立重力方向参考(单位向量)与 |a| 基线"""
        sx = sy = sz = 0.0
        n = 0
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < int(seconds * 1000):
            d = self.read()
            sx += d["ax"]; sy += d["ay"]; sz += d["az"]
            n += 1
            time.sleep_ms(10)
        if n < 20:
            raise ValueError('Gravity calibration needs at least20 samples')
        gx, gy, gz = sx / n, sy / n, sz / n
        m = (gx * gx + gy * gy + gz * gz) ** 0.5
        if not .8 <= m <= 1.2:
            raise ValueError('Abnormal gravity; do not enable motion gate')
        self._g_ref = m
        if m > 1e-6:
            self._grav = (gx / m, gy / m, gz / m)
        return self._grav

    def motion_deviations(self, update_ref=True):
        """读一次, 返回三个运动判据 (dev_mag, omega, tilt_deg)

        【关键物理事实 —— 决定伪迹抑制能不能生效】
        dev_mag = ||a| - |g_ref||
            对**平移/冲击**敏感。但对**纯转动完全不敏感** ——
            因为旋转不改变加速度矢量的模长(重力大小恒为 1g)。
            实测: 用手晃板子时 dev_mag 多数只有 0.05~0.25 g,
            而同时 |ω| 高达 70~390 °/s。**只用它会把绝大多数手部运动漏掉。**

        omega = |ω - 零偏|
            对**转动**敏感。而手腕/手指的抖动主要就是转动 →
            这是三个判据里对"戴在手上的传感器"最有用的一个。

        tilt_deg = 当前重力方向与参考方向的夹角(度)
            同样对转动敏感(比 dev_mag 灵敏得多), 且不需要陀螺、
            不受零偏漂移影响, 可作陀螺判据的冗余备份。
        """
        d = self.read()
        ax, ay, az = d["ax"], d["ay"], d["az"]
        mag = (ax * ax + ay * ay + az * az) ** 0.5
        ref = self._g_ref if self._g_ref is not None else 1.0
        dev_mag = abs(mag - ref)

        bx, by, bz = self._gyro_bias
        wx = d["gx"] - bx; wy = d["gy"] - by; wz = d["gz"] - bz
        omega = (wx * wx + wy * wy + wz * wz) ** 0.5

        tilt = 0.0
        if self._grav is not None and mag > 1e-6:
            dot = (ax * self._grav[0] + ay * self._grav[1]
                   + az * self._grav[2]) / mag
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0
            tilt = math.degrees(math.acos(dot))
            # 缓慢更新参考方向 → 允许"缓慢改变姿势", 但"快速抖动"仍会被检出
            if update_ref and tilt < self.tilt_ref_update_deg:
                a = 0.01
                gx = self._grav[0] + a * (ax / mag - self._grav[0])
                gy = self._grav[1] + a * (ay / mag - self._grav[1])
                gz = self._grav[2] + a * (az / mag - self._grav[2])
                m = (gx * gx + gy * gy + gz * gz) ** 0.5 or 1.0
                self._grav = (gx / m, gy / m, gz / m)
        return dev_mag, omega, tilt

    def is_motion(self, update_ref=True):
        """多判据运动判定 —— 返回 (is_motion, (dev_mag, omega, tilt_deg))

        取"或": 任一判据超阈值即判为运动 → 该段 RR 应剔除。
        """
        dev, omega, tilt = self.motion_deviations(update_ref)
        self._dev_ema = 0.6 * self._dev_ema + 0.4 * dev
        self._omega_ema = 0.5 * self._omega_ema + 0.5 * omega
        self._tilt_ema = 0.5 * self._tilt_ema + 0.5 * tilt
        hit = (dev > self.motion_threshold_g or omega > self.gyro_threshold_dps
               or tilt > self.tilt_threshold_deg or self._dev_ema > self.motion_threshold_g
               or self._omega_ema > self.gyro_threshold_dps
               or self._tilt_ema > self.tilt_threshold_deg)
        if hit:
            self._motion_cnt += 1
            self._still_cnt = 0
        else:
            self._still_cnt += 1
            self._motion_cnt = 0
        return hit, (dev, omega, tilt)

    def update_motion(self):
        """兼容旧接口: 只用加速度模长判据。返回 (dev_mag, is_motion)"""
        dev = self.accel_mag()
        ref = self._g_ref if self._g_ref is not None else 1.0
        dev = abs(dev - ref)
        self._dev_ema = 0.7 * self._dev_ema + 0.3 * dev
        return dev, self._dev_ema > self.motion_threshold_g

    # ---- 阈值(可由外部注入, 默认值来自 config) ----
    motion_threshold_g = 0.25      # 判据①: 加速度模长偏差
    gyro_threshold_dps = 20.0      # 判据②: 角速度模长
    tilt_threshold_deg = 8.0       # 判据③: 重力方向偏转
    tilt_ref_update_deg = 20.0     # 参考方向更新门槛(超过则认为是抖动, 不更新)

    # ------------------------------------------------ 静态基线
    def calibrate_still(self, seconds=2.0):
        """采静态基线: 记录静止时的平均 |a|。返回 (g_ref, 噪声标准差)"""
        n = 0
        s = 0.0
        vals = []
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < int(seconds * 1000):
            v = self.accel_mag()
            vals.append(v)
            s += v
            n += 1
            time.sleep_ms(10)
        if n == 0:
            return 1.0, 0.0
        mean = s / n
        var = sum((v - mean) ** 2 for v in vals) / n
        self._g_ref = mean
        return mean, var ** 0.5

    # ------------------------------------------------ 中断验证(可选)
    def enable_data_ready_int(self, latch=False, active_low=False):
        """使能"数据就绪"中断, 用于验证 INT 引脚通路

        latch=False → 50us 脉冲(适合 irq 计数, 中断服务内不必读 I2C)
        latch=True  → 保持高电平直到读 INT_STATUS
        """
        cfg = 0x00
        if active_low:
            cfg |= 0x80          # INT_LEVEL: 1=低有效
        if latch:
            cfg |= 0x20          # LATCH_INT_EN
        cfg |= 0x02              # INT_RD_CLEAR 的互补: 保持 push-pull(bit6=0)
        self._w(REG_INT_PIN_CFG, cfg)
        self._w(REG_INT_ENABLE, 0x01)   # bit0 = DATA_RDY_EN
        time.sleep_ms(10)

    def disable_int(self):
        self._w(REG_INT_ENABLE, 0x00)

    def int_status(self):
        return self._r(REG_INT_STATUS)[0]

    # ------------------------------------------------ 计数器封装
    def counter(self, pin_no=None):
        """返回一个 (start, stop) 计数器, 用于统计 INT 上升沿次数"""
        from machine import Pin
        pin_no = self.int_pin_no if pin_no is None else pin_no
        if pin_no is None:
            raise ValueError("未指定 INT 引脚")
        return _EdgeCounter(pin_no)


class _EdgeCounter:
    """INT 上升沿计数器

    极简: 只做计数, 不做 I2C —— 避免在中断服务里访问总线。
    """

    def __init__(self, pin_no):
        from machine import Pin
        self.pin = Pin(pin_no, Pin.IN)
        self.count = 0
        self._prev = self.pin.value()
        self._poll = False

    def _isr(self, _p):
        self.count += 1

    def start(self):
        from machine import Pin
        self.count = 0
        try:
            self.pin.irq(trigger=Pin.IRQ_RISING, handler=self._isr)
        except Exception:
            # 某些引脚不支持中断 → 退化为轮询
            self._poll = True

    def stop(self):
        if not self._poll:
            try:
                self.pin.irq(handler=None)
            except Exception:
                pass
        return self.count

    def poll_tick(self):
        """轮询模式的等价物(每次调用一次)"""
        v = self.pin.value()
        if v and not self._prev:
            self.count += 1
        self._prev = v
