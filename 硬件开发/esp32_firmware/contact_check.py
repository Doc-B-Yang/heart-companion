# -*- coding: utf-8 -*-
"""
contact_check.py - 指腹贴合 / 信号质量门禁 (在 ESP32 上跑, 8 秒)

为什么需要它:
  实测发现"IR 直流很高、HR 看着也正常(≈75), 但 RMSSD 是 130ms(真实应 20~50)"。
  逐层排查后确认不是算法问题, 而是**信号本身不合格**。三个独立的病征:

    ① 脉搏太弱   —— 光回来了, 却没带着动脉搏动回来
    ② 幅度不稳   —— 相邻心搏原始幅度可达 3 倍差(925 vs 350 counts)
    ③ RR 抖动    —— RMSSD/平均RR > 0.12, 即 RR 序列被伪迹主导

  HR 只需要"数拍子", 弱一点也能对; 但 HRV 需要**逐拍时间精确到几毫秒**,
  以上任何一条都会让它彻底失真。而这三条**改算法都救不了** —— 必须先把
  信号弄干净。本工具就是用来把这件事变成可量化的门禁。

三项判据(全部达标才能开始正式测量):
    指标            好          勉强        不合格
    ────────────────────────────────────────────────
    脉搏/直流 PI    ≥0.8%       0.4~0.8%    <0.4%
    幅度CV          ≤30%        30~45%      >45%
    RR抖动指数      ≤0.08       0.08~0.12   >0.12

用法:
    mpremote connect COM6 run contact_check.py

    反复运行, 每次换一种放法, 取三项都最好的一档。推荐依次试:
      ① 指腹轻搭, 几乎不用力   ← 多数人这一档最好(压紧了会压瘪血管)
      ② 明显加压(指尖发白)      ← 预期就是本次这种病征: PI 低 + 幅度不稳
      ③ 挪位置, 让窗口对准指腹正中(别让指甲侧挡住)
      ④ 换手指(食指/中指指腹)
      ⑤ 手肘有支撑、手掌放松 —— 手悬空时的微调最伤 HRV
      ⑥ 别让台灯/阳光直射传感器
"""

import math
import time
from machine import I2C, Pin

import config
from max30102 import MAX30102

try:
    from mpu6050 import MPU6050
except ImportError:
    MPU6050 = None

SEC = globals().get("SEC", 8)
FS = 100.0
LINE = "=" * 74


class IIR:
    """一阶 EMA 高通/低通, n 级级联"""

    def __init__(self, fc, mode, n=1, fs=FS):
        self.a = min(1.0, 2 * math.pi * fc / fs)
        self.mode = mode
        self.n = n
        self.s = [None] * n
        self.l = [0.0] * n

    def __call__(self, x):
        y = x
        if self.mode == "hp":
            for k in range(self.n):
                if self.s[k] is None:
                    self.s[k] = y
                self.s[k] += self.a * (y - self.s[k])
                y = y - self.s[k]
        else:
            for k in range(self.n):
                self.l[k] += self.a * (y - self.l[k])
                y = self.l[k]
        return y


class PP:
    """峰峰跟踪"""

    def __init__(self, win, skip=0.0):
        self.buf = []
        self.win = win
        self.skip = skip
        self.pps = []

    def push(self, v):
        self.buf.append(v)
        if len(self.buf) > self.win:
            self.buf.pop(0)

    def snap(self, t):
        if t < self.skip or len(self.buf) < self.win // 3:
            return
        self.pps.append(max(self.buf) - min(self.buf))

    def med(self):
        if not self.pps:
            return 0.0
        s = sorted(self.pps)
        return s[len(s) // 2]


def main():
    print("\n" + LINE)
    print("  指腹贴合 / 信号质量门禁   %d 秒" % SEC)
    print(LINE)

    i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
              sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
    ppg = MAX30102(i2c, config.MAX30102_ADDR)
    if not ppg.present():
        print("✘ MAX30102 未找到")
        return
    ppg.begin(sample_rate=config.MAX30102_SAMPLE_RATE,
              led_current=config.LED_IR_CURRENT,
              avg=config.MAX30102_FIFO_AVG,
              adc_range=config.MAX30102_ADC_RANGE)
    code = ppg.auto_gain()
    print("auto_gain: LED=0x%02X (%.1f mA)  IR_DC=%.0f  in_range=%s"
          % (code, code * 0.2, ppg.gain_dc, ppg.gain_ok))

    imu = None
    if MPU6050 is not None and config.MPU6050_ENABLED:
        try:
            imu = MPU6050(i2c, addr=config.MPU6050_ADDR)
            imu.begin(accel_range=2, gyro_range=250,
                      dlpf=config.MPU6050_DLPF,
                      rate_hz=config.MPU6050_RATE_HZ)
            imu.motion_threshold_g = config.MPU6050_MOTION_G
            imu.gyro_threshold_dps = config.MPU6050_GYRO_DPS
            imu.tilt_threshold_deg = config.MPU6050_TILT_DEG
            imu.calibrate_gyro(1.0)
            imu.init_gravity(0.6)
        except OSError:
            imu = None

    # 脉搏带: 高通 0.8Hz 两级(压掉呼吸/位移漂移) + 低通 5Hz
    hp2 = IIR(0.8, "hp", n=2)
    lp = IIR(5.0, "lp")
    # 漂移带: 低通 0.4Hz —— 它是"基线漂移"本身, 不是脉搏
    lp_drift = IIR(0.4, "lp")

    tr_pulse = PP(200)
    tr_drift = PP(300, skip=3.0)

    # 峰检(在脉搏带上)
    peaks = []          # (t_ms, 幅度)
    last_pk = -1e9

    mov_n = 0
    n = 0
    sum_dc = 0
    t0 = time.ticks_ms()
    last = t0

    print("\n⏳ 采集 %d 秒, 请保持不动 ..." % SEC)
    while True:
        now = time.ticks_ms()
        if time.ticks_diff(now, t0) >= SEC * 1000:
            break
        if imu is not None:
            m, _d = imu.is_motion()
            if m:
                mov_n += 1
        try:
            batch = ppg.read_fifo()
        except OSError:
            batch = []
        for (_r, ir) in batch:
            n += 1
            sum_dc += ir
            y = lp(hp2(ir))
            tr_pulse.push(y)
            tr_drift.push(lp_drift(ir))
            # 峰检: 局部极大 + 不应期 400ms
            # (push 之后 buf[-1]=当前样本, [-2]=上一拍, 故 [-2] 是极大时取它)
            b = tr_pulse.buf
            if len(b) >= 3 and b[-2] > b[-3] and b[-2] >= b[-1] and b[-2] > 0:
                pt = n * 1000.0 / FS - 10.0
                if pt - last_pk >= 400:
                    peaks.append((pt, b[-2]))
                    if len(peaks) > 60:
                        peaks.pop(0)
                    last_pk = pt
        if time.ticks_diff(now, last) >= 1000:
            last = now
            el = time.ticks_diff(now, t0) / 1000.0
            tr_pulse.snap(el)
            tr_drift.snap(el)

    if n < 50:
        print("✘ 样本太少(%d), 检查接线" % n)
        return

    dc = sum_dc / float(n)
    pp_pulse = tr_pulse.med()
    pp_drift = tr_drift.med()
    pi = 100.0 * pp_pulse / dc if dc else 0.0

    # 幅度稳定性: 峰幅度的变异系数 CV
    amps = [a for _t, a in peaks]
    if len(amps) >= 4:
        amps2 = amps[1:]                     # 丢掉首个(滤波暂态)
        mu = sum(amps2) / len(amps2)
        var = sum((a - mu) ** 2 for a in amps2) / len(amps2)
        cv = (var ** 0.5) / mu if mu > 1e-6 else 9.9
    else:
        cv = 9.9

    # RR 抖动指数
    rr = [peaks[i][0] - peaks[i - 1][0] for i in range(1, len(peaks))]
    rr = [v for v in rr if 320 <= v <= 2000]
    if len(rr) >= 3:
        mean_rr = sum(rr) / len(rr)
        rms = math.sqrt(sum((rr[i] - rr[i - 1]) ** 2
                            for i in range(1, len(rr))) / (len(rr) - 1))
        jitter = rms / mean_rr if mean_rr > 0 else 9.9
        hr_est = 60000.0 / mean_rr
    else:
        jitter = 9.9
        hr_est = 0.0

    print("\n" + LINE)
    print("  结果")
    print(LINE)
    print("  样本 %d (期望 %d)   IR 直流 %.0f" % (n, int(SEC * FS), dc))
    print("  脉搏带(0.8~5Hz) 峰峰 = %6.0f   → PI = %.3f%% of DC   [判据①]"
          % (pp_pulse, pi))
    print("  漂移带(≤0.4Hz)  峰峰 = %6.0f   → %.3f%% of DC        (不是脉搏)"
          % (pp_drift, 100.0 * pp_drift / dc if dc else 0))
    print("  检出峰 %d 个   幅度CV = %.1f%%                       [判据②]"
          % (len(amps), cv * 100))
    print("  心率估计 %.1f bpm   RR抖动指数 = %.3f                    [判据③]"
          % (hr_est, jitter))
    print("  运动样本 %d / %d" % (mov_n, n))

    # ---------------- 判定 ----------------
    print("\n" + LINE)
    print("  判定")
    print(LINE)
    if dc >= 262100:
        print("  ✘ ADC 饱和 —— 先查量程(16384nA)与 LED 电流, 不要动算法")
        return
    if dc <= 500:
        print("  ✘ 几乎无光返回 —— 指腹没盖住, 或 LED 电流定错档")
        return

    def mark(ok, warn):
        return "✔" if ok else ("⚠" if warn else "✘")

    o1 = pi >= 0.8
    w1 = 0.4 <= pi < 0.8
    o2 = cv <= 0.30
    w2 = 0.30 < cv <= 0.45
    o3 = jitter <= 0.08
    w3 = 0.08 < jitter <= 0.12

    print("  %s ① 脉搏强度 : PI %.3f%%   (好≥0.8   勉强0.4~0.8   差<0.4)"
          % (mark(o1, w1), pi))
    print("  %s ② 幅度稳定 : CV %.1f%%   (好≤30    勉强30~45     差>45)"
          % (mark(o2, w2), cv * 100))
    print("  %s ③ RR 抖动  : %.3f     (好≤0.08  勉强0.08~0.12  差>0.12)"
          % (mark(o3, w3), jitter))

    allok = o1 and o2 and o3
    print()
    if allok:
        print("  ➜ 三项全部达标 —— 可以开始正式测量/实验。")
        print("     HR 与 HRV 都可信, 后续 %s 模式的准确率才有意义。"
              % "pipeline_check")
    else:
        bad = []
        if not o1:
            bad.append("脉搏太弱")
        if not o2:
            bad.append("幅度不稳")
        if not o3:
            bad.append("RR 抖动大")
        print("  ➜ 未达标: %s" % "、".join(bad))
        print()
        if not o2 or not o3:
            print("  【重点】幅度不稳 / RR 抖动 是 **HR 看着正常但 HRV 完全失真**")
            print("     的元凶。实测未达标时会得到 RMSSD 130ms(真实应 20~50)。")
            print("     最常见原因按概率排序:")
            print("       1. 手悬空 / 手肘没支撑 —— 手指在做微小调整")
            print("       2. 压得太紧(指尖发白) —— 血管被压瘪, 波形畸变(双峰)")
            print("       3. 指腹没完全盖住窗口, 边缘漏光")
            print("     处理: 手肘撑桌面 → 指腹**轻搭**、几乎不用力 → 完全覆盖窗口后")
            print("           保持 10 秒不动再读结果。")
        if not o1:
            print("  · 脉搏弱: 指温低时灌注会明显变差, 可先搓热手指再测。")
    print("\n" + LINE)
    print("  反复跑本脚本, 换放法, 取三项都最好的一档")
    print(LINE)


main()
