# -*- coding: utf-8 -*-
"""
wrist_test.py - 测量部位对照实验(腕部 vs 指腹), 负对照

目的: 用同一套信号链、同一套判据, 量化"把 MAX30102 贴在手腕"相对"指腹"的
      信号质量劣化幅度, 为项目硬件选型(腕式必须使用专用 AFM 芯片)提供实测依据。

与 measure.py 的关键区别: **不因增益校验失败而中止**。
      腕部灌注弱, auto_gain 很可能顶到电流上限仍达不到目标直流区间 ——
      这一现象本身就是核心结论, 必须完整记录下来。

对照基准(指腹实测, 2026-09-26):
      IR 直流 49000~75000 / AC 幅度 416~1000 / AC-DC 比 0.6~1.0%
      HR 75~89 bpm / RMSSD 91 ms

用法(PC 端): mpremote connect COM6 run wrist_test.py
"""

import time
from machine import I2C, Pin

import config
from max30102 import MAX30102
from hrv import HRVAnalyzer

DURATION_S = 60
SETTLE_S = 6
SAT = 262143

# ---------------- 初始化 ----------------
i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
          sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
ppg = MAX30102(i2c, config.MAX30102_ADDR)
if not ppg.present():
    raise SystemExit("MAX30102 NOT FOUND: 检查 SDA=21/SCL=22/3.3V")
ppg.begin(sample_rate=config.MAX30102_SAMPLE_RATE,
          led_current=config.LED_IR_CURRENT,
          avg=config.MAX30102_FIFO_AVG,
          adc_range=config.MAX30102_ADC_RANGE)
print("SPO2_CFG=0x%02X  ADC_RANGE=%dnA  (腕部对照实验)"
      % (ppg._rreg(0x0A), config.MAX30102_ADC_RANGE))

code = ppg.auto_gain()
print("auto_gain: LED=0x%02X  IR_DC=%.0f  in_range=%s"
      % (code, ppg.gain_dc, ppg.gain_ok))
if not ppg.gain_ok:
    print(">> 关键结论: 腕部自适应增益已到可用上限, 仍达不到目标直流区间")
    print(">>   电流码 0x%02X | 实测直流 %.0f | 目标 [25000, 100000]"
          % (code, ppg.gain_dc))
    print(">>   => 腕部反射光电流不足, 信噪比先天受限")

time.sleep_ms(1500)

hrv = HRVAnalyzer(fs=config.MAX30102_SAMPLE_RATE,
                  window_sec=config.HRV_WINDOW_SEC,
                  min_peak_ms=config.PPG_MIN_PEAK_MS,
                  max_peak_ms=config.PPG_MAX_PEAK_MS,
                  peak_ratio=config.PPG_PEAK_RATIO)

# ---------------- 收敛期 ----------------
print("静置收敛 %ds ..." % SETTLE_S)
ts = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), ts) < SETTLE_S * 1000:
    for _r, ir in ppg.read_fifo():
        hrv.add_sample(ir)
    time.sleep_ms(5)
hrv.reset()

# ---------------- 正式采集 ----------------
print("开始采集 %ds ..." % DURATION_S)
t0 = time.ticks_ms()
imin = 1 << 30
imax = 0
n_sat = 0
count = 0
dc_sum = 0
dc_n = 0

while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
    for _r, ir in ppg.read_fifo():
        if ir < imin:
            imin = ir
        if ir > imax:
            imax = ir
        if ir >= SAT:
            n_sat += 1
        dc_sum += ir
        dc_n += 1
        count += 1
        hrv.add_sample(ir)
    time.sleep_ms(10)

# ---------------- 汇总 ----------------
m = hrv.metrics()
dc_mean = (dc_sum / dc_n) if dc_n else 0.0
acdc = (m['ac'] / dc_mean * 100.0) if dc_mean > 0 else 0.0

print("")
print("===== 腕部对照结果 =====")
print("采样点数 : %d  (%.1f Hz)   饱和样本 %d" % (count, count / float(DURATION_S), n_sat))
print("LED 电流 : 0x%02X" % code)
print("IR 直流  : 均值 %.0f   范围 %d ~ %d" % (dc_mean, imin, imax))
print("AC 幅度  : %.1f    AC/DC = %.2f %%" % (m['ac'], acdc))
print("HR       : %.1f bpm" % m['hr'])
print("SDNN     : %.1f ms   RMSSD %.1f ms   pNN50 %.1f %%"
      % (m['sdnn'], m['rmssd'], m['pnn50']))
print("RR 数     : %d (剔除离群 %d)" % (m['nn'], m['rej']))
print("RR 序列   : %s" % hrv.rr[-30:])
print("")
print("----- 与指腹基准对照 (2026-09-26) -----")
print("指标        指腹基准           腕部本次")
print("IR 直流     49000 ~ 75000      %d ~ %d" % (imin, imax))
print("AC 幅度     416 ~ 1000         %.0f" % m['ac'])
print("AC/DC       0.60 ~ 1.00 %%      %.2f %%" % acdc)
print("HR          75 ~ 89 bpm        %.1f bpm" % m['hr'])
print("RMSSD       91 ms              %.1f ms" % m['rmssd'])
print("")

verdict = []
if imin < 20000:
    verdict.append("IR直流严重偏低")
if m['ac'] < 200:
    verdict.append("AC幅度过小(近噪声)")
if not (40 <= m['hr'] <= 150):
    verdict.append("HR不合理")
if not (10 <= m['rmssd'] <= 120):
    verdict.append("RMSSD失真")
print("腕部可用性判定 :", "与指腹相当" if not verdict else "不合格 -> " + ", ".join(verdict))
