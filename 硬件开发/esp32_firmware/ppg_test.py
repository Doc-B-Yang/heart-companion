# -*- coding: utf-8 -*-
"""
ppg_test.py - 纯 MAX30102 测试(不含雷达)

用途: 单独验证 MAX30102 这一路能产出什么 HRV 特征、信号质量如何,
      用于评估"单模态 PPG 能否支撑情绪分类"。
流程: 初始化 -> auto_gain(带校验) -> 静置收敛 -> 正式采集 -> 汇总判定
用法: mpremote connect COM6 run ppg_test.py

实测(2026-09-26): LED 收敛到 0x4A / IR_DC=64308 / AC≈400 / AC-DC≈0.62%;
                 HR=67.4bpm(正确), 但 RMSSD=174.7ms / pNN50=74.3% (失真)
                 -> 弱灌注下峰位抖动大, 单模态 PPG 的 HRV 精度不足以支撑多类情绪分类
"""

import time
from machine import I2C, Pin

import config
from max30102 import MAX30102
from hrv import HRVAnalyzer

DURATION_S = 90
SETTLE_S = 8
SAT = 262143

# ---------------- 初始化 ----------------
i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
          sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
ppg = MAX30102(i2c, config.MAX30102_ADDR)
if not ppg.present():
    raise SystemExit("MAX30102 NOT FOUND: 检查 SDA=21/SCL=22/3.3V")
if not ppg.begin(sample_rate=config.MAX30102_SAMPLE_RATE,
                 led_current=config.LED_IR_CURRENT,
                 avg=config.MAX30102_FIFO_AVG,
                 adc_range=config.MAX30102_ADC_RANGE):
    raise SystemExit("begin() failed")
print("[PPG] MAX30102 OK  SPO2_CFG=0x%02X  ADC_RANGE=%dnA"
      % (ppg._rreg(0x0A), config.MAX30102_ADC_RANGE))

code = ppg.auto_gain()
print("[PPG] auto_gain: LED=0x%02X  IR_DC=%.0f  in_range=%s"
      % (code, ppg.gain_dc, ppg.gain_ok))
if not ppg.gain_ok:
    raise SystemExit("!! 增益校验未通过: 传感器上无手指/光耦极差 -> 请按标准贴合后重跑")

time.sleep_ms(1500)

hrv = HRVAnalyzer(fs=config.MAX30102_SAMPLE_RATE,
                  window_sec=config.HRV_WINDOW_SEC,
                  min_peak_ms=config.PPG_MIN_PEAK_MS,
                  max_peak_ms=config.PPG_MAX_PEAK_MS,
                  peak_ratio=config.PPG_PEAK_RATIO)

# ---------------- 收敛期 ----------------
print("[PPG] 静置收敛 %ds ... 请保持手指稳定贴合" % SETTLE_S)
ts = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), ts) < SETTLE_S * 1000:
    for _r, ir in ppg.read_fifo():
        hrv.add_sample(ir)
    time.sleep_ms(5)
hrv.reset()

# ---------------- 正式采集 ----------------
print("[PPG] 开始正式采集 %ds ..." % DURATION_S)
t0 = time.ticks_ms()
last = t0
imin = 1 << 30
imax = 0
n_sat = 0
count = 0

while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
    now = time.ticks_ms()
    for _r, ir in ppg.read_fifo():
        if ir < imin:
            imin = ir
        if ir > imax:
            imax = ir
        if ir >= SAT:
            n_sat += 1
        count += 1
        hrv.add_sample(ir)
    if time.ticks_diff(now, last) >= 1000:
        last = now
        m = hrv.metrics()
        print("t=%3ds  HR=%5.1f SDNN=%5.1f RMSSD=%5.1f pNN50=%5.1f nn=%3d ac=%6.1f rej=%2d"
              % (time.ticks_diff(now, t0) // 1000, m['hr'], m['sdnn'], m['rmssd'],
                 m['pnn50'], m['nn'], m['ac'], m['rej']))
    time.sleep_ms(10)

# ---------------- 汇总 ----------------
m = hrv.metrics()
print("")
print("===== MAX30102 测试汇总 =====")
print("采样点数 : %d  (%.1f Hz)   饱和样本 %d" % (count, count / float(DURATION_S), n_sat))
print("IR 范围   : %d ~ %d   (AC/DC 参考)" % (imin, imax))
print("HR       : %.1f bpm" % m['hr'])
print("SDNN     : %.1f ms" % m['sdnn'])
print("RMSSD    : %.1f ms" % m['rmssd'])
print("pNN50    : %.1f %%" % m['pnn50'])
print("sdsd     : %.1f ms" % m['sdsd'])
print("有效 RR  : %d  (剔除离群 %d)" % (m['nn'], m['rej']))
print("AC 幅度   : %.1f" % m['ac'])
print("RR 序列   : %s" % hrv.rr[-40:])

q = []
if n_sat > 0:
    q.append("ADC饱和")
if m['nn'] < 30:
    q.append("RR不足30")
if not (40 <= m['hr'] <= 150):
    q.append("HR异常")
if not (10 <= m['rmssd'] <= 120):
    q.append("RMSSD异常")
print("整体判定 :", "GOOD (信号可用于 HRV 特征提取)" if not q else "CHECK -> " + ", ".join(q))
