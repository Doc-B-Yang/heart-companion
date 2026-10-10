# -*- coding: utf-8 -*-
"""
finger_test.py - 手指测试: 运动伪迹抑制的"有/无 IMU"对照

目的: 用同一路 PPG 数据同时喂两个 HRV 分析器 —— 一个带 IMU 运动门控,
      一个不带 —— 直接看出运动伪迹对 HRV 指标的污染幅度, 以及抑制的收益。

测试编排 (共 45 秒, 脚本会打印提示):
    0~15s   静止     手放好不要动       ← 建立可信基线
   15~30s   晃动     轻轻晃动手/捏传感器 ← 制造运动伪迹
   30~45s   恢复     重新放稳不要动

⚠️ 运行前请先把指腹放到 MAX30102 上 —— 开机 auto_gain 会依据当时的
   IR 直流值调 LED 电流, 手指不在就会把电流定错档。

用法:
    mpremote connect COM6 run finger_test.py
    mpremote connect COM6 run finger_test_long.py   # 90 秒加长版
"""

import time
from machine import I2C, Pin

import config
from max30102 import MAX30102
from hrv import HRVAnalyzer

try:
    from mpu6050 import MPU6050
except ImportError:
    MPU6050 = None

# --- 可从启动器覆盖 (见 finger_test_long.py) ---
# `mpremote run` 不能传脚本参数, 且 MicroPython 的 ESP32 端口没有 sys.argv,
# 所以加长版由启动器先注入 FINGER_SEC / STILL1 / SHAKE_B 再 exec 本文件。
FINGER_SEC = globals().get("FINGER_SEC", 45)
STILL1     = globals().get("STILL1", 15)
SHAKE_A    = STILL1
SHAKE_B    = globals().get("SHAKE_B", 30)

LINE = "=" * 78


def phase_of(t):
    if t < STILL1:
        return "静止"
    if t < SHAKE_B:
        return "晃动"
    return "恢复"


def main():
    print("\n" + LINE)
    print("  运动伪迹抑制对照测试  (MAX30102 + MPU6050)")
    print(LINE)

    i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
              sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
    scan = ["0x%02X" % a for a in i2c.scan()]
    print("I2C 扫描: %s" % (", ".join(scan) if scan else "(无器件)"))

    # ---------- PPG ----------
    ppg = MAX30102(i2c, config.MAX30102_ADDR)
    if not ppg.present():
        print("✘ MAX30102 未找到, 退出")
        return
    ppg.begin(sample_rate=config.MAX30102_SAMPLE_RATE,
              led_current=config.LED_IR_CURRENT,
              avg=config.MAX30102_FIFO_AVG,
              adc_range=config.MAX30102_ADC_RANGE)
    code = ppg.auto_gain()
    print("MAX30102 OK  auto_gain: LED=0x%02X  IR_DC=%.0f  in_range=%s"
          % (code, ppg.gain_dc, ppg.gain_ok))
    if not ppg.gain_ok:
        print("  ⚠ 增益未落入目标区间 —— 检查指腹是否压实/是否遮挡环境光")
    time.sleep_ms(800)

    # ---------- IMU ----------
    imu = None
    if MPU6050 is not None and config.MPU6050_ENABLED:
        try:
            imu = MPU6050(i2c, addr=config.MPU6050_ADDR)
            imu.who_am_i()
            imu.begin(accel_range=2, gyro_range=250,
                      dlpf=config.MPU6050_DLPF,
                      rate_hz=config.MPU6050_RATE_HZ)
            imu.motion_threshold_g = config.MPU6050_MOTION_G
            imu.gyro_threshold_dps = config.MPU6050_GYRO_DPS
            imu.tilt_threshold_deg = config.MPU6050_TILT_DEG
            print("MPU6050 OK  WHO_AM_I=0x%02X (%s)" % (imu.who_am_i(), imu.chip))
        except OSError as e:
            print("MPU6050 初始化失败: %s" % e)
            imu = None
    if imu is None:
        print("✘ 无 IMU → 无法做对照 (本测试必须两块都在线)")
        return

    print("\n⏳ 静止标定 %.1fs —— 请保持手/板子完全静止 (不要松开手指) ..."
          % (config.MPU6050_CALIB_SEC + config.MPU6050_GRAVITY_SEC + 0.8))
    bx, by, bz = imu.calibrate_gyro(config.MPU6050_CALIB_SEC)
    imu.init_gravity(config.MPU6050_GRAVITY_SEC)
    g_ref, noise = imu.calibrate_still(0.8)
    print("  陀螺零偏 (%+.2f,%+.2f,%+.2f) °/s | |a|基线 %.4f g | 噪声 %.4f g"
          % (bx, by, bz, g_ref, noise))

    # ---------- 两个分析器: 同一路数据, 一个门控一个不门控 ----------
    kw = dict(fs=config.MAX30102_SAMPLE_RATE,
              window_sec=config.HRV_WINDOW_SEC,
              min_peak_ms=config.PPG_MIN_PEAK_MS,
              max_peak_ms=config.PPG_MAX_PEAK_MS,
              peak_ratio=config.PPG_PEAK_RATIO)
    h_gate = HRVAnalyzer(**kw)      # 有 IMU 门控
    h_raw = HRVAnalyzer(**kw)       # 无门控(对照组)

    print("\n" + LINE)
    print("  测试编排:")
    print("    0 ~ %-2ds  静止  —— 手放好, 完全不要动" % STILL1)
    print("    %d ~ %-2ds  晃动  —— 请轻轻晃动手 / 轻捏传感器" % (STILL1, SHAKE_B))
    print("    %d ~ %-2ds  恢复  —— 重新放稳, 不要动" % (SHAKE_B, FINGER_SEC))
    print(LINE)
    print("  现在开始 —— 请先保持静止 ...\n")

    print("  %-4s %-4s %-7s %-6s %-5s | %-16s | %-16s | %s"
          % ("t", "阶段", "IR_DC", "PI%", "Q", "门控HR/RMSSD/nn",
             "未门控HR/RMSSD/nn", "运动"))

    t0 = time.ticks_ms()
    last_rep = t0
    stage = {"静止": [[], [], []], "晃动": [[], [], []], "恢复": [[], [], []]}
    #  每个阶段收集: [gated_rmssd, raw_rmssd, raw_hr- gated_hr]
    seen = -1
    mo_ema = 0.0
    mo_n = 0

    while True:
        now = time.ticks_ms()
        el = time.ticks_diff(now, t0) / 1000.0
        if el >= FINGER_SEC:
            break

        moving, devs = imu.is_motion()
        if moving:
            mo_n += 1
        mo_ema = 0.95 * mo_ema + 0.05 * (1.0 if moving else 0.0)

        try:
            for (red, ir) in ppg.read_fifo():
                h_gate.add_sample(ir, motion=moving)
                h_raw.add_sample(ir, motion=False)   # 对照组: 不丢弃任何东西
        except OSError:
            pass

        if time.ticks_diff(now, last_rep) >= 1000:
            last_rep = now
            sec = int(el)
            if sec != seen:
                seen = sec
                ph = phase_of(sec)
                mg = h_gate.metrics()
                mr = h_raw.metrics()
                print("  %-4d %-4s %-7.0f %-6.2f %-5.2f | %5.1f %5.1f %4d |"
                      " %5.1f %5.1f %4d | %s%s"
                      % (sec, ph, h_gate.dc or 0, h_gate.perfusion_pct() * 1.0,
                         mg["sig_quality"],
                         mg["hr"], mg["rmssd"], mg["nn"],
                         mr["hr"], mr["rmssd"], mr["nn"],
                         "MOV" if moving else "...",
                         "  |ω|=%.0f" % devs[1] if moving else ""))
                if ph in stage and mg["nn"] >= 10 and mr["nn"] >= 10:
                    stage[ph][0].append(mg["rmssd"])
                    stage[ph][1].append(mr["rmssd"])
                    stage[ph][2].append(mr["hr"] - mg["hr"])

        time.sleep_ms(10)

    # ---------------- 汇总 ----------------
    mg = h_gate.metrics()
    mr = h_raw.metrics()
    print("\n" + LINE)
    print("  汇总 (末次窗口)")
    print(LINE)
    print("  %-14s %7s %7s %7s %7s %6s" %
          ("", "HR", "SDNN", "RMSSD", "pNN50", "nn"))
    print("  %-14s %7.1f %7.1f %7.1f %7.1f %6d" %
          ("门控(带IMU)", mg["hr"], mg["sdnn"], mg["rmssd"], mg["pnn50"],
           mg["nn"]))
    print("  %-14s %7.1f %7.1f %7.1f %7.1f %6d" %
          ("未门控(对照)", mr["hr"], mr["sdnn"], mr["rmssd"], mr["pnn50"],
           mr["nn"]))

    print()
    print("  分阶段 RMSSD 对比 (数值取自各阶段稳定期, nn>=10):")
    print("  %-6s %-10s %-10s %-8s %s" %
          ("阶段", "门控RMSSD", "未门控", "膨胀倍数", "样本"))
    for ph in ("静止", "晃动", "恢复"):
        g, r, _dh = stage[ph]
        if g and r:
            gm = sum(g) / len(g)
            rm = sum(r) / len(r)
            print("  %-6s %-10.1f %-10.1f %-8.2f %d" %
                  (ph, gm, rm, (rm / gm if gm > 1e-6 else 0), len(g)))
        else:
            print("  %-6s (数据不足)" % ph)

    print()
    print("  运动统计: 运动样本 %d 个 (%.1f%%), 运动段 %d 次, 门控丢弃样本 %d"
          % (mo_n, 100.0 * mo_ema, mg["segs"], mg["motion_n"]))
    print("  信号质量: 门控 Q=%.3f  未门控 Q=%.3f" %
          (mg["sig_quality"], mr["sig_quality"]))

    d_rm = mr["rmssd"] - mg["rmssd"]
    print()
    if mg["nn"] < 10:
        print("  ⚠ 门控路径有效拍数不足(%d) —— 可能一直在动, 或指腹贴合不好" % mg["nn"])
        print("    请复查: PI%.2f%% (需 >0.3)  运动占比 %.1f%%" %
              (mg["pi"], mg["motion"] * 100))
    elif abs(d_rm) < 5:
        print("  ➜ 两条路径 RMSSD 接近(差 %.1f ms): 本次全程较安静, 运动伪迹少。" % d_rm)
        print("    这本身是好结果 —— 说明静止测量时 IMU 门控不会误伤真实 HRV。")
    else:
        print("  ➜ 未门控路径 RMSSD 比门控高 %.1f ms —— 差值即为被 IMU 拦下的运动伪迹。" % d_rm)
        print("    未门控的高 RMSSD 是**假的**(伪迹导致), 会直接污染情绪识别。")
    print(LINE)


main()
