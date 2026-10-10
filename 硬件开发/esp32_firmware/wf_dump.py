# -*- coding: utf-8 -*-
"""
wf_dump.py - 原始波形抓取诊断 (在 ESP32 上跑)

按排障铁律"先抓原始波形, 再改算法": 不改任何滤波/判决参数, 只把
MAX30102 的原始 IR 序列 + 采样节奏 + 检出的 RR 序列原样打出来。

诊断点:
  A) 实际采样率 —— add_sample 按样本序号×10ms 计时。若芯片实际输出率
     偏离 100Hz, 整条时间轴被缩放 → RR/HR/RMSSD 系统性失真。
  B) FIFO 溢出 —— 溢出=丢样本, 同样破坏时间轴。
  C) 波形 DC / 峰峰值 —— 判断灌注是否够。
  D) RR 序列原始分布 —— 区分"逐拍抖动"与"偶发坏拍"。

内存约束: ESP32 存不下长波形(实测 15s 的 list 会 MemoryError, 见
  "列表扩容需 8KB 连续堆" )。故**逐秒打印当秒的原始样本后立即丢弃**,
  只保留一个 1 秒的小缓冲 → 与固件的内存红线一致。

用法:
    mpremote connect COM6 run wf_dump.py          # 8 秒
    mpremote connect COM6 run wf_dump_15.py       # 15 秒

⚠️ 运行前先把指腹放好并保持不动。
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

DUMP_SEC = globals().get("DUMP_SEC", 8)
CHUNK = 100
LINE = "=" * 76


def main():
    print("\n" + LINE)
    print("  原始波形抓取诊断  (%d 秒, 保持不动)" % DUMP_SEC)
    print(LINE)

    i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
              sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
    print("I2C 扫描: %s" % ", ".join("0x%02X" % a for a in i2c.scan()))

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
    try:
        spo2 = ppg._rreg(ppg.REG_SPO2_CONFIG)
        print("SPO2_CONFIG=0x%02X (ADC_RGE=%d, SR=%d, PW=%d)  FIFO_CONFIG=0x%02X"
              % (spo2, (spo2 >> 5) & 3, (spo2 >> 2) & 7, spo2 & 3,
                 ppg._rreg(ppg.REG_FIFO_CONFIG)))
    except Exception:
        pass

    imu = None
    if MPU6050 is not None and config.MPU6050_ENABLED:
        try:
            imu = MPU6050(i2c, addr=config.MPU6050_ADDR)
            imu.begin(accel_range=2, gyro_range=250,
                      dlpf=config.MPU6050_DLPF, rate_hz=config.MPU6050_RATE_HZ)
            imu.motion_threshold_g = config.MPU6050_MOTION_G
            imu.gyro_threshold_dps = config.MPU6050_GYRO_DPS
            imu.tilt_threshold_deg = config.MPU6050_TILT_DEG
            imu.calibrate_gyro(1.2)
            imu.init_gravity(0.6)
            print("MPU6050 OK (陀螺零偏已标定)")
        except OSError:
            imu = None

    h = HRVAnalyzer(fs=config.MAX30102_SAMPLE_RATE,
                    window_sec=config.HRV_WINDOW_SEC,
                    min_peak_ms=config.PPG_MIN_PEAK_MS,
                    max_peak_ms=config.PPG_MAX_PEAK_MS,
                    peak_ratio=config.PPG_PEAK_RATIO)

    sec_buf = []          # 只存当秒样本(打印后清空) —— 内存红线
    sec_cnt = []
    sec_dc = []
    sec_pp = []
    movs = 0
    ovf0 = getattr(ppg, "ovf_count", 0)
    t0 = time.ticks_ms()
    last = t0
    n_tot = 0

    print("\n⏳ 采集 %d 秒 ... 请保持不动\n" % DUMP_SEC)
    while True:
        now = time.ticks_ms()
        if time.ticks_diff(now, t0) >= DUMP_SEC * 1000:
            break
        moving = False
        if imu is not None:
            moving, _d = imu.is_motion()
            if moving:
                movs += 1
        try:
            batch = ppg.read_fifo()
        except OSError:
            batch = []
        for (_r, ir) in batch:
            sec_buf.append(ir)
            h.add_sample(ir, motion=moving)
            n_tot += 1

        if time.ticks_diff(now, last) >= 1000:
            last = now
            m = h.metrics()
            sec_cnt.append(len(sec_buf))
            sec_pp.append(max(sec_buf) - min(sec_buf) if sec_buf else 0)
            sec_dc.append(sum(sec_buf) / len(sec_buf) if sec_buf else 0)
            print("  [%2ds] 本秒样本=%3d 累计=%4d HR=%.1f RMSSD=%.1f nn=%d "
                  "rej=%d Q=%.2f" % (len(sec_cnt), len(sec_buf), n_tot,
                                     m["hr"], m["rmssd"], m["nn"], m["rej"],
                                     m["sig_quality"]))
            # 当秒原始样本逐行打印(供离线复算), 打印后立即释放
            for s in range(0, len(sec_buf), CHUNK):
                print("  RAW,%d,%d,%s" % (len(sec_cnt), s,
                      ",".join(str(v) for v in sec_buf[s:s + CHUNK])))
            del sec_buf[:]

    # ---------------- A) 采样节奏 ----------------
    n = n_tot
    expect = DUMP_SEC * config.MAX30102_SAMPLE_RATE
    ovf = getattr(ppg, "ovf_count", 0) - ovf0
    real_rate = (n + ovf) / float(DUMP_SEC)
    print("\n" + LINE)
    print("  A) 采样节奏")
    print(LINE)
    print("  实际样本数 %d  期望 %d  达成 %.1f%%" %
          (n, expect, 100.0 * n / expect))
    print("  每秒样本数 %s" % sec_cnt)
    print("  FIFO 溢出(采集窗口内增量) = %d" % ovf)
    print("  ⇒ 芯片实际输出率 ≈ (收到 %d + 溢出 %d) / %ds = %.1f Hz"
          % (n, ovf, DUMP_SEC, real_rate))
    if sec_cnt:
        lo_c, hi_c = min(sec_cnt), max(sec_cnt)
        print("  每秒样本数 最小 %d 最大 %d 波动 %d" % (lo_c, hi_c, hi_c - lo_c))
        if hi_c - lo_c > 8:
            print("  ✘ 读取节奏不稳(突发丢样本) —— 时间轴会被压缩")
    if ovf > 0:
        print("  ✘ FIFO 溢出 %d 个样本被丢弃 —— 时间轴被压缩 → RR 失真" % ovf)
    else:
        print("  ✔ 无 FIFO 溢出")
    if abs(real_rate - 100.0) > 5.0:
        print("  ✘ 实际输出率 %.1f Hz 偏离 100 Hz 超 5%% —— 时间轴被缩放!" % real_rate)
        print("    add_sample 固定按 10ms/样本计时; 应改为按实测输出率计时。")
    else:
        print("  ✔ 输出率 %.1f Hz 与设定基本一致(偏差 %.1f%%), 时间轴可信"
              % (real_rate, 100.0 * (real_rate - 100.0) / 100.0))
        if abs(real_rate - 100.0) > 1.0:
            print("    (注意: 有 %.1f%% 的系统偏差, 会让 RR 同比例偏移)"
                  % abs(real_rate - 100.0))

    # ---------------- B) 波形幅度 ----------------
    if sec_dc:
        dc = sum(sec_dc) / len(sec_dc)
        med_pp = sorted(sec_pp)[len(sec_pp) // 2]
        print("\n" + LINE)
        print("  B) 波形幅度")
        print(LINE)
        print("  IR 直流(逐秒均值) = %.0f" % dc)
        print("  逐秒峰峰 %s" % sec_pp)
        print("  逐秒峰峰中位 = %d  →  峰峰/直流 = %.3f%%"
              % (med_pp, 100.0 * med_pp / dc if dc else 0))
        if dc >= 262100:
            print("  ✘ ADC 饱和 —— 先查量程/增益, 不要动算法")
        elif med_pp < 80:
            print("  ✘ 幅度过小 —— 峰位必然抖动, HRV 不可信")
        else:
            print("  ✔ 幅度充足")

    # ---------------- C) RR 序列 ----------------
    rr = list(h.rr)
    print("\n" + LINE)
    print("  C) RR 序列 (检出 %d 拍, 检测剔除 %d, 清洗后 %d)"
          % (len(rr), h.rr_rej, len(h._clean_rr())))
    print(LINE)
    if rr:
        print("  RR(ms):")
        s = "    "
        for i, v in enumerate(rr):
            s += "%5.0f" % v
            if (i + 1) % 16 == 0:
                print(s)
                s = "    "
        if s.strip():
            print(s)
        diffs = [rr[i] - rr[i - 1] for i in range(1, len(rr))]
        if diffs:
            ad = sorted(abs(x) for x in diffs)
            print("\n  |ΔRR| 中位 %d  90分位 %d  最大 %d ms"
                  % (ad[len(ad) // 2], ad[int(len(ad) * 0.9)], ad[-1]))
            print("  |ΔRR|>50ms 比例 %.1f%%  (静息应 <10%%)"
                  % (100.0 * sum(1 for x in ad if x > 50) / len(ad)))
            print("  ΔRR(ms): " + " ".join("%+.0f" % v for v in diffs))
            # 贡献度: 单个最大 ΔRR 对 RMSSD 的贡献
            rms_all = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
            worst = max(diffs, key=lambda x: abs(x))
            rest = [d for d in diffs if d != worst]
            rms_wo = (sum(d * d for d in rest) / len(rest)) ** 0.5 if rest else 0.0
            print("\n  RMSSD(全部)        = %.1f ms" % rms_all)
            print("  RMSSD(去掉最差一拍) = %.1f ms   (被那一拍单独抬高了 %.1f ms)"
                  % (rms_wo, rms_all - rms_wo))
            if rms_all - rms_wo > 0.35 * rms_all:
                print("  ⇒ 少数坏拍主导了 RMSSD —— 属**伪迹/漏搏**, 应强化坏拍剔除,")
                print("    而不是调整滤波器或阈值。")
    else:
        print("  ✘ 未检出任何 RR")

    print("\n" + LINE)
    print("  运动样本 %d / %d" % (movs, n))
    print(LINE)


main()
