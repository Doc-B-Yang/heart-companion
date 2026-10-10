# -*- coding: utf-8 -*-
"""
mpu_test.py - MPU6050 硬件测试 (在 ESP32 上运行)

测试项:
  1. I2C 总线扫描      —— 确认 0x68 在线, 并列出同总线其它器件
  2. 芯片识别          —— WHO_AM_I + **寄存器写-回读校验**(比 ID 更可靠)
  3. 原始数据          —— 排除"全 0 / 全饱和"这类接线不良特征
  4. 静态质量          —— 静止 3 秒测噪声底(加速度 + 陀螺), 校验重力模长
  5. 实时运动监视      —— 观察运动标志是否正确翻转
  6. INT 中断通路      —— 数上升沿, 证明 INT 那根线是通的
  7. 运动阈值标定      —— 由噪声底反推推荐阈值

用法:
    mpremote connect COM6 run mpu_test.py            # 标准测试(含 12s 监视)
    mpremote connect COM6 run mpu_test_shake.py      # 30s 长监视, 验证运动响应
    mpremote connect COM6 run mpu_test_quick.py      # 跳过监视, 只做硬件体检

  ⚠️ 两个硬约束: (1) `mpremote run` 不支持传脚本参数;
                 (2) MicroPython 的 ESP32 端口没有 sys.argv。
     所以模式通过"启动器注入 MPU_MODE 全局变量"传递 —— 见 mpu_test_shake.py。

  shake 模式给出 30 秒窗口, 期间请**持续轻敲/晃动板子**, 用于确认
  运动判定确实能触发(标准模式的 12 秒窗口容易被错过)。
"""

import sys
import time

import config

try:
    from machine import I2C, Pin
except ImportError:
    print("本脚本必须在 ESP32 (MicroPython) 上运行")
    sys.exit(1)

from mpu6050 import MPU6050


# ---- 运行模式 ----------------------------------------------------------
# ⚠️ MicroPython 的 ESP32 端口**没有 sys.argv**, 且 `mpremote run` 也不支持
#    给脚本传参数。因此模式改用"启动器注入全局变量"的方式:
#      启动器里先写  MPU_MODE = "shake"
#      再  exec(open("mpu_test.py").read())
#    exec 共用调用方的 globals, 所以这里能读到 MPU_MODE。
_MODE = globals().get("MPU_MODE", "")
try:
    _av = sys.argv                       # REPL 下手动设 sys.argv 也能用
    if "--quick" in _av:
        _MODE = "quick"
    if "--shake" in _av:
        _MODE = "shake"
except AttributeError:
    pass

QUICK = (_MODE == "quick")
SHAKE = (_MODE == "shake")
MONITOR_SEC = 0 if QUICK else (30 if SHAKE else 12)

LINE = "=" * 62


def hr(title=""):
    if title:
        print("\n" + "-" * 62)
        print("  " + title)
        print("-" * 62)
    else:
        print("\n" + LINE)


# ============================================================ 1. I2C 扫描
def t_scan():
    hr("【1/7】I2C 总线扫描")
    i2c = I2C(config.I2C_ID, scl=Pin(config.I2C_SCL),
              sda=Pin(config.I2C_SDA), freq=config.I2C_FREQ)
    devs = i2c.scan()
    names = {0x57: "MAX30102(PPG)", 0x48: "ADS1115(ADC)", 0x68: "MPU6050(IMU)",
             0x69: "MPU6050(AD0=VDD)", 0x76: "BMP280?", 0x77: "BMP280?"}
    if not devs:
        print("  ✘ 总线上没有任何器件 —— SDA/SCL 接反、没共地, 或模块没供电")
        return None
    for d in devs:
        print("    0x%02X  %s" % (d, names.get(d, "未知器件")))
    if 0x68 in devs or 0x69 in devs:
        print("  ✔ MPU6050 在线")
    else:
        print("  ✘ 未发现 MPU6050 (期望 0x68)")
        print("    排查: AD0 是否接 GND? SDA=GPIO21 / SCL=GPIO22 是否接反? VCC 是否 3.3V?")
        return None
    return i2c


# ============================================================ 2. 芯片识别
def t_ident(imu):
    hr("【2/7】芯片识别 (WHO_AM_I 寄存器 0x75)")
    v = imu.who_am_i()
    print("    读取值 = 0x%02X  →  %s" % (v, imu.chip))
    if v == 0x68:
        print("  ✔ 正品 MPU6050")
    elif imu.is_compatible():
        print("  ✔ 寄存器兼容片 —— MPU6050 已停产多年, 市售 GY-521 基本都是兼容 die,")
        print("    只要寄存器读写正常即可。下面用**写-回读校验**做最终确认。")
    else:
        print("  ⚠ 未收录的 ID, 用写-回读校验判断是否可用")

    print()
    print("    寄存器写-回读校验:")
    ok, rows = imu.verify_registers()
    for name, want, got in rows:
        print("      %-14s 写 0x%02X → 读 0x%02X   %s"
              % (name, want, got, "✔" if want == got else "✘"))
    if ok:
        print("  ✔ 6/6 项一致 —— I2C 读写通路完好, 芯片可按 MPU6050 使用")
        return True
    n_ok = sum(1 for _n, w, g in rows if w == g)
    print("  ✘ 仅 %d/6 项一致 —— I2C 通信不可靠" % n_ok)
    print("    排查: 杜邦线松动 / 面包板接触不良 / 上拉缺失 / 降速到 100kHz 复测")
    return False


# ============================================================ 3. 原始数据
def t_raw(imu):
    hr("【3/7】原始数据读取 (突发读 14 字节 @0x3B)")
    try:
        v = imu.read_raw()
    except Exception as e:
        print("  ✘ 读取失败:", repr(e))
        return False
    print("    ax=%6d ay=%6d az=%6d | temp=%6d | gx=%6d gy=%6d gz=%6d" % v)
    if all(x == 0 for x in v):
        print("  ✘ 全为 0 —— 芯片未唤醒或通信异常")
        return False
    if all(x in (32767, -32768, -1) for x in v):
        print("  ✘ 全为饱和值 —— 通信异常")
        return False
    print("  ✔ 数据有效(非全 0 / 非饱和)")
    d = imu.read()
    print("    换算后: a=(%.3f, %.3f, %.3f) g | ω=(%.2f, %.2f, %.2f) °/s | T=%.1f °C"
          % (d["ax"], d["ay"], d["az"], d["gx"], d["gy"], d["gz"], d["temp"]))
    return True


# ============================================================ 4. 静态质量
def t_static(imu):
    hr("【4/7】静态质量 (安静放置 3 秒)")
    if SHAKE:
        print("    ⚠️ 本段必须**完全静止** —— 请先松手, 不要碰板子/桌子!")
        for k in (4, 3, 2, 1):
            print("      %d ..." % k)
            time.sleep(1)
    print("    ⏳ 采集中, 请保持板子静止 ...")
    ka = 16384.0 if imu.accel_range == 2 else \
        (8192.0 if imu.accel_range == 4 else
         (4096.0 if imu.accel_range == 8 else 2048.0))
    mags = []
    axs = []; ays = []; azs = []
    gxs = []; gys = []; gzs = []
    t0 = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), t0) < 3000:
        ax, ay, az, _t, gx, gy, gz = imu.read_raw()
        x = ax / ka; y = ay / ka; z = az / ka
        mags.append((x * x + y * y + z * z) ** 0.5)
        axs.append(x); ays.append(y); azs.append(z)
        d = imu.read()
        gxs.append(d["gx"]); gys.append(d["gy"]); gzs.append(d["gz"])
        time.sleep_ms(10)

    def stat(a):
        m = sum(a) / len(a)
        return m, (sum((v - m) ** 2 for v in a) / len(a)) ** 0.5

    n = len(mags)
    mean, std = stat(mags)
    m_ax, s_ax = stat(axs)
    m_ay, s_ay = stat(ays)
    m_az, s_az = stat(azs)
    _mgx, s_gx = stat(gxs)
    _mgy, s_gy = stat(gys)
    _mgz, s_gz = stat(gzs)
    g_bias = (sum(gxs) / len(gxs), sum(gys) / len(gys), sum(gzs) / len(gzs))
    g_noise = max(s_gx, s_gy, s_gz)

    print("    样本数 = %d" % n)
    print("    加速度 |a| : 均值 %.4f g | 标准差 %.4f g | 峰峰 %.4f g"
          % (mean, std, max(mags) - min(mags)))
    print("    各轴均值   : ax=%+.3f  ay=%+.3f  az=%+.3f g" % (m_ax, m_ay, m_az))
    print("    各轴噪声   : ax %.4f  ay %.4f  az %.4f g" % (s_ax, s_ay, s_az))
    print("    陀螺零偏   : gx=%+.2f  gy=%+.2f  gz=%+.2f °/s  (规格 ±20 内属正常)"
          % g_bias)
    print("    陀螺噪声   : %.3f °/s (取最大轴)" % g_noise)

    ok_g = abs(mean - 1.0) <= 0.05
    ok_n = std <= 0.02
    print("    重力校验   : 偏差 %+.4f g  %s (容差 ±0.05)" % (mean - 1.0, "✔" if ok_g else "✘"))
    print("    噪声校验   : 标准差 %.4f  %s (静止应 ≤0.02)" % (std, "✔" if ok_n else "⚠"))

    axes = (("X", m_ax), ("Y", m_ay), ("Z", m_az))
    dom = max(axes, key=lambda p: abs(p[1]))
    print("    主导轴     : %s (%.3f g) → %s" % (dom[0], dom[1],
          "板子近似平放, 该轴竖直朝%s" % ("上" if dom[1] > 0 else "下")))
    imu.set_gyro_bias(g_bias)
    imu.init_gravity(0.3)
    print("    ✔ 已记录陀螺零偏与重力方向参考, 运动判据将扣除它们")
    return mean, std, g_noise


# ============================================================ 5. 实时监视
def t_monitor(imu):
    hr("【5/7】实时运动监视 (%d 秒)" % MONITOR_SEC)
    if SHAKE:
        print("    ✅ 现在开始! 请**持续轻敲 / 晃动板子**, 直到本段结束")
    else:
        print("    👉 可敲桌子 / 轻晃板子 / 点触传感器, 观察触发标志")
    print()
    print("    触发标志含义: a=加速度判据  ω=角速度判据  t=倾角判据")
    print()
    print("    %-6s %-7s %-7s %-7s %-7s %-6s %s"
          % ("t(s)", "|a|", "偏差a", "|ω|", "倾角°", "触发", "偏差柱"))
    t0 = time.ticks_ms()
    shown = -1
    peak_dev = 0.0
    peak_omega = 0.0
    peak_tilt = 0.0
    cnt = {"a": 0, "w": 0, "t": 0, "any": 0, "n": 0}
    while time.ticks_diff(time.ticks_ms(), t0) < MONITOR_SEC * 1000:
        d = imu.read()
        g = (d["ax"] ** 2 + d["ay"] ** 2 + d["az"] ** 2) ** 0.5
        hit, (dev, omega, tilt) = imu.is_motion()   # 三判据取或
        cnt["n"] += 1
        if hit:
            cnt["any"] += 1
        if dev > imu.motion_threshold_g:
            cnt["a"] += 1
        if omega > imu.gyro_threshold_dps:
            cnt["w"] += 1
        if tilt > imu.tilt_threshold_deg:
            cnt["t"] += 1
        if dev > peak_dev:
            peak_dev = dev
        if omega > peak_omega:
            peak_omega = omega
        if tilt > peak_tilt:
            peak_tilt = tilt
        el = time.ticks_diff(time.ticks_ms(), t0) / 1000.0
        if int(el * 3) > shown:
            shown = int(el * 3)
            bar = "#" * min(30, int(dev * 60))
            flag = ("a" if dev > imu.motion_threshold_g else "") \
                 + ("ω" if omega > imu.gyro_threshold_dps else "") \
                 + ("t" if tilt > imu.tilt_threshold_deg else "")
            print("    %-6.2f %-7.3f %-7.3f %-7.1f %-7.1f %-6s %s"
                  % (el, g, dev, omega, tilt, flag if flag else "...", bar))
        time.sleep_ms(10)
    print()
    print("    峰值: 偏差a = %.3f g | |ω| = %.1f °/s | 倾角 = %.1f°"
          % (peak_dev, peak_omega, peak_tilt))
    print("    判为运动的采样比例 = %.1f%% (%d/%d)"
          % (100.0 * cnt["any"] / max(1, cnt["n"]), cnt["any"], cnt["n"]))
    print()
    print("    各判据**单独**能覆盖多少采样 (阈值: a>%.2fg, ω>%.0f°/s, t>%.0f°):"
          % (imu.motion_threshold_g, imu.gyro_threshold_dps, imu.tilt_threshold_deg))
    for key, label in (("a", "① 加速度模长偏差"), ("w", "② 角速度"),
                       ("t", "③ 重力方向倾角")):
        c = cnt[key]
        print("      %-18s %5d / %d  = %5.1f%%" % (label, c, cnt["n"],
              100.0 * c / max(1, cnt["n"])))
    print()
    hit_a = cnt["a"] > 0
    hit_g = cnt["w"] > 0
    hit_t = cnt["t"] > 0
    if cnt["any"] == 0:
        print("  ⚠ 全程未触发运动判定(板子基本没动?)")
        print("    若要验证运动响应: mpremote connect COM6 run mpu_test_shake.py")
    elif hit_g or hit_t:
        print("  ✔ 运动检测有效")
        if not hit_a and (hit_g or hit_t):
            print("     ⚠️ 重要: 加速度模长判据**完全没触发**, 而角速度/倾角判据大量触发。")
            print("        这是必然的 —— 旋转不改变加速度矢量的模长(重力恒为 1g),")
            print("        所以 ||a|-1g| 对纯转动**天然不敏感**。")
            print("        手部抖动的主动成分正是转动 ⇒ **只用加速度模长做伪迹抑制会大面积漏检**。")
        else:
            print("     (本次加速度判据也触发了, 说明含平移分量; 但仍需保留角速度判据)")
    return peak_dev, peak_omega, peak_tilt, cnt


# ============================================================ 6. INT 通路
def t_int(imu):
    hr("【6/7】INT 中断通路验证 (GPIO%d)" % imu.int_pin_no)
    if imu.int_pin_no is None:
        print("  - 未配置 INT 引脚, 跳过")
        return None
    p = Pin(imu.int_pin_no, Pin.IN)
    idle = p.value()
    print("    引脚空闲电平 = %d (空闲应为 0; 恒为 1 → 可能悬空或接了 VCC)" % idle)
    print("    使能 DATA_RDY 中断(50us 脉冲), 计数 3 秒上升沿 ...")

    try:
        imu.enable_data_ready_int(latch=False, active_low=False)
    except Exception as e:
        print("  ✘ 配置中断失败:", repr(e))
        return None

    cnt = imu.counter()
    cnt.start()
    time.sleep(3)
    edges = cnt.stop()
    imu.disable_int()

    expect = int(imu.rate_hz * 3)
    print("    3 秒内上升沿 = %d 次 (按 %.0f Hz, 期望约 %d)" % (edges, imu.rate_hz, expect))
    if edges == 0:
        print("  ✘ 未捕获到任何边沿")
        print("    排查: ① INT 是否真接在 GPIO%d" % imu.int_pin_no)
        print("          ② 杜邦线是否插在同一列的同一槽位")
        print("          ③ 该引脚是否被其它外设占用")
        print("    注: INT 非必需 —— 主固件用轮询读加速度, 不接也能工作")
        return 0
    if edges < expect * 0.5:
        print("  ⚠ 边沿偏少(轮询退化或引脚抖动), 但通路已通")
    else:
        print("  ✔ INT 通路正常, 采样节拍与设定一致")
    return edges


# ============================================================ 7. 阈值标定
def t_threshold(imu, noise, peak_dev, peak_omega, cnt=None):
    hr("【7/7】运动阈值标定")
    thr = imu.motion_threshold_g
    print("    静态噪声底       = %.4f g" % noise)
    print("    噪声余量         = %.1f 倍 (阈值/噪声)" % (thr / max(noise, 1e-6)))
    print("    建议最低阈值     = %.3f g  (8×噪声底)" % max(0.05, round(noise * 8, 3)))
    print("    当前 config 阈值 = %.2f g" % thr)
    if noise * 5 < thr:
        print("  ✔ 阈值远高于噪声, 静止时不会误触发")
    else:
        print("  ✘ 阈值与噪声差距不足, 静止时可能误判为运动"
              "(shake 模式下若刚晃动过, 此段读数无效 —— 需重新静止测量)")
    print()
    print("    三个运动判据:")
    print("      ① 加速度模长偏差  ||a| - |g_ref|| > %.2f g   → 平移/冲击" % thr)
    print("      ② 角速度模长      |ω - 零偏|      > %.0f °/s → 转动"
          % imu.gyro_threshold_dps)
    print("      ③ 重力方向倾角    与参考方向夹角   > %.0f°   → 转动/倾斜"
          % imu.tilt_threshold_deg)
    bx, by, bz = imu.gyro_bias()
    print("    已标定陀螺零偏 = (%+.2f, %+.2f, %+.2f) °/s" % (bx, by, bz))
    print()
    if cnt:
        na = cnt["a"]; nw = cnt["w"]; nt = cnt["t"]; nn = max(1, cnt["n"])
        print("    本次实测各判据单独覆盖率: ① %.0f%%   ② %.0f%%   ③ %.0f%%"
              % (100.0 * na / nn, 100.0 * nw / nn, 100.0 * nt / nn))
        if na == 0 and (nw or nt):
            print()
            print("    【结论·重要】判据① 完全失效, ②③ 有效。")
            print("      原因: 旋转**不改变**加速度矢量模长(重力恒 1g), 所以 ||a|-1g|")
            print("            对纯转动天然不敏感。手部抖动的主动成分正是转动。")
            print("      ⇒ 伪迹抑制必须带上 ② 或 ③; 只用 ① 会大面积漏检运动段。")
        elif cnt["any"] == 0:
            print("    本次全程静止, 各判据均未触发 —— 阈值标定需用 shake 模式验证。")
        else:
            print("    判据① 也有触发, 说明本次含平移分量; 但手部场景仍必须保留 ②③。")
    print()
    print("    推荐判据(取或):")
    print("      is_motion = (dev > %.2f g) or (|ω| > %.0f °/s) or (tilt > %.0f°)"
          % (thr, imu.gyro_threshold_dps, imu.tilt_threshold_deg))


# ============================================================ 主流程
def main():
    print("\n" + LINE)
    print("   ESP32 + MPU6050 硬件测试")
    print("   项目: 基于实时心率变异性驱动的主动式情感陪伴系统")
    print(LINE)

    results = {}

    i2c = t_scan()
    if i2c is None:
        print("\n" + LINE + "\n  结论: ✘ I2C 未找到 MPU6050, 后续测试中止\n" + LINE)
        return

    imu = MPU6050(i2c, addr=config.MPU6050_ADDR, int_pin=config.MPU6050_INT_PIN)
    imu.motion_threshold_g = config.MPU6050_MOTION_G
    imu.gyro_threshold_dps = config.MPU6050_GYRO_DPS
    imu.tilt_threshold_deg = config.MPU6050_TILT_DEG

    if not t_ident(imu):
        print("\n  结论: ✘ 寄存器读写校验未通过, 中止")
        return
    results["ident"] = True

    print("\n    初始化: accel=±2g, gyro=±250°/s, DLPF=3(44Hz), 目标 100 Hz ...")
    imu.begin(accel_range=2, gyro_range=250, dlpf=3, rate_hz=100)
    print("    ✔ 实际输出率 = %.1f Hz" % imu.rate_hz)
    time.sleep_ms(100)

    if not t_raw(imu):
        print("\n  结论: ✘ 数据读取异常, 中止")
        return

    mean_g, noise, g_noise = t_static(imu)
    imu.calibrate_still(1.0)      # 建立运动判定基线

    peak_dev = peak_omega = peak_tilt = None
    cnt = None
    if MONITOR_SEC:
        peak_dev, peak_omega, peak_tilt, cnt = t_monitor(imu)
    else:
        imu.calibrate_still(1.0)

    edges = t_int(imu)
    t_threshold(imu, noise, peak_dev, peak_omega, cnt)

    print("\n" + LINE)
    print("   测试结论")
    print(LINE)
    print("   I2C 通信       ✔ 0x%02X 在线, 6/6 寄存器写-回读一致" % config.MPU6050_ADDR)
    print("   芯片型号       %s" % imu.chip)
    print("   重力模长       %.4f g  %s" % (mean_g, "✔" if abs(mean_g - 1) <= 0.05 else "✘"))
    print("   加速度噪声     %.4f g  %s" % (noise, "✔" if noise <= 0.02 else "⚠"))
    print("   陀螺噪声       %.3f °/s" % g_noise)
    print("   运动阈值       %.2f g  %s" % (imu.motion_threshold_g,
          "✔ 噪声余量充足" if noise * 5 < imu.motion_threshold_g
          else "⚠ 建议提高到 %.2f" % max(0.25, round(noise * 8, 3))))
    if peak_dev is not None:
        print("   运动响应峰值   dev %.3f g | |ω| %.1f °/s | 倾角 %.1f°"
              % (peak_dev, peak_omega, peak_tilt))
    print("   INT(GPIO%-2d)    %s" % (config.MPU6050_INT_PIN,
          "✔ 通路正常(%d 边沿)" % edges if edges else "✘ 未通(不影响主功能)"))
    print()
    static_ok = abs(mean_g - 1) <= 0.05 and noise <= 0.02
    if static_ok:
        print("   ➜ 加速度计可用。可接入主固件做运动伪迹抑制。")
    else:
        print("   ➜ 静态指标未达标 —— shake 模式下若刚晃动过, 此段读数无效;")
        print("      请在**完全静止**时重跑  mpu_test_quick.py  确认。")
    if cnt is None:
        print("   ➜ 运动响应未验证 —— 跑  mpu_test_shake.py  确认阈值合理。")
    elif cnt["any"] == 0:
        print("   ➜ 本次全程静止, 未取得运动数据; 跑 mpu_test_shake.py 验证运动响应。")
    elif cnt["a"] == 0 and (cnt["w"] or cnt["t"]):
        print("   ➜ 【重要】伪迹抑制必须用角速度/倾角判据, 不能只用加速度模长。")
    else:
        print("   ➜ 运动响应已验证, 保留三判据取或。")
    print(LINE)


main()
