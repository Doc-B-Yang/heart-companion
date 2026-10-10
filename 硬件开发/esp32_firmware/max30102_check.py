# -*- coding: utf-8 -*-
"""
max30102_check.py — MAX30102 接线 / 通信 分层诊断

用法(PC 端):
    mpremote connect COM6 run max30102_check.py

为什么需要这个脚本:
    MAX30102 复位后 LED 是**关闭**的 —— LED1_PA(0x0C) / LED2_PA(0x0D) /
    MODE_CONFIG(0x09) 三个寄存器的复位值都是 0x00。必须由程序依次写入
    LED 电流 → 点亮模式, 灯才会亮。
    所以"灯不亮"本身**不能**证明接线错; 它也可能是"程序根本没跑到那一步"。
    本脚本把故障按 供电 → 总线电平 → 器件应答 → 强制点灯 → 独立通信证据
    逐层分离, 每层给出明确结论。

分层依据(为什么是这个顺序):
    L1 引脚电平   —— 排除"总线被短路/被拉死", 这一层错了后面全无意义
    L2 总线扫描   —— 分清是"整条总线挂"还是"只有 MAX30102 挂"(极关键)
    L3 型号识别   —— 确认 0x57 上坐的就是 MAX30102 而不是别的器件
    L4 强制点灯   —— 绕开业务逻辑, 直接写寄存器点灯, 肉眼确认 LED 驱动通路
    L5 温度读取   —— 不依赖 LED 的第三方通信证据; L4 失败而 L5 成功时,
                     说明是 LED 驱动/供电问题而非总线问题
"""

import time
from machine import I2C, Pin

try:
    import config
    SDA = config.I2C_SDA
    SCL = config.I2C_SCL
    FREQ = config.I2C_FREQ
    ADDR = config.MAX30102_ADDR
except ImportError:
    SDA, SCL, FREQ, ADDR = 21, 22, 400000, 0x57
    print("[!!] 未找到 config.py, 改用默认引脚 SDA=21 SCL=22")


# MAX30102 寄存器
REG_FIFO_WR_PTR = 0x04
REG_OVF_COUNTER = 0x05
REG_FIFO_RD_PTR = 0x06
REG_FIFO_CONFIG = 0x08
REG_MODE_CONFIG = 0x09
REG_SPO2_CONFIG = 0x0A
REG_LED1_PA     = 0x0C
REG_LED2_PA     = 0x0D
REG_TEMP_INT    = 0x1F
REG_TEMP_FRAC   = 0x20
REG_TEMP_EN     = 0x21
REG_REV_ID      = 0xFE
REG_PART_ID     = 0xFF
EXPECTED_PART_ID = 0x15

MODE_RESET = 0x40
MODE_HR    = 0x02
MODE_SPO2  = 0x03

# 已知的同总线器件(用于判断"是整条总线挂了还是单个器件挂了")
KNOWN = {
    0x57: "MAX30102 (PPG)",
    0x48: "ADS1115  (GSR, ADDR->GND)",
    0x68: "MPU6050  (IMU, AD0->GND)",
    0x49: "ADS1115  (ADDR 接错! 应为 GND=0x48)",
    0x69: "MPU6050  (AD0 接错! 应为 GND=0x68)",
}

findings = []


def say(txt=""):
    print(txt)


def rule(txt=""):
    print("-" * 58)
    if txt:
        print(txt)


def note(tag, txt):
    """tag: OK / !! / -- / >>"""
    print("  [%s] %s" % (tag, txt))
    findings.append((tag, txt))


# ============================================================ L1
rule("L1  总线引脚电平粗测 (SDA=GPIO%d / SCL=GPIO%d)" % (SDA, SCL))
say("  原理: 总线空闲时两条线都应被外部上拉电阻拉高。")
say("  方法: **只用输入模式**测, 绝不以输出/开漏方式驱动总线 ——")
say("        驱动模式下读到的是自己拉的电平, 而不是总线的真实状态。")
say("        判据: 外部约 3.3k 上拉 vs ESP32 内部约 45k 上/下拉。")
say("          内部上拉读 1 -> 线正常;  内部上拉读 0 -> 被硬拉低")
say("          内部下拉读 1 -> 外部上拉在位;  读 0 -> 该线没有外部上拉")
say("")

pin_level = {}
for name, no in (("SDA", SDA), ("SCL", SCL)):
    try:
        pu = Pin(no, Pin.IN, Pin.PULL_UP)
        time.sleep_ms(10)
        up = pu.value()
        pd = Pin(no, Pin.IN, Pin.PULL_DOWN)
        time.sleep_ms(10)
        dn = pd.value()
        Pin(no, Pin.IN)
        pin_level[name] = (up, dn)
        if up == 0 and dn == 0:
            note("!!", "%s(GPIO%d) 内部上拉=%d 内部下拉=%d  -> 被**硬拉低**"
                       % (name, no, up, dn))
        elif up == 1 and dn == 1:
            note("OK", "%s(GPIO%d) 内部上拉=%d 内部下拉=%d  -> 正常, 外部上拉在位"
                       % (name, no, up, dn))
        elif up == 1 and dn == 0:
            note("--", "%s(GPIO%d) 内部上拉=%d 内部下拉=%d  -> 无外部上拉(可能未接到模块)"
                       % (name, no, up, dn))
        else:
            note("--", "%s(GPIO%d) 内部上拉=%d 内部下拉=%d  -> 读数异常"
                       % (name, no, up, dn))
    except Exception as e:
        note("--", "%s(GPIO%d) 电平读取失败: %s" % (name, no, e))

if pin_level.get("SDA") == (0, 0):
    note("!!", "SDA 被硬拉低。定量: 与 3.3k 上拉并联仍读出逻辑 0, "
               "要求下拉阻抗 < 约 1.1kΩ —— 这已是**一根导线接地**的量级,")
    note(">>", "   不是芯片漏电。最常见: 杜邦线插进了 GND 那一列 / "
               "与 GND 插在同一排孔 / 接到了某个模块的 GND 脚。")
elif pin_level.get("SCL") == (0, 0):
    note("!!", "SCL 被硬拉低, 同上")

# 释放引脚, 准备交给 I2C
try:
    Pin(SDA, Pin.IN)
    Pin(SCL, Pin.IN)
except Exception:
    pass

# ============================================================ L2
rule("L2  I2C 总线扫描  (freq=%d Hz)" % FREQ)
found = []
scan_err = None
try:
    i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=FREQ)
    found = list(i2c.scan())
except Exception as e:
    scan_err = e
    try:
        # 400k 不稳时降速重试 —— 上拉偏弱/线偏长的典型症状
        i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=100000)
        found = list(i2c.scan())
        note("--", "400 kHz 扫描失败, 降到 100 kHz 后成功 -> 总线容性负载偏大")
    except Exception:
        pass

if scan_err and not found:
    note("!!", "扫描异常: %s" % scan_err)

say("")
say("  扫描结果: %s" % (", ".join("0x%02X" % a for a in found) if found else "(没有任何器件应答)"))
say("")
for a in found:
    note("OK", "0x%02X 在线%s" % (a, "  <- " + KNOWN[a] if a in KNOWN else ""))
for a, name in ((0x57, "MAX30102"),):
    if a not in found:
        note("!!", "0x%02X 缺失  <- %s 不通" % (a, name))

say("")
say("  >> 判读(这一步的信息量最大, 请看器件组合而不是单看 MAX30102):")
have = set(found)
if ADDR in have:
    note("OK", "0x57 在线上 -> MAX30102 的 SDA/SCL/VIN/GND 四根线都通了, "
               "问题不在接线, 在后面的初始化/程序")
elif not have:
    note("!!", "一条总线全空 -> 不是单个模块的锅。优先查:")
    note(">>", "   (a) 面包板上下两条电源轨是否连通(默认不通!) -> 模块可能根本没电")
    note(">>", "   (b) 3V3 与 GND 是否真的引到了面包板")
    note(">>", "   (c) SDA/SCL 有没有接反")
elif 0x48 in have or 0x68 in have:
    note("!!", "其它器件在线、唯独 0x57 不在 -> **铁证: 故障就在 MAX30102 这四根线上**")
    note(">>", "   其它模块把上拉和供电都证明了, 所以不是总线/供电/共地的问题。")
    note(">>", "   只剩三种可能: ①VIN 没接 ②SDA 或 SCL 漏接/接错孔 "
               "③模块本体损坏")
    note(">>", "   最快验证: 把 MAX30102 拔下来再扫一次。若总线依旧正常, "
               "故障 100% 在模块或它的线上。")

# ============================================================ L3
rule("L3  型号识别 (PART_ID @ 0x%02X reg 0xFF, 期望 0x15)" % ADDR)
part = rev = None
if i2c is not None and ADDR in have:
    try:
        part = i2c.readfrom_mem(ADDR, REG_PART_ID, 1)[0]
        rev = i2c.readfrom_mem(ADDR, REG_REV_ID, 1)[0]
        say("  PART_ID = 0x%02X   REV_ID = 0x%02X" % (part, rev))
        if part == EXPECTED_PART_ID:
            note("OK", "型号正确, 芯片活着")
        else:
            note("!!", "PART_ID 不是 0x15 -> 0x57 上坐的可能不是 MAX30102 "
                       "(或芯片未正常供电)")
    except Exception as e:
        note("!!", "读寄存器失败: %s -> 地址应答了但数据读不回来, "
                   "常见于上拉偏弱 / 线太长 / 接触不良" % e)
else:
    note("--", "跳过(0x57 不在线上)")

# ============================================================ L4
rule("L4  强制点灯 (绕开业务逻辑, 直接写寄存器)")
say("  红光亮起需要依次写: LED1_PA(0x0C) -> LED2_PA(0x0D) -> MODE_CONFIG(0x09)")
say("  复位后这三个都是 0x00 -> 所以**没初始化时灯本来就是灭的**。")
say("")

led_ok = False
if part == EXPECTED_PART_ID:
    try:
        def wr(reg, val):
            i2c.writeto_mem(ADDR, reg, bytes([val & 0xFF]))

        def rd(reg):
            return i2c.readfrom_mem(ADDR, reg, 1)[0]

        # 软复位
        wr(REG_MODE_CONFIG, MODE_RESET)
        time.sleep_ms(100)
        t0 = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), t0) < 500:
            if (rd(REG_MODE_CONFIG) & MODE_RESET) == 0:
                break
            time.sleep_ms(20)
        note("OK", "软复位完成, MODE_CONFIG=0x%02X" % rd(REG_MODE_CONFIG))

        # FIFO 指针归零
        wr(REG_FIFO_WR_PTR, 0x00)
        wr(REG_OVF_COUNTER, 0x00)
        wr(REG_FIFO_RD_PTR, 0x00)

        # 100 Hz / 411us / ADC 满量程 16384nA
        wr(REG_FIFO_CONFIG, 0x10 | 0x0F)
        wr(REG_SPO2_CONFIG, 0x60 | (1 << 2) | 0x03)

        # ---- 点灯: 12.6mA 档 ----
        wr(REG_LED1_PA, 0x3F)
        wr(REG_LED2_PA, 0x3F)
        wr(REG_MODE_CONFIG, MODE_SPO2)
        say("")
        say("  >>> 现在 LED 电流 = 0x3F (约 12.6 mA / 每颗) , 持续 5 秒 <<<")
        say("  >>> 请把传感器窗口对着自己的眼睛(别贴太近), 看有没有暗红色微光 <<<")
        say("  >>> 更可靠的做法: 用**手机前置摄像头**对着窗口看, "
            "红外光肉眼几乎看不见但摄像头能看到紫白光点 <<<")
        time.sleep(5)

        # ---- 提到较亮档 ----
        wr(REG_LED1_PA, 0xC8)
        wr(REG_LED2_PA, 0xC8)
        say("")
        say("  >>> 现在提到 0xC8 (约 40 mA) , 再持续 8 秒 —— 应该明显更亮 <<<")
        time.sleep(8)

        # 用 FIFO 是否有数据作为"LED 真的在发光"的客观证据
        wr(REG_FIFO_WR_PTR, 0x00)
        wr(REG_FIFO_RD_PTR, 0x00)
        time.sleep_ms(300)
        wr_ptr = rd(REG_FIFO_WR_PTR)
        say("")
        say("  客观佐证: 300 ms 内 FIFO 写入指针 = %d (100 Hz 下期望约 30)" % wr_ptr)
        if wr_ptr >= 10:
            note("OK", "FIFO 在持续进数据 -> 传感器确实在工作")
            led_ok = True
        elif wr_ptr > 0:
            note("--", "FIFO 有少量数据(%d) -> 采样率异常或有丢包" % wr_ptr)
        else:
            note("!!", "FIFO 无数据 -> 即使灯亮, 采集链路也不正常")

        # 关灯(收尾, 避免模块长时间发热)
        wr(REG_MODE_CONFIG, 0x80)
        note("OK", "已关闭 LED 与采集 (MODE_CONFIG=0x80 低功耗)")
    except Exception as e:
        note("!!", "点灯过程出错: %s" % e)
else:
    note("--", "跳过(型号未识别, 不做写寄存器操作以免误伤别的器件)")

# ============================================================ L5
rule("L5  独立通信证据 — 读芯片温度 (不依赖 LED)")
if part == EXPECTED_PART_ID:
    try:
        def wr2(reg, val):
            i2c.writeto_mem(ADDR, reg, bytes([val & 0xFF]))

        # 软复位后回到正常模式再读温度
        wr2(REG_MODE_CONFIG, MODE_RESET)
        time.sleep_ms(100)
        wr2(REG_MODE_CONFIG, MODE_HR)
        time.sleep_ms(50)
        wr2(REG_TEMP_EN, 0x01)
        time.sleep_ms(50)
        ti = i2c.readfrom_mem(ADDR, REG_TEMP_INT, 1)[0]
        tf = i2c.readfrom_mem(ADDR, REG_TEMP_FRAC, 1)[0]
        if ti > 127:
            ti -= 256
        t = ti + tf * 0.0625
        say("  芯片温度 = %.2f C" % t)
        if 15.0 <= t <= 60.0:
            note("OK", "温度在合理范围 -> I2C **读写双向**都正常, 通信链路无问题")
            note(">>", "如果此时灯还是不亮, 结论是: 通信没问题, 是 LED 本身/供电"
                       "(VIN 电压不足) 的问题")
        else:
            note("--", "温度读数 %.2f C 不在 15~60 范围 -> 数据可疑" % t)
    except Exception as e:
        note("!!", "读温度失败: %s" % e)
else:
    note("--", "跳过")

# ============================================================ 结论
rule()
say("诊断结论汇总")
rule()
bad = [t for t in findings if t[0] == "!!"]
warn = [t for t in findings if t[0] == "--"]
if not bad:
    say("  未发现硬故障。")
    if ADDR in have and part == EXPECTED_PART_ID:
        say("  通信与芯片都正常 -> 若之前'灯不亮', 原因是")
        say("    (a) 程序没跑到 begin(), 或 (b) 环境光太亮看不清 8% 占空比的红光。")
        say("  验证: 直接跑 main.py, 看首行 I2C 扫描有没有 0x57、")
        say("        以及 'MAX30102: OK' 之后 auto_gain 是否打印。")
else:
    say("  发现 %d 条异常:" % len(bad))
    for _, t in bad:
        say("    - %s" % t)
if warn:
    say("")
    say("  %d 条提示:" % len(warn))
    for _, t in warn:
        say("    - %s" % t)

say("")
rule()
say("手工对照清单(逐条看实物, 不要凭记忆)")
rule()
say("  1. 看**丝印字母**, 不要看引脚位置。市售模块至少三种引脚顺序:")
say("       A: VIN  GND  SCL  SDA  INT")
say("       B: VCC  GND  SDA  SCL  INT   <- SDA/SCL 与 A 相反!")
say("       C: VCC  SCL  SDA  GND  INT")
say("     按位置照搬另一种模块的顺序, 就会把 SDA/SCL 接反。")
say("  2. 面包板**上下两条电源轨默认不连通**。若 3V3 接在上轨、")
say("     模块 GND 接在下轨, 模块根本没电。必须用跳线把两侧 GND 轨连起来。")
say("  3. 面包板**中间断槽**两侧不导通; 劣质板还有单孔接触不良。")
say("     把可疑的线换一个孔试。")
say("  4. 电源: VIN 必须接 **3V3** 而不是 5V(会通过上拉把 5V 灌进 GPIO)。")
say("  5. 隔空看模块背面有没有**虚焊/焊桥**, 传感器窗口有没有被东西挡住。")
say("")
say("  最有效的一招 —— 隔离法:")
say("    把 MPU6050 和 ADS1115 全部拔掉, 只留 MAX30102 一根一根线接,")
say("    每接一根就 `i2c.scan()` 一次。哪一根接上去 0x57 才出现, 答案就出来了。")
