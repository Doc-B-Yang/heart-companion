# -*- coding: utf-8 -*-
"""
wire_check.py — 四模态接线全面核验(ESP32 侧, 不依赖任何模块"看起来亮不亮")

用法(PC 端):
    mpremote connect COM6 run wire_check.py

为什么要有这个脚本:
    "灯不亮" 不能判定接线对错 —— MAX30102 的 LED1_PA(0x0C)/LED2_PA(0x0D)/
    MODE_CONFIG(0x09) 复位值全是 0x00, 不上电写这三处灯本来就不亮。
    唯一可信的判据是 **I2C 通不通**, 以及**每颗器件的身份寄存器读不读得到**。

设计原则(方法论, 别改):
    1. 测引脚电平**只用输入模式**(Pin.IN / Pin.PULL_UP / Pin.PULL_DOWN),
       绝不用 OPEN_DRAIN 不传 value 的写法 —— 开漏下 value 默认可能是 0,
       等于自己把线拉低, 会得出假结论。
    2. 每条"异常"结论都配一个**对照组**: 拿空闲 GPIO 做同样测量,
       证明方法本身在本芯片上有效, 否则无权说"这条线坏了"。
    3. 区分"外部器件拉低"与"引脚本身损坏"必须靠**物理断开**,
       软件只能给出判据清单, 不能替用户下结论 —— 见 L2 结尾的说明。

层级:
    L0  对照组(空闲 GPIO) —— 验证测量方法有效
    L1  总线空闲电平 + 上/下拉负载特性
    L2  GPIO21 输出驱动测试(判断引脚驱动器是否还活着)
    L3  多速率总线扫描
    L4  器件身份核验(仅当 L3 有应答时执行)
"""

import time
from machine import Pin, I2C

SDA = 21
SCL = 22

# 空闲引脚作对照组: 这些脚在本项目里没有接任何东西
CONTROL_PINS = (25, 26, 27, 32, 33)

# 期望在线器件: ADS1115 / MAX30102 / MPU6050
EXPECT = {0x48: "ADS1115 (GSR 的 ADC)", 0x57: "MAX30102 (PPG)", 0x68: "MPU6050 (IMU)"}


def rule(t):
    print("\n" + "=" * 62)
    print(" " + t)
    print("=" * 62)


def say(t=""):
    print(t)


def note(tag, t):
    print("  [%s] %s" % (tag, t))


# ============================================================ L0
rule("L0  对照组 —— 先用空闲 GPIO 证明'本方法在本芯片上有效'")
say("  做法: 对 GPIO%s 依次测 'PULL_UP / PULL_DOWN / 悬空' 三种状态。" % (CONTROL_PINS,))
say("  预期: 这三个读数应为 1 / 0 / (不定)。若连空闲脚的 PULL_UP 都读不到 1,")
say("        那前面所有关于 SDA=0 的结论都不可信, 不能往下走。")
say("")

ctrl_ok = 0
for n in CONTROL_PINS:
    try:
        up = Pin(n, Pin.IN, Pin.PULL_UP).value()
        time.sleep_ms(5)
        dn = Pin(n, Pin.IN, Pin.PULL_DOWN).value()
        time.sleep_ms(5)
        fl = Pin(n, Pin.IN).value()
        Pin(n, Pin.IN)
        flag = "OK" if (up == 1 and dn == 0) else "??"
        if flag == "OK":
            ctrl_ok += 1
        note(flag, "GPIO%-2d  上拉=%d  下拉=%d  悬空=%d" % (n, up, dn, fl))
    except Exception as e:
        note("--", "GPIO%-2d 测量失败: %s" % (n, e))

if ctrl_ok >= 3:
    note("OK", "对照组通过(%d/%d) -> 测量方法有效, 下面的结论有资格成立"
         % (ctrl_ok, len(CONTROL_PINS)))
else:
    note("!!", "对照组只有 %d/%d 通过 -> **方法本身可疑**, 请先别信后面的任何结论"
         % (ctrl_ok, len(CONTROL_PINS)))


# ============================================================ L1
rule("L1  总线空闲电平 + 负载特性 (SDA=GPIO%d / SCL=GPIO%d)" % (SDA, SCL))
say("  原理: 总线空闲时两条线都应被外部上拉电阻拉到高。")
say("        · 读 1            -> 上拉在位, 正常")
say("        · 内部上拉也拉不动 -> 被『远强于上拉』的通路拉低, 是硬接地量级")
say("")

lv = {}
for name, no in (("SDA", SDA), ("SCL", SCL)):
    try:
        up = Pin(no, Pin.IN, Pin.PULL_UP).value()
        time.sleep_ms(5)
        dn = Pin(no, Pin.IN, Pin.PULL_DOWN).value()
        time.sleep_ms(5)
        fl = Pin(no, Pin.IN).value()
        time.sleep_ms(5)
        Pin(no, Pin.IN)
        lv[name] = up
        if up == 1:
            note("OK", "%s(GPIO%d)  上拉=%d  下拉=%d  悬空=%d  -> 正常"
                 % (name, no, up, dn, fl))
        else:
            note("!!", "%s(GPIO%d)  上拉=%d  下拉=%d  悬空=%d  -> 被强拉低"
                 % (name, no, up, dn, fl))
    except Exception as e:
        lv[name] = None
        note("--", "%s(GPIO%d) 测量失败: %s" % (name, no, e))

say("")
if lv.get("SDA") == 0 and lv.get("SCL") == 0:
    note("!!", "两条线都是 0 -> 优先怀疑**没共地**, 或两个引脚接反后同接到 GND")
elif lv.get("SDA") == 0:
    note("!!", "只有 SDA 被拉低。ESP32 内部上拉约 45kΩ; 若模块侧上拉约 1.6~3.3kΩ,")
    say("       要把它压到逻辑 0, 下拉阻抗需在**几百欧以内** —— 这不是芯片漏电,")
    say("       是**一根导线直接接地**的量级。")
elif lv.get("SCL") == 0:
    note("!!", "只有 SCL 被拉低, 同上判断")
else:
    note("OK", "两条线都被上拉, 物理层正常 -> 直接看 L3/L4")


# ============================================================ L2
rule("L2  GPIO21 负载特性 —— 区分'有东西在拉'与'引脚悬空/断线'")
say("  ⚠ 先排掉一个会骗人的做法(本脚本第一版就踩了):")
say("     把引脚设为 OUTPUT 再 p.value(), 读到的**可能是输出锁存器**")
say("     而不是焊盘真实电平 —— 那样驱 1 必然读回 1, 测不出任何东西。")
say("     所以下面改成 **'驱到某电平 -> 切成输入 -> 观察它自己回到哪里'**。")
say("")
say("  为什么需要这一步: L1 只证明了'内部上拉拉不动它', 但这有两种可能 ——")
say("     (甲) 有通路把它**拉向 0**(短路/接地/器件漏电)")
say("     (乙) 引脚**根本没接上东西**, 而内部上拉本身是坏的")
say("  这两者的区分就靠下面的 T2: 先把它**主动驱到 1**, 再撒手。")
say("     · 撒手后**塌回 0** -> 确实存在一条把它拉向 0 的通路  -> 甲")
say("     · 撒手后**保持 1** -> 之前读到 0 是别的原因(如内部上拉失效) -> 乙")
say("")
try:
    # T1: 驱 0 后撒手(仅内部下拉), 应稳定为 0
    p = Pin(SDA, Pin.OUT)
    p.value(0)
    time.sleep_ms(30)
    Pin(SDA, Pin.IN, Pin.PULL_DOWN)
    time.sleep_ms(30)
    t1 = Pin(SDA, Pin.IN, Pin.PULL_DOWN).value()

    # T2: 驱 1 后撒手(仅内部下拉), 看它塌不塌
    p = Pin(SDA, Pin.OUT)
    p.value(1)
    time.sleep_ms(30)
    Pin(SDA, Pin.IN, Pin.PULL_DOWN)
    time.sleep_ms(30)
    t2 = Pin(SDA, Pin.IN, Pin.PULL_DOWN).value()

    # T3: 驱 1 后撒手(完全悬空, 无内部上下拉), 看它塌不塌
    p = Pin(SDA, Pin.OUT)
    p.value(1)
    time.sleep_ms(30)
    Pin(SDA, Pin.IN)
    time.sleep_ms(30)
    t3 = Pin(SDA, Pin.IN).value()

    Pin(SDA, Pin.IN)
    note("--", "T1 驱0后撒手(内部下拉) = %d   期望 0" % t1)
    note("--", "T2 驱1后撒手(内部下拉) = %d   期望: 有短路->0 / 无负载->1" % t2)
    note("--", "T3 驱1后撒手(完全悬空) = %d   期望: 有短路->0 / 无负载->1" % t3)
    say("")
    if t2 == 0 or t3 == 0:
        note("!!", "驱 1 撒手后立刻塌回 0 -> **确认存在一条主动拉向 0 的通路**")
        say("       (排除'引脚悬空'与'只是内部上拉坏了'这两种解释, 判为【甲】)")
    else:
        note("??", "驱 1 撒手后保持 1 -> 与 L1 的'上拉读 0'矛盾, 需重测 L1")
except Exception as e:
    note("--", "负载特性测试失败: %s" % e)

say("")
note("仍然分不开的事", "『外部一根导线接地』与『芯片内部 ESD 打穿后接地』")
say("       在电压/阻抗这两个维度上**完全同形**, 软件无解。")
say("       但概率上: ESD 打穿通常表现为**开断或千欧级漏电**,")
say("       要做到'内部上拉都拉不动'需要**几百欧以内**, 那更像一根导线。")
say("")
note("决定性 10 秒动作", "把 SDA 那根杜邦线**从 GPIO21 上拔下来, 让它悬空**,")
say("       然后重跑本脚本:")
say("         · L1 的 SDA 恢复为 1 -> GPIO21 完好, 问题在那根线/面包板/模块侧")
say("         · L1 的 SDA 仍为 0    -> GPIO21 引脚已损坏 -> 换脚:")
say("             把 wire_check.py 顶部 SDA 改成 23(或 18/19/5/16),")
say("             main.py / bridge 侧同步改, 杜邦线挪到新脚即可")
note("总线上限提醒", "GPIO 对地短路时输出电流可达数十 mA, 久接会发烫;")
say("       排查期间别让它长时间驱 1 顶着短路")


# ============================================================ L3
rule("L3  多速率总线扫描")
say("  为什么要多速率: 上拉太弱 / 线太长时, 400kHz 失败而 50kHz 能通;")
say("                  三档全无应答才说明是物理层问题, 而不是时序问题。")
say("")
found = []
for f in (400000, 100000, 50000, 10000):
    try:
        i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=f)
        a = sorted(i2c.scan())
        txt = ",".join("0x%02X" % x for x in a) if a else "(无应答)"
        print("  %7d Hz -> %s" % (f, txt))
        if a and not found:
            found = a
    except Exception as e:
        print("  %7d Hz -> 异常: %s" % (f, e))

say("")
if not found:
    note("!!", "三档全无应答 -> I2C 物理层不通, **后面所有模块级测试都无法进行**")
    say("       这不是『算法/参数』问题, 先解决 L1 的低电平")
else:
    missing = [x for x in EXPECT if x not in found]
    for x in sorted(found):
        note("OK", "0x%02X = %s" % (x, EXPECT.get(x, "未知器件")))
    for x in missing:
        note("!!", "0x%02X = %s  **未应答**" % (x, EXPECT[x]))


# ============================================================ L4
def _rd(i2c, addr, reg, n=1):
    return i2c.readfrom_mem(addr, reg, n)


if found:
    rule("L4  器件身份核验(读身份寄存器, 比『扫到地址』更硬)")

    # --- MAX30102 ---
    if 0x57 in found:
        try:
            i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=400000)
            part = _rd(i2c, 0x57, 0xFF)[0]
            rev = _rd(i2c, 0x57, 0xFE)[0]
            if part == 0x15:
                note("OK", "MAX30102 PART_ID=0x%02X (期望 0x15)  REV=0x%02X" % (part, rev))
            else:
                note("!!", "0x57 有应答但 PART_ID=0x%02X != 0x15 -> 可能不是 MAX30102" % part)
        except Exception as e:
            note("!!", "MAX30102 身份读取失败: %s" % e)
    else:
        note("--", "MAX30102 不在总线上, 跳过")

    # --- ADS1115: 写-回读校验(比身份寄存器可靠) ---
    if 0x48 in found:
        try:
            i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=400000)
            ok = 0
            for lo, hi in ((0x00, 0x83), (0x05, 0x83)):
                val = (hi << 8) | lo
                i2c.writeto_mem(0x48, 0x01, bytes([hi, lo]))
                time.sleep_ms(10)
                rb = _rd(i2c, 0x48, 0x01, 2)
                got = (rb[0] << 8) | rb[1]
                if got == val:
                    ok += 1
            if ok == 2:
                note("OK", "ADS1115 CONFIG 写-回读 2/2 一致 -> 器件可用")
            else:
                note("!!", "ADS1115 写-回读只有 %d/2 一致 -> 可疑" % ok)
        except Exception as e:
            note("!!", "ADS1115 校验失败: %s" % e)
    else:
        note("--", "ADS1115 不在总线上, 跳过")

    # --- MPU6050: WHO_AM_I 非标准(实测返回 0x75 = 寄存器地址), 改用写-回读 ---
    if 0x68 in found:
        try:
            i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=400000)
            who = _rd(i2c, 0x68, 0x75)[0]
            note("--", "MPU6050 WHO_AM_I=0x%02X  (本批模块实测非标准, 不作为判据)" % who)
            probes = ((0x19, 0x04), (0x1A, 0x03), (0x1C, 0x10),
                      (0x1B, 0x18), (0x6B, 0x01), (0x6C, 0x00))
            ok = 0
            for reg, val in probes:
                i2c.writeto_mem(0x68, reg, bytes([val]))
                time.sleep_ms(5)
                rb = _rd(i2c, 0x68, reg)[0]
                if rb == val:
                    ok += 1
            if ok == len(probes):
                note("OK", "MPU6050 写-回读 %d/%d 一致 -> 器件可用" % (ok, len(probes)))
            else:
                note("!!", "MPU6050 写-回读 %d/%d -> 可疑" % (ok, len(probes)))
        except Exception as e:
            note("!!", "MPU6050 校验失败: %s" % e)
    else:
        note("--", "MPU6050 不在总线上, 跳过")


# ============================================================ 汇总
rule("汇总")
all_ok = (lv.get("SDA") == 1 and lv.get("SCL") == 1 and len(found) >= 3)
if all_ok:
    note("OK", "物理层 + 三器件全部就位 -> 可以进入功能测试")
    say("       下一步: mpremote connect COM6 run max30102_check.py  (强制点灯/温度)")
else:
    note("!!", "接线**未通过**, 功能测试无法进行。按下面顺序处理:")
    if lv.get("SDA") != 1:
        say("       1. 对照 ESP32 丝印, 确认 SDA 线插的是 GPIO21 而不是**紧邻的 GND 针**")
        say("          (多数 DevKit 上 GPIO21 的相邻针就是 GND, 最容易插错一格)")
        say("       2. 把 SDA 从 GPIO21 拔下**悬空**, 重跑本脚本 -> 若 L1 恢复为 1, 则")
        say("          GPIO21 完好, 问题在那根线/它的去向; 若仍为 0, GPIO21 已损坏")
        say("       3. 逐个模块拔掉, 每次重跑, 找出是哪颗模块把 SDA 拉死")
        say("       4. 万用表蜂鸣档: SDA 对 GND 应为不通(不响)")
    if lv.get("SCL") != 1:
        say("       5. SCL(GPIO22) 同样处理")
    if lv.get("SDA") == 1 and lv.get("SCL") == 1 and not found:
        say("       6. 电平正常但扫不到 -> 查是否**没共地**, 或所有模块都没接 3.3V")
say("")
print(" 监视器(边拔边看): mpremote connect COM6 run bus_watch.py")
print("=" * 62)
