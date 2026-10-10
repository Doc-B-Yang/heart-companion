# -*- coding: utf-8 -*-
"""
bus_recover.py — I2C 总线恢复(9 个时钟脉冲)

用法:
    mpremote connect COM6 run bus_recover.py

要解决的问题:
    "SDA 恒为低"有两种成因, 现象完全相同, 必须区分:
      (A) 某根线/某个引脚直接接地      -> 挂死, 时钟脉冲无用
      (B) 某个从机卡在 ACK 相位         -> 时钟脉冲可释放
    为什么容易混淆: I2C 从机在 ACK 时用**开漏**拉低 SDA, 阻抗只有
    几十欧, 与"导线接地"在电平上无法区分。必须靠"打时钟看是否恢复"来分辨。

原理:
    从机在 ACK 相位拉低 SDA 后, 会等待主机发出后续时钟。连打 9 个时钟
    让它走完当前字节(8 位 + 1 个 ACK), 再补一个 STOP 条件, 状态机就复位了。
    这是 I2C 规范推荐的 bus clear 流程。

安全性:
    全程只用开漏(OPEN_DRAIN) + 内部上拉(约 45kΩ)。即使线上真有短路,
    也只会流过约 3.3V/45k ≈ 73µA, 不会烧任何东西。
"""

import time
from machine import Pin, I2C

SDA = 21
SCL = 22
FREQ = 400000

CLOCKS = 9


def sda_level():
    """只用输入模式读 SDA(带内部上拉), 不驱动总线。"""
    p = Pin(SDA, Pin.IN, Pin.PULL_UP)
    time.sleep_ms(5)
    v = p.value()
    Pin(SDA, Pin.IN)
    return v


def scl_level():
    p = Pin(SCL, Pin.IN, Pin.PULL_UP)
    time.sleep_ms(5)
    v = p.value()
    Pin(SCL, Pin.IN)
    return v


def scan():
    try:
        i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=FREQ)
        return sorted(i2c.scan())
    except Exception:
        return []


print("=" * 60)
print(" I2C 总线恢复  SDA=GPIO%d  SCL=GPIO%d" % (SDA, SCL))
print("=" * 60)

before_sda = sda_level()
before_scl = scl_level()
print(" 恢复前: SDA=%d  SCL=%d   在线器件=%s"
      % (before_sda, before_scl,
         ",".join("0x%02X" % a for a in scan()) or "(无)"))

if before_sda == 1:
    print("\n SDA 本来就是高电平, 无需恢复。")
    if not scan():
        print(" 但总线仍无器件应答 -> 问题不在总线挂死, 而在接线/供电。")
else:
    print("\n SDA 为低, 开始发送 %d 个时钟脉冲..." % CLOCKS)
    # 释放 SDA(开漏 + 上拉, 逻辑上为高阻), SCL 也开漏
    psda = Pin(SDA, Pin.OPEN_DRAIN, Pin.PULL_UP, value=1)
    pscl = Pin(SCL, Pin.OPEN_DRAIN, Pin.PULL_UP, value=1)
    time.sleep_ms(2)

    for i in range(CLOCKS):
        pscl.value(0)
        time.sleep_us(5)
        pscl.value(1)
        time.sleep_us(5)
        cur = psda.value()
        print("   时钟 %d: SDA 读数 = %d%s"
              % (i + 1, cur, "   <- 从机已释放!" if cur == 1 else ""))

    # 补一个 STOP 条件: SDA 低 -> SCL 高 -> SDA 高
    psda.value(0)
    time.sleep_us(5)
    pscl.value(1)
    time.sleep_us(5)
    psda.value(1)
    time.sleep_us(5)

    # 释放引脚
    psda.init(Pin.IN)
    pscl.init(Pin.IN)
    time.sleep_ms(10)

    after_sda = sda_level()
    print()
    print(" 恢复后: SDA=%d  SCL=%d" % (after_sda, scl_level()))
    print(" 在线器件=%s" % (",".join("0x%02X" % a for a in scan()) or "(无)"))

    print()
    print("-" * 60)
    if after_sda == 1:
        print(" >> 结论: SDA 被时钟脉冲**释放**了")
        print("    ⇒ 成因是 (B): 某个从机卡在 ACK 相位, 属于总线挂死,")
        print("      不是接线短路。常见触发: 上电瞬间总线时序错乱、")
        print("      主机复位时从机正处于传输中。")
        print("    处理: 重新 i2c.scan() 确认器件回来; 若仍无应答, 给所有")
        print("          模块断电 3 秒再上电(冷启动)后重扫。")
    else:
        print(" >> 结论: 打了 %d 个时钟 SDA 仍是低" % CLOCKS)
        print("    ⇒ 成因是 (A): 硬接地。不是从机卡死, 时钟脉冲救不回来。")
        print("      可疑点(按概率排序):")
        print("       1) SDA 杜邦线插进了 GND 那一列 / 与 GND 同插一排孔")
        print("          (面包板同一横排 5 个孔是互通的, 极易犯)")
        print("       2) SDA 线接到了某个模块的 GND 引脚")
        print("       3) 面包板某孔内部短路")
        print("       4) ESP32 的 GPIO21 已损坏")
        print("      下一步: 跑 bus_watch.py, 边拔模块边看 SDA 何时恢复。")
print()
print("=" * 60)
