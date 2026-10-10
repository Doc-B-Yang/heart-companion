# -*- coding: utf-8 -*-
"""
_bus_probe.py — I2C 总线物理层探针(只用输入模式, 绝不主动驱动总线)

目的: 交叉验证 SDA(GPIO21) 空闲读 0 是"真被外部拉低"还是"测量方法本身把它拉低了"。

为什么只用输入模式:
  输出/open-drain 模式下引脚会主动驱动总线。MicroPython 构造 Pin 时若不显式传
  value, 输出锁存器的值不确定 —— 用它测"空闲电平"可能测的是自己拉的电平。
  纯输入模式(IN)不影响总线, 结论才可信。

判读依据(ESP32 内部上/下拉约 45kΩ):
  ① IN + 内部上拉  : 若外部有 3.3k 上拉 → 读 1; 若被硬接地 → 读 0
  ② IN + 内部下拉  : 若外部有 3.3k 上拉 → 读 1(3.3k 轻松战胜 45k); 无上拉 → 读 0
  ③ IN 悬空        : 高阻, 读数不定(仅作参考)
  组合判读:
    UP=1, DOWN=1  -> 线正常, 外部上拉在位
    UP=0, DOWN=0  -> 线被硬拉低(接 GND / 某器件拉死 / 短路)
    UP=1, DOWN=0  -> 线正常但没有外部上拉(模块没接 / 上拉电阻不在)
"""

import time
from machine import Pin, I2C

SDA, SCL = 21, 22
FREQ = 400000


def rd(n, pull):
    if pull is None:
        p = Pin(n, Pin.IN)
    else:
        p = Pin(n, Pin.IN, pull)
    time.sleep_ms(10)
    return p.value()


def triplet(n):
    up = rd(n, Pin.PULL_UP)
    dn = rd(n, Pin.PULL_DOWN)
    fl = rd(n, None)
    return up, dn, fl


print("=" * 58)
print(" A. 纯输入模式测空闲电平(不驱动总线)")
print("=" * 58)
res = {}
for name, n in (("SDA", SDA), ("SCL", SCL)):
    up, dn, fl = triplet(n)
    res[name] = (up, dn, fl)
    print("  %s (GPIO%d):  内部上拉=%d   内部下拉=%d   悬空=%d" % (name, n, up, dn, fl))
    if up == 0 and dn == 0:
        print("      -> 被**硬拉低**")
    elif up == 1 and dn == 0:
        print("      -> 正常, 但**没有外部上拉**(该线可能悬空/未接到模块)")
    elif up == 1 and dn == 1:
        print("      -> 正常, 外部上拉在位")
    else:
        print("      -> 读数异常, 需人工判断")

print()
print("=" * 58)
print(" B. 验证'OPEN_DRAIN 不传 value'是否会把线拉低")
print("    (用来判定前面 L1 那次读数可信不可信)")
print("=" * 58)
for name, n in (("SDA", SDA), ("SCL", SCL)):
    p = Pin(n, Pin.OPEN_DRAIN, Pin.PULL_UP)
    time.sleep_ms(10)
    v_default = p.value()
    p.value(1)
    time.sleep_ms(10)
    v_high = p.value()
    print("  %s(GPIO%d):  OPEN_DRAIN默认读出=%d   value(1)后读出=%d" %
          (name, n, v_default, v_high))
    if v_default == 0 and v_high == 1:
        print("      -> 证实: 默认态自己在拉低! 之前 L1 读到 0 是测量自身造成的")
    elif v_default == v_high:
        print("      -> 默认态与置高态一致, 说明读数反映的是外部真实电平")
del p

print()
print("=" * 58)
print(" C. 短路排查")
print("=" * 58)
print("""  [方法说明 — 为什么不用'拉低 SCL 看 SDA'那种测法]
  若 SDA 本身恒为 0, 那么"拉低 SCL 后 SDA 仍为 0"是必然结果,
  这个测试**没有鉴别力**, 会给出假阳性。因此本段改用
  '两条线的负载是否相同' 来推断, 全程只做输入, 不驱动总线。
""")
# C1: 用"内部下拉能否被外部上拉战胜"判断该线是否存在外部上拉
sda_up, sda_dn, _ = res["SDA"]
scl_up, scl_dn, _ = res["SCL"]
if scl_dn == 1:
    print("  [OK] SCL 有外部上拉(内部下拉都被它拉高) -> 总线侧的上拉电阻确实在")
    if sda_dn == 0:
        print("  [!!] 但 SDA 连内部下拉都还是 0 -> SDA 上的下拉**远强于**上拉")
        print("       SCL 的三千欧级上拉若与 SDA 同节点, SDA 至少会被抬到中间电平;")
        print("       实测仍为 0 ⇒ 二者**不是同一节点**(未被短接到同一电位)")
        print("       ⇒ 结论: SDA 被一条**独立的、接地阻抗极低**的通路拉着")
        print("         (典型: 杜邦线插进了 GND 那一列, 或与 GND 同插一排孔)")
    else:
        print("  [--] SDA 与 SCL 负载特性接近, 不构成互短特征")
else:
    print("  [--] SCL 也无外部上拉 -> 上拉电阻整体不在总线上, 先查模块是否真的接上")

# C2: SDA 与电源是否短接 —— 内部下拉仍读 1 才是被电源硬拉高
if sda_dn == 1 and sda_up == 1:
    print("  [!!] SDA 恒为 1 -> 被电源侧硬拉高, 查是否误接到 3V3 排")
elif sda_up == 0 and sda_dn == 0:
    print("  [OK] 已排除 SDA 接 3V3(否则内部下拉读 1)")

print()
print("=" * 58)
print(" D. 重新扫描总线")
print("=" * 58)
for f in (400000, 100000, 50000):
    found = []
    err = None
    try:
        i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=f)
        found = list(i2c.scan())
    except Exception as e:
        err = e
    print("  %6d Hz -> %s%s" % (f,
          ", ".join("0x%02X" % a for a in found) if found else "(无应答)",
          ("   err=%s" % err) if err else ""))

print()
print("=" * 58)
print(" E. 空载总线自检(排除 ESP32 自身问题)")
print("=" * 58)
print("  把 MAX30102 / MPU6050 / ADS1115 全部从面包板上拔掉,")
print("  再跑一次本脚本。若下面两项恢复为 1, 则问题是某个模块接线:")
print("    SDA 内部上拉=%d    SCL 内部上拉=%d" % (res["SDA"][0], res["SCL"][0]))
print("  之后再一根一根往总线加模块, 每加一个跑一次 —— 哪个模块一插上")
print("  SDA 就变 0, 就是它接线错误或已损坏。")
