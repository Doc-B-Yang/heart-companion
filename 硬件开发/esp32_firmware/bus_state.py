# -*- coding: utf-8 -*-
"""
bus_state.py -- 一句话总线状态检查 (SDA=GPIO21 / SCL=GPIO22)

用途: 每做完一个物理动作就重跑一次, 看 SDA 是否恢复。
只用 **纯输入模式 + 内部上拉/下拉** 读电平, 绝不输出,
以免测量动作本身把总线拉低 (这是本项目踩过的坑)。

    mpremote connect COM6 run bus_state.py

判读:
    上拉读回 = 0  -> 被外部强拉低 (硬接地量级)
    上拉读回 = 1  -> 正常
    下拉读回 = 1  -> 外部上拉在位 (通常是模块上的 3.3k / 4.7k 上拉电阻)
"""

from machine import Pin

SDA = 21
SCL = 22


def probe(no):
    """返回 (上拉读回, 下拉读回); 只读不写。"""
    up = Pin(no, Pin.IN, Pin.PULL_UP).value()
    dn = Pin(no, Pin.IN, Pin.PULL_DOWN).value()
    Pin(no, Pin.IN)          # 归还为高阻, 免得继续影响总线
    return up, dn


def main():
    sda_up, sda_dn = probe(SDA)
    scl_up, scl_dn = probe(SCL)

    if sda_up == 0:
        sda_txt = "被拉低  X"
    else:
        sda_txt = "正常    OK"
    if scl_up == 0:
        scl_txt = "被拉低  X"
    else:
        scl_txt = "正常    OK"

    print("")
    print("=" * 48)
    print("  总线状态   SDA=GPIO%d   SCL=GPIO%d" % (SDA, SCL))
    print("=" * 48)
    print("  SDA   上拉=%d  下拉=%d    %s" % (sda_up, sda_dn, sda_txt))
    print("  SCL   上拉=%d  下拉=%d    %s" % (scl_up, scl_dn, scl_txt))
    print("-" * 48)

    if sda_up == 0 and scl_up == 1:
        print("  结论: 只有 SDA 被强拉向 0")
        print("        -> 继续按指挥做下一步物理动作")
    elif sda_up == 1 and scl_up == 1:
        print("  结论: SDA 已恢复!")
        print("        -> 下一步跑 wire_check.py 做完整核验")
    elif sda_up == 0 and scl_up == 0:
        print("  结论: 两条都被拉低")
        print("        -> 先查是否共地 / 电源是否上电")
    else:
        print("  结论: SCL 异常, 先查 SCL 那条线")
    print("=" * 48)
    print("")


main()
