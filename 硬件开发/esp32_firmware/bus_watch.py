# -*- coding: utf-8 -*-
"""
bus_watch.py — I2C 总线实时监视器(边拔边看)

用法(PC 端):
    mpremote connect COM6 run bus_watch.py

为什么需要它:
    bus_probe.py 已经定量证明 SDA(GPIO21) 被**一条阻抗极低的通路拉到地**
    (下拉阻抗 < 约 1.1kΩ, 与 3.3k 上拉并联仍读逻辑 0)。
    但"是谁在拉"靠静态测量分不出来 —— 必须**逐个摘除器件**, 观察哪一次摘除
    让 SDA 恢复为 1。

    本脚本持续盯着两条总线电平, 一旦发生**跳变就立刻打印**, 所以你拔线的
    瞬间就能看到结果, 不需要拔完再跑一次脚本。

操作步骤:
    1. 运行本脚本, 会先打一份基线快照
    2. 保持脚本运行, **一次只拔一个模块**(或一根线), 看屏幕有没有跳变
    3. 看到 [*** 跳变 ***] 就是找到了 —— 那个器件/那根线就是元凶
    4. 想更细: 只拔那一根**可疑的线**(比如 SDA 线), 而不是整个模块

时长由 WATCH_SEC 控制(默认 90 秒), 想提前结束按 Ctrl-C。
"""

import time
from machine import Pin, I2C

SDA   = 21
SCL   = 22
FREQ  = 400000
WATCH_SEC = 90          # 监视时长(秒)
POLL_MS   = 200         # 电平轮询周期
SCAN_MS   = 3000        # 总线扫描周期


def levels():
    """只用输入模式读两条线, 返回 (sda, scl)。不驱动总线。"""
    out = []
    for n in (SDA, SCL):
        p = Pin(n, Pin.IN, Pin.PULL_UP)
        out.append(p.value())
        Pin(n, Pin.IN)
    return out[0], out[1]


def scan():
    try:
        i2c = I2C(0, scl=Pin(SCL), sda=Pin(SDA), freq=FREQ)
        return sorted(i2c.scan())
    except Exception:
        return []


def fmt_addrs(addrs):
    return ",".join("0x%02X" % a for a in addrs) if addrs else "(无)"


def banner(sda, scl, addrs):
    print("  SDA=%d  SCL=%d   在线器件: %s" % (sda, scl, fmt_addrs(addrs)))


print("=" * 62)
print(" I2C 总线实时监视器   SDA=GPIO%d  SCL=GPIO%d  @%d Hz" % (SDA, SCL, FREQ))
print("=" * 62)
print(" 判读: SDA 正常应为 1。当前若为 0, 说明有东西把它硬拉到地。")
print(" 本脚本会持续盯 90 秒; 请边跑边拔模块, 拔一个看一次屏幕。")
print(" 想提前结束按 Ctrl-C。")
print("=" * 62)

try:
    sda, scl = levels()
except Exception as e:
    print("电平读取失败: %s" % e)
    sda, scl = -1, -1
addrs = scan()
print("\n[基线快照 @0.0s]")
banner(sda, scl, addrs)
if sda == 0:
    print("  -> SDA 处于异常低电平, 请开始逐个摘除器件观察跳变")
print("-" * 62)

t0 = time.ticks_ms()
last_scan = t0
last_beat = t0
sda0 = sda
history = [(0.0, sda, scl, tuple(addrs))]
change_n = 0

while True:
    el = time.ticks_diff(time.ticks_ms(), t0)
    if el >= WATCH_SEC * 1000:
        break

    # ---- 电平 ----
    try:
        nsda, nscl = levels()
    except Exception:
        nsda, nscl = -1, -1

    # ---- 扫描(降频) ----
    naddrs = addrs
    if time.ticks_diff(time.ticks_ms(), last_scan) >= SCAN_MS:
        last_scan = time.ticks_ms()
        naddrs = scan()

    changed = (nsda != sda) or (nscl != scl) or (naddrs != addrs)
    if changed:
        change_n += 1
        sec = el / 1000.0
        print("\n[*** 跳变 #%d  @%.1fs ***]" % (change_n, sec))
        print("   SDA %d -> %d     SCL %d -> %d" % (sda, nsda, scl, nscl))
        print("   在线器件: %s -> %s" % (fmt_addrs(addrs), fmt_addrs(naddrs)))
        if sda == 0 and nsda == 1:
            print("   >>> SDA 恢复为高 —— 你刚才拔掉/动过的东西就是元凶 <<<")
        if nsda == 0 and sda == 1:
            print("   >>> SDA 被拉低 —— 你刚才接上的东西就是元凶 <<<")
        if 0x57 in naddrs and 0x57 not in addrs:
            print("   >>> MAX30102 (0x57) 上线了 <<<")
        if 0x57 in addrs and 0x57 not in naddrs:
            print("   >>> MAX30102 (0x57) 掉线了 <<<")
        sda, scl, addrs = nsda, nscl, naddrs
        history.append((sec, sda, scl, tuple(addrs)))
    else:
        sda, scl, addrs = nsda, nscl, naddrs
        # 心跳, 证明脚本还活着
        if time.ticks_diff(time.ticks_ms(), last_beat) >= 5000:
            last_beat = time.ticks_ms()
            print("  ... %.0fs  SDA=%d SCL=%d  在线=%s"
                  % (el / 1000.0, sda, scl, fmt_addrs(addrs)))

    time.sleep_ms(POLL_MS)

print()
print("=" * 62)
print(" 监视结束 —— 状态变化历史")
print("=" * 62)
print("  %7s   %4s %4s   %s" % ("时刻(s)", "SDA", "SCL", "在线器件"))
for sec, a, b, ad in history:
    print("  %7.1f   %4d %4d   %s" % (sec, a, b, fmt_addrs(list(ad))))

print()
sda_final, scl_final = levels()
print(" 最终状态: SDA=%d  SCL=%d  在线器件=%s"
      % (sda_final, scl_final, fmt_addrs(scan())))
if sda_final == 1:
    print(" SDA 已恢复 —— 按上面历史里最后一次 0->1 跳变的时刻, 对应你拔的那个器件。")
else:
    print(" SDA 仍为 0 —— 说明拉低它的不是被拔掉的模块, 而是:")
    print("   · ESP32 与面包板之间那根 SDA 线本身(或它插错的孔)")
    print("   · 面包板内部短路(换一个孔/换一块板试)")
    print("   · ESP32 的 GPIO21 已损坏")
    print(" 下一步: 只留 ESP32, SDA 线从 GPIO21 拉出后**悬空不接任何东西**,")
    print(" 再跑一次本脚本。若空载时 SDA 仍是 0, 问题在 ESP32 或线材本身。")
