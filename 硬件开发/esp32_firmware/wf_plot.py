# -*- coding: utf-8 -*-
"""
_wf_plot.py - 把抓回的 IR 序列画成 ASCII 图, 直接看波形长什么样

排障铁律: 先看波形, 再谈算法。数值看不出来的形状问题, 画出来一眼就清楚。

用法: python _wf_plot.py [捕获文件] [起始秒] [持续秒]
"""

import re
import sys


def load(path):
    lvl = re.compile(r"^RAW,(\d+),(\d+),(.*)$")
    samples = []
    for line in open(path, encoding="utf-8"):
        m = lvl.match(line.strip())
        if not m:
            continue
        samples.extend(int(t) for t in m.group(3).split(",") if t.strip())
    return samples


def ema(vals, a):
    out = []
    s = vals[0]
    for v in vals:
        s = s + a * (v - s)
        out.append(s)
    return out


def firmware_filter(x, fs=100.0):
    """复刻 hrv.py 的信号链: HP(alpha=0.02) → LP(alpha=0.25)"""
    dc = ema(x, 0.02)
    hp = [x[i] - dc[i] for i in range(len(x))]
    lp = ema(hp, 0.25)
    return hp, lp


def plot(title, vals, rows=18, width=100, lo=None, hi=None):
    n = len(vals)
    if n > width:
        step = n / float(width)
        v = [vals[int(i * step)] for i in range(width)]
    else:
        v = list(vals)
    lo = min(v) if lo is None else lo
    hi = max(v) if hi is None else hi
    rng = (hi - lo) or 1.0
    grid = [[" "] * len(v) for _ in range(rows)]
    for c, y in enumerate(v):
        r = int((y - lo) / rng * (rows - 1))
        r = max(0, min(rows - 1, r))
        grid[rows - 1 - r][c] = "#"
    print("\n%s   [%.0f .. %.0f]" % (title, lo, hi))
    for r in range(rows):
        lab = hi - (hi - lo) * r / (rows - 1)
        print("  %7.0f |%s" % (lab, "".join(grid[r])))
    print("          +" + "-" * len(v))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "wf_sample_20260930.txt"
    t0 = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    dur = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    x = load(path)
    fs = 101.0
    print("=" * 74)
    print("  波形可视化   样本 %d  用时 %.1fs  实测率 ≈%.0f Hz"
          % (len(x), len(x) / fs, fs))
    print("=" * 74)

    s = int(t0 * fs)
    e = min(len(x), int((t0 + dur) * fs))
    seg = x[s:e]
    if len(seg) < 30:
        print("区间太短")
        return

    hp, lp = firmware_filter(x)          # 全序列滤, 取段(避免边界暂态)

    plot("① 原始 IR (未滤波)", seg)
    plot("② 去直流后 HP(fc≈0.3Hz) —— 注意慢漂移还剩多少",
         hp[s:e], lo=min(hp[s:e]), hi=max(hp[s:e]))
    plot("③ HP+LP 后(固件实际用来找峰的信号)", lp[s:e],
         lo=min(lp[s:e]), hi=max(lp[s:e]))

    # 局部极大值(固件口径)
    y = lp[s:e]
    pk = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 0:
            pk.append((i, y[i]))
    amp = sorted(a for _i, a in pk)
    med = amp[len(amp) // 2] if amp else 0
    print("\n  该段内局部极大 %d 个, 幅度中位 %.0f" % (len(pk), med))
    print("  峰位(相对段首, ms) / 幅度:")
    for i, a in pk:
        print("      %6.0f ms   幅度 %6.1f   (%s)"
              % (i * 1000.0 / fs, a,
                 "高于门限" if a > 0.5 * med else "低于 0.5×中位"))
    if med:
        gate = max(60.0, 0.5 * med)
        print("  固件门限 = 0.5×中位 = %.0f  →  %d 个峰能过"
              % (gate, sum(1 for _i, a in pk if a > gate)))

    # 频谱: 脉搏带 vs 漂移带
    print("\n  全序列频谱(自相关法估主周期)")
    m = len(x)
    mean = sum(x) / m
    z = [v - mean for v in x]
    best = None
    for lag in range(int(0.5 * fs), int(5.0 * fs)):
        s1 = sum(z[i] * z[i + lag] for i in range(0, m - lag, 2))
        if best is None or s1 > best[1]:
            best = (lag, s1)
    if best:
        print("    自相关最强 lag = %d 样本 = %.0f ms  → %.1f 次/分"
              % (best[0], best[0] * 1000.0 / fs, 60000.0 / (best[0] * 1000.0 / fs)))
        print("    (若这个数字落在 12~30 就是呼吸/慢漂移, 落在 60~120 才是脉搏)")
    print("=" * 74)


main()
