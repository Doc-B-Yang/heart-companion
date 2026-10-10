# -*- coding: utf-8 -*-
"""
_fiducial_test.py - 用真实抓回的波形, 离线对拍几种"心搏基准点"方法

背景: 实测发现峰位法(RR 取收缩峰间隔)在真实指腹信号上抖动极大:
  · 相邻心搏原始幅度可达 3 倍差(925 vs 350 counts)
  · 部分心搏是**双峰**(收缩峰 + 反射波峰, 相隔 ~176ms)
  · 由此产生的"早到假峰"会把紧随其后的真峰顶进不应期 → RR 变成
    (短 + 长) 对, 例如 656 + 997 ≈ 2×826

原理: 收缩**峰顶**对幅度调制、反射波、基线漂移都敏感; 而**升支最陡处**
      (最大正斜率) 或**升支起点(足点)** 是波形中最稳定的位置 —— 它是
      压力波到达的物理时刻, 与幅度无关。文献里 PPG-HRV 的标准做法就是
      用 foot / max-slope 作为基准点, 而不是峰顶。

本脚本把若干候选方法在同一段真实数据上跑一遍, 比较 RMSSD 是否落回
生理区间(静息 20~50 ms), 以此决定改哪个。

用法: python _fiducial_test.py [捕获文件]
"""

import math
import re
import sys

sys.path.insert(0, ".")
from hrv import HRVAnalyzer      # 基准(现状)  # noqa: E402

FS = 101.0          # 实测输出率


# ------------------------------------------------------------ 工具
def load(path):
    lvl = re.compile(r"^RAW,(\d+),(\d+),(.*)$")
    out = []
    for line in open(path, encoding="utf-8"):
        m = lvl.match(line.strip())
        if m:
            out.extend(int(t) for t in m.group(3).split(",") if t.strip())
    return out


def ema(v, a):
    o = []
    s = v[0]
    for t in v:
        s = s + a * (t - s)
        o.append(s)
    return o


def highpass(x, fc, fs=FS, n=2):
    """n 级一阶高通级联(比单级陡), 用于压掉呼吸/位移引起的基线漂移"""
    y = list(x)
    a = min(1.0, 2 * math.pi * fc / fs)
    for _ in range(n):
        d = ema(y, a)
        y = [y[i] - d[i] for i in range(len(y))]
    return y


def lowpass(x, fc, fs=FS):
    return ema(x, min(1.0, 2 * math.pi * fc / fs))


def rmssd(rr):
    if len(rr) < 2:
        return 0.0
    return math.sqrt(sum((rr[i] - rr[i - 1]) ** 2
                         for i in range(1, len(rr))) / (len(rr) - 1))


def sdnn(rr):
    if len(rr) < 2:
        return 0.0
    m = sum(rr) / len(rr)
    return math.sqrt(sum((v - m) ** 2 for v in rr) / (len(rr) - 1))


def pnn50(rr):
    if len(rr) < 2:
        return 0.0
    return 100.0 * sum(1 for i in range(1, len(rr))
                       if abs(rr[i] - rr[i - 1]) > 50) / (len(rr) - 1)


def pair_repair(rr, tol=0.12, short=0.85):
    """(短,长) 对修复: 二者之和 ≈ 2×中位 → 拆回两个等于均值的拍"""
    out = list(rr)
    i = 0
    n = 0
    while i < len(out) - 1 and len(out) >= 5:
        med = sorted(out)[len(out) // 2]
        a, b = out[i], out[i + 1]
        if (abs(a + b - 2 * med) <= tol * 2 * med
                and (a < short * med or b < short * med)
                and a > 0.4 * med and b > 0.4 * med):
            m = (a + b) / 2.0
            out[i] = m
            out[i + 1] = m
            n += 1
            i += 2
        else:
            i += 1
    return out, n


def report(name, rr, extra=""):
    if len(rr) < 4:
        print("  %-40s (RR 不足: %d)" % (name, len(rr)))
        return None
    hr = (sum(rr) / len(rr))
    hr_bpm = 60000.0 / hr
    print("  %-40s HR %5.1f  RMSSD %6.1f  SDNN %6.1f  pNN50 %5.1f  n=%3d %s"
          % (name, hr_bpm, rmssd(rr), sdnn(rr), pnn50(rr), len(rr), extra))
    return rmssd(rr)


# ------------------------------------------------------------ 候选方法
def method_current(x, peak_ratio=0.5):
    """现状: 固件 hrv.py 的峰位法"""
    h = HRVAnalyzer(fs=100, window_sec=60, min_peak_ms=450,
                    max_peak_ms=2000, peak_ratio=peak_ratio)
    for v in x:
        h.add_sample(v)
    return list(h.rr)


def _maxslope(x, hp_fc, hp_n, lp_fc, diff_n, slope_ratio, refr_ms):
    """公共骨架: 带通 → 差分 → 找"升支最陡处"作为基准点

    返回 (基准点时间 ms 列表, 差分序列)
    """
    y = lowpass(highpass(x, hp_fc, n=hp_n), lp_fc)
    n = len(y)
    # 用 diff_n 个样本的差做斜率估计(抗单点噪声; diff_n=3 约等于 30ms 基线)
    d = [0.0] * n
    for i in range(diff_n, n):
        d[i] = y[i] - y[i - diff_n]
    # 斜率门限: 取"正斜率局部极大"的中位数比例
    cand = []
    for i in range(diff_n + 1, n - 1):
        if d[i] > d[i - 1] and d[i] >= d[i + 1] and d[i] > 0:
            cand.append((i, d[i]))
    if not cand:
        return [], d
    amps = sorted(a for _i, a in cand)
    med = amps[len(amps) // 2]
    gate = slope_ratio * med
    beats = []
    last = -1e9
    for i, a in cand:
        if a < gate:
            continue
        t = i * 1000.0 / FS
        if t - last >= refr_ms:
            beats.append(t)
            last = t
    return beats, d


def method_maxslope(x, tag="max-slope(升支最陡处)",
                    hp_fc=0.7, hp_n=2, lp_fc=5.0, diff_n=3,
                    slope_ratio=0.4, refr_ms=350, repair=True):
    beats, _d = _maxslope(x, hp_fc, hp_n, lp_fc, diff_n, slope_ratio, refr_ms)
    rr = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
    rr = [v for v in rr if 320 <= v <= 2000]
    extra = ""
    if repair:
        rr, n = pair_repair(rr)
        extra = "(修复 %d)" % n
    return report(tag, rr, extra), rr


def method_foot(x, tag="foot(升支起点)",
                hp_fc=0.7, hp_n=2, lp_fc=5.0, diff_n=3,
                slope_ratio=0.4, refr_ms=350, search_ms=120):
    """足点: 在"升支最陡处"往前回溯到斜率首次转正的位置"""
    y = lowpass(highpass(x, hp_fc, n=hp_n), lp_fc)
    beats, d = _maxslope(x, hp_fc, hp_n, lp_fc, diff_n, slope_ratio, refr_ms)
    if not beats:
        return None, []
    n = len(y)
    k = int(search_ms * FS / 1000.0)
    out = []
    for t in beats:
        i = int(t * FS / 1000.0)
        j = i
        lo = max(diff_n + 1, i - k)
        # 向前找斜率最小值(上升前的谷)
        while j > lo and d[j] > 0:
            j -= 1
        best = j
        for m in range(j, min(i + 1, n)):
            if d[m] < d[best]:
                best = m
        out.append(best * 1000.0 / FS)
    rr = [out[i] - out[i - 1] for i in range(1, len(out))]
    rr = [v for v in rr if 320 <= v <= 2000]
    rr, nfix = pair_repair(rr)
    return report(tag, rr, "(修复 %d)" % nfix), rr


def method_peakwin(x, tag="峰位+拍窗内取最大峰"):
    """峰位法 + 拍窗约束: 接受一拍后, 在 0.45×RR 内不再接受新峰,
    且在该窗内取幅度最大的候选峰 —— 直接压掉"早到假峰"。
    """
    y = lowpass(highpass(x, 0.7, n=2), 5.0)
    n = len(y)
    cand = []
    for i in range(1, n - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 0:
            cand.append((i, y[i]))
    if not cand:
        return None, []
    amps = sorted(a for _i, a in cand)
    med = amps[len(amps) // 2]
    gate = 0.45 * med
    beats = []
    refr = 450
    last = -1e9
    for i, a in cand:
        t = i * 1000.0 / FS
        if a < gate or t - last < refr:
            continue
        beats.append(t)
        last = t
    rr = [beats[i] - beats[i - 1] for i in range(1, len(beats))]
    rr = [v for v in rr if 320 <= v <= 2000]
    rr, nfix = pair_repair(rr)
    return report(tag, rr, "(修复 %d)" % nfix), rr


# ------------------------------------------------------------ 主流程
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "wf_sample_20260930.txt"
    x = load(path)
    print("=" * 108)
    print("  心搏基准点方法对拍   样本 %d  用时 %.1fs  实测率 %.1f Hz"
          % (len(x), len(x) / FS, FS))
    print("=" * 108)
    print("  判据: 静息真实 RMSSD 应落在 20~50 ms; pNN50 应 <10%")
    print()

    report("① 现状: 峰位法 (固件 hrv.py)", method_current(x))
    report("①' 现状 + (短,长)对修复", pair_repair(method_current(x))[0],
           "(修复 %d)" % pair_repair(method_current(x))[1])

    print()
    for fc in (0.5, 0.7, 1.0):
        method_maxslope(x, "② max-slope  HP=%.1fHz×2" % fc, hp_fc=fc)
    print()
    method_foot(x, "③ foot(足点) HP=0.7Hz×2")
    print()
    method_peakwin(x)
    print()
    # 斜率差分窗口敏感性
    for dn in (2, 3, 5, 7):
        method_maxslope(x, "④ max-slope diff=%d (%.0fms)" % (dn, dn * 1000 / FS),
                        diff_n=dn)
    print("=" * 108)


if __name__ == "__main__":
    main()
