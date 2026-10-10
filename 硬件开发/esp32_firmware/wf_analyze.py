# -*- coding: utf-8 -*-
"""
_wf_analyze.py - 用抓回的原始 IR 序列离线复算, 定位坏拍根因

思路(排障铁律: 先抓波形, 再改算法):
  1. 从 wf_dump 的输出里还原完整 IR 序列(逐秒 RAW 行拼起来)
  2. 用**固件同一份** hrv.py 跑一遍 → 得到检出的峰与 RR
  3. 检查每个峰的幅度, 判断坏拍是"低幅假峰"还是"高幅真峰被拦"
  4. 三种修法各算一次 RMSSD, 量化收益:
       ① 现状(仅 _clean_rr 的 30% 中位剔除)
       ② 提高门限 peak_ratio
       ③ RR 对合并修复(656+997≈2×中位 → 拆回两拍均值)
       ④ ②+③ 同时

用法: python _wf_analyze.py [捕获文件]
"""

import math
import re
import sys

sys.path.insert(0, ".")
from hrv import HRVAnalyzer  # noqa: E402

FS = 100.0


def load(path):
    """还原 IR 序列 + 每秒样本数"""
    lvl = re.compile(r"^RAW,(\d+),(\d+),(.*)$")
    samples = []
    per_sec = []
    cur = 0
    for line in open(path, encoding="utf-8"):
        m = lvl.match(line.strip())
        if not m:
            continue
        if int(m.group(2)) == 0 and cur:
            per_sec.append(cur)
            cur = 0
        vals = [int(t) for t in m.group(3).split(",") if t.strip()]
        samples.extend(vals)
        cur += len(vals)
    if cur:
        per_sec.append(cur)
    return samples, per_sec


def run_full(samples, peak_ratio=0.5):
    """用固件的 HRVAnalyzer 跑, 并记录每个峰的幅度与时间"""
    h = HRVAnalyzer(fs=100, window_sec=60, min_peak_ms=450,
                    max_peak_ms=2000, peak_ratio=peak_ratio)
    peak_log = []
    for v in samples:
        before = len(h.peaks)
        h.add_sample(v)
        if len(h.peaks) > before:
            peak_log.append((h.peaks[-1], h.peak_amps[-1]))
    return h, peak_log


def rmssd(rr):
    if len(rr) < 2:
        return 0.0
    return math.sqrt(sum((rr[i] - rr[i - 1]) ** 2
                         for i in range(1, len(rr))) / (len(rr) - 1))


def pair_merge_repair(rr, tol=0.12, short_ratio=0.85):
    """RR 对合并修复

    机理: 一个假峰 S 出现在真峰 P1 后 656ms(早于 P2), 而 P2 距 S 只有
    170ms < 不应期 → P2 被拦掉 → 得到 (P1→S)=656 与 (S→P3)=996,
    二者之和 ≈ 2×真实 RR。

    做法: 若 rr[i]+rr[i+1] ≈ 2×中位(±tol) 且 rr[i] 明显短于中位,
    则把两拍**拆回两个等于其均值的拍** —— 消除 ΔRR 上的虚假跳变,
    同时不引入虚构的变异性(Kubios 式"漏搏/多搏"自动修复)。
    """
    out = [v for v in rr]
    i = 0
    n_fix = 0
    while i < len(out) - 1:
        if len(out) < 5:
            break
        med = sorted(out)[len(out) // 2]
        a, b = out[i], out[i + 1]
        if (abs(a + b - 2 * med) <= tol * 2 * med) and \
                (a < short_ratio * med or b < short_ratio * med) and \
                (a > 0.4 * med and b > 0.4 * med):
            m = (a + b) / 2.0
            out[i] = m
            out[i + 1] = m
            n_fix += 1
            i += 2
        else:
            i += 1
    return out, n_fix


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "wf_sample_20260930.txt"
    samples, per_sec = load(path)
    n = len(samples)
    print("=" * 74)
    print("  离线复算   样本 %d 个   逐秒 %s" % (n, per_sec))
    print("  实测采样率 = %.2f Hz" % (n / float(len(per_sec))))
    print("=" * 74)

    h, pk = run_full(samples)
    rr = list(h.rr)
    amps = [a for _t, a in pk]

    print("\n[1] 检出峰 (%d 个)" % len(pk))
    print("  峰幅度: 中位 %.0f  最小 %.0f  最大 %.0f"
          % (sorted(amps)[len(amps) // 2], min(amps), max(amps)))
    lo_thresh = 0.7 * sorted(amps)[len(amps) // 2]
    low = [(t, a) for t, a in pk if a < lo_thresh]
    print("  低于中位 70%%(%.0f) 的峰: %d 个" % (lo_thresh, len(low)))
    for t, a in low:
        print("       t=%8.0f ms  幅度=%.0f  (中位的 %.0f%%)"
              % (t, a, 100.0 * a / sorted(amps)[len(amps) // 2]))

    print("\n  峰间隔序列表:")
    print("    " + " ".join("%.0f" % r for r in rr))
    med = sorted(rr)[len(rr) // 2] if rr else 0
    print("  RR 中位 = %.0f ms (%.1f bpm)" % (med, 60000.0 / med if med else 0))

    print("\n[2] 三种修法的效果对比")
    print("  %-34s %10s %10s" % ("方法", "RMSSD(ms)", "RR 数"))
    base = rr
    print("  %-34s %10.1f %10d" % ("① 现状(仅 30%% 中位剔除)", rmssd(base), len(base)))

    for pr in (0.6, 0.7, 0.8):
        h2, pk2 = run_full(samples, peak_ratio=pr)
        rr2 = list(h2.rr)
        print("  %-34s %10.1f %10d"
              % ("② peak_ratio=%.1f" % pr, rmssd(rr2), len(rr2)))

    fixed, nfix = pair_merge_repair(base)
    print("  %-34s %10.1f %10d   (修复 %d 处)"
          % ("③ RR 对合并修复", rmssd(fixed), len(fixed), nfix))

    h3, pk3 = run_full(samples, peak_ratio=0.7)
    f3, n3 = pair_merge_repair(list(h3.rr))
    print("  %-34s %10.1f %10d   (修复 %d 处)"
          % ("④ ②(0.7) + ③ 一起", rmssd(f3), len(f3), n3))

    print("\n[3] 结论")
    if med:
        r_now = rmssd(base)
        r_best = min(rmssd(fixed), rmssd(f3))
        print("  当前 RMSSD = %.1f ms  →  修复后可达 %.1f ms" % (r_now, r_best))
        if 15 <= r_best <= 60:
            print("  ✔ 修复后落入**生理合理区间**(静息 RMSSD 20~50ms) ——")
            print("    说明坏拍是主因, 且可通过算法修掉, 不必换硬件。")
        else:
            print("  ⚠ 修复后仍偏离生理区间, 需继续查信号质量。")
    print("=" * 74)


main()
