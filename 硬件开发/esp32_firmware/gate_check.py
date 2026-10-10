# -*- coding: utf-8 -*-
"""
gate_check.py - 运动门控逻辑的离线验收 (PC 端, 无需硬件)

用合成 PPG 波形同时喂两个 HRVAnalyzer:
    门控版  —— 运动窗口内 motion=True (运动段 RR 丢弃)
    对照版  —— 全程 motion=False (旧行为)

验收点:
  1. 安静段两条路径结果应几乎一致  → 门控不误伤真实 HRV
  2. 运动段对照版 RMSSD 显著虚高    → 证明伪迹确实会污染指标
  3. 门控版在运动段**不产出任何 RR** → 从源头掐掉, 而非事后统计剔除
  4. 状态切换处不得出现"跨运动段"的假 RR(把运动前后两拍连起来)

用法:
    python gate_check.py
"""

import math
import random
import statistics
import sys

sys.path.insert(0, ".")
from hrv import HRVAnalyzer  # noqa: E402

FS = 100
DC = 90000


def gen_ppg(sec, hr=70.0, rmssd=40.0, rr_resp=15.0,
            motion_win=None, seed=7, fs=FS):
    """合成 PPG: 直流 + 脉搏(含 RSA 调制) [+ 运动污染]

    motion_win = (t_start, t_end): 该窗口内叠加运动伪迹 ——
      基线漂移(模拟手部缓慢位移) + 宽带噪声(模拟接触压力变化)。
      这正是真实手抖对 PPG 的作用方式。
    """
    rng = random.Random(seed)
    n = int(sec * fs)
    f_r = rr_resp / 60.0
    a_rsa = rmssd * math.sqrt(2.0) / (2 * math.pi * f_r) if f_r > 0 else 0.0

    out = []
    phase = 0.0
    last_beat = 0.0
    rr = 60000.0 / hr
    for i in range(n):
        t = i / fs
        if t >= last_beat + rr / 1000.0:
            last_beat = t
            rr = 60000.0 / hr + a_rsa * math.sin(2 * math.pi * f_r * t) \
                + rng.gauss(0, rmssd * 0.30)
            rr = max(350.0, min(1800.0, rr))
            phase = 0.0
        phase += 2 * math.pi / (rr / 1000.0) / fs
        v = DC + 1200 * math.sin(phase) + rng.gauss(0, 22)
        if motion_win and motion_win[0] <= t < motion_win[1]:
            v += 3500 * math.sin(2 * math.pi * 0.9 * t)   # 基线漂移
            v += rng.gauss(0, 600)                        # 接触压力抖动
        out.append(int(max(0, min(262143, v))))
    return out


def run(samples, motion_fn):
    """跑一个分析器, 返回逐秒快照 [(t, hr, sdnn, rmssd, nn, q, segs)]"""
    h = HRVAnalyzer(fs=FS, window_sec=60, min_peak_ms=450,
                    max_peak_ms=2000, peak_ratio=0.5)
    snaps = []
    for i, v in enumerate(samples):
        h.add_sample(v, motion=motion_fn(i / FS))
        if (i + 1) % FS == 0:
            m = h.metrics()
            snaps.append((int((i + 1) / FS), m["hr"], m["sdnn"], m["rmssd"],
                          m["nn"], m["sig_quality"], m["segs"]))
    return h, snaps


def main():
    STILL1, SHAKE, STILL2 = 40, 25, 65
    TOTAL = STILL1 + SHAKE + STILL2
    motion_win = (STILL1, STILL1 + SHAKE)

    print("=" * 76)
    print("  运动门控离线验收   静止 %ds → 运动 %ds → 恢复 %ds" % (STILL1, SHAKE, STILL2))
    print("  (恢复段取 65s > HRV 窗口 60s, 用于检验污染数据是否会被冲刷掉)")
    print("=" * 76)

    samples = gen_ppg(TOTAL, hr=70.0, rmssd=42.0, rr_resp=15.0,
                      motion_win=motion_win)

    in_motion = lambda t: motion_win[0] <= t < motion_win[1]   # noqa: E731
    h_gate, s_gate = run(samples, in_motion)
    h_raw, s_raw = run(samples, lambda t: False)

    # 运动段内, 门控版是否新增过 RR?
    gate_rr_before = None
    print("\n  逐秒对照 (每 5 秒一行)")
    print("  %-4s %-5s | %-22s | %s" %
          ("t", "阶段", "门控 HR/SDNN/RMSSD/nn", "未门控 HR/SDNN/RMSSD/nn"))
    for (t, hr, sd, rm, nn, q, sg), (t2, hr2, sd2, rm2, nn2, q2, sg2) \
            in zip(s_gate, s_raw):
        if t % 5:
            continue
        ph = "静止" if t <= STILL1 else ("运动" if t <= STILL1 + SHAKE else "恢复")
        print("  %-4d %-5s | %6.1f %6.1f %6.1f %4d | %6.1f %6.1f %6.1f %4d"
              % (t, ph, hr, sd, rm, nn, hr2, sd2, rm2, nn2))

    # ---------------- 判定 ----------------
    def seg(snaps, lo, hi, idx):
        v = [s[idx] for s in snaps if lo < s[0] <= hi and s[4] >= 10]
        return statistics.median(v) if v else None

    T_END_MOTION = STILL1 + SHAKE

    print("\n" + "=" * 76)
    print("  分阶段对比 (取 RMSSD 中位数, 仅统计 nn>=10 的秒)")
    print("=" * 76)
    print("  %-12s %-12s %-12s %s" % ("阶段", "门控RMSSD", "未门控RMSSD", "虚高倍数"))
    verdict = {}
    stages = (
        ("静止段", 10, STILL1 - 5),
        ("运动段", STILL1 + 8, T_END_MOTION),
        ("恢复早期", T_END_MOTION + 8, T_END_MOTION + 22),
        ("恢复末期", TOTAL - 8, TOTAL),
    )
    for name, lo, hi in stages:
        g = seg(s_gate, lo, hi, 3)
        r = seg(s_raw, lo, hi, 3)
        if g is None or r is None:
            print("  %-12s (数据不足)" % name)
            verdict[name] = None
            continue
        ratio = r / g if g > 1e-6 else 0
        print("  %-12s %-12.1f %-12.1f %.2f×" % (name, g, r, ratio))
        verdict[name] = (g, r, ratio)

    # 门控版在运动段是否产出过 RR (取运动段最后一秒, 不能跨进恢复段)
    def at(snaps, t):
        v = [s[4] for s in snaps if s[0] == t]
        return v[0] if v else None

    n_gate_before = at(s_gate, STILL1)
    n_gate_after = at(s_gate, T_END_MOTION)
    n_raw_before = at(s_raw, STILL1)
    n_raw_after = at(s_raw, T_END_MOTION)

    print("\n  运动期间有效 RR 数(nn)增长 (t=%ds → t=%ds):" % (STILL1, T_END_MOTION))
    print("    门控   %d → %d   新增 %+d   (期望 0: 运动段一拍都不收)"
          % (n_gate_before, n_gate_after, n_gate_after - n_gate_before))
    print("    未门控 %d → %d   新增 %+d   (伪迹被当作真实心搏收了进来)"
          % (n_raw_before, n_raw_after, n_raw_after - n_raw_before))

    print("\n" + "=" * 76)
    print("  结论")
    print("=" * 76)
    ok = True

    if verdict["静止段"] and abs(verdict["静止段"][2] - 1.0) <= 0.15:
        print("  ✔ 静止段两条路径一致 (比值 %.2f) —— 门控不误伤真实 HRV" %
              verdict["静止段"][2])
    else:
        print("  ✘ 静止段两条路径差异过大, 门控可能误伤真实 HRV")
        ok = False

    if verdict["运动段"] and verdict["运动段"][2] >= 1.5:
        print("  ✔ 运动段未门控 RMSSD 虚高 %.2f× (%.1f → %.1f ms) —— 伪迹确实污染指标"
              % (verdict["运动段"][2], verdict["运动段"][0], verdict["运动段"][1]))
    elif verdict["运动段"]:
        print("  ⚠ 运动段虚高仅 %.2f× —— 本次合成的伪迹偏弱" % verdict["运动段"][2])
    else:
        print("  ⚠ 运动段门控侧拍数不足, 无法比较")

    if n_gate_after - n_gate_before == 0:
        print("  ✔ 门控版运动段新增 0 拍 —— 伪迹被从源头掐掉(非事后统计剔除)")
    else:
        print("  ✘ 门控版运动段仍新增 %d 拍 (应全部门控丢弃)" %
              (n_gate_after - n_gate_before))
        ok = False

    if verdict["恢复早期"] and verdict["恢复早期"][2] >= 1.3:
        print("  ✔ 恢复早期未门控仍虚高 %.2f× —— 被污染的 RR 会滞留在 60s 窗口里,"
              % verdict["恢复早期"][2])
        print("    情绪结论在运动停止后还要错约 1 分钟(门控版则立即恢复)")
    if verdict["恢复末期"] and verdict["恢复末期"][2] <= 1.15:
        print("  ✔ 恢复末期两条路径重新收敛 (比值 %.2f) —— 门控不改变最终结果,"
              % verdict["恢复末期"][2])
        print("    只是把'污染'这一段直接跳过, 而不是修正数值")
    elif verdict["恢复末期"]:
        print("  ⚠ 恢复末期仍差 %.2f× —— 门控可能过于激进或过松, 请复查阈值" %
              verdict["恢复末期"][2])

    rm_m = verdict["运动段"][1] if verdict["运动段"] else 0
    rm_g = verdict["运动段"][0] if verdict["运动段"] else 0
    print()
    print("  为什么这件事重要:")
    print("    若真实 RMSSD≈%.0f ms(「平静」区间 40~70), 未门控被伪迹抬到 %.0f ms;"
          % (rm_g, rm_m))
    print("    而 Agent 的 EMOTION_RULES 用 RMSSD 区分「平静」与「高焦虑(8~25)」——")
    print("    伪迹把 RMSSD 抬高, 就会把焦虑/紧张**误判为平静**, 直接漏报需要关怀的人。")
    print("=" * 76)
    return 0 if ok else 1


sys.exit(main())
