# -*- coding: utf-8 -*-
"""
pipeline_check.py - 全链路验收脚本（固件仿真 → bridge → Agent）
================================================================

回答三个必须用数据回答的问题:
  Q1 这条链路（固件字段 → 契约包 → 情绪识别）到底通不通?
  Q2 情绪识别准确率是多少? 哪几类容易混?
  Q3 个人静息基线归一化到底是"锦上添花"还是"必需"? —— 用同一批数据对比。

方法
----
  用 firmware_sim 生成**已知情绪**的 PPG → 走与真机完全相同的处理路径
  （hrv 算法链 → bridge 组装 → HeartCompanionSystem.tick）→ 统计识别结果。

注意: 这是**算法链路的自洽性验收**，不是人体验证。真实准确率必须在
      20 人以上的被试实验中得到（见计划书 §5）。本脚本的意义是把
      "接口/字段/量纲/阈值"这类工程问题在硬件到货前就清零。

用法
----
    python pipeline_check.py                # 稳态场景准确率(原始量纲)
    python pipeline_check.py --with-calib   # 同时对比"个人基线归一化"
    python pipeline_check.py --script       # 剧本回放, 打印情绪时间线
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge import AgentLink, Baseline, Bridge, TachogramSpectrum   # noqa: E402
from firmware_sim import SCENARIOS, SEQUENCES, FirmwareSim          # noqa: E402

TRUTH2EMO = {"calm": "平静", "happy": "愉悦", "fatigue": "疲劳",
             "low_anx": "低焦虑", "high_anx": "高焦虑"}

WARMUP_SEC = 60          # 前 60 秒为算法收敛期, 不计入统计
RUN_SEC = 200            # 每个场景总时长


class _Args:
    """伪造 argparse 命名空间, 供 Bridge 复用"""
    def __init__(self, baseline_file, user="check", no_agent=True,
                 min_quality=0.5, lfhf_calib=None):
        self.baseline_file = str(baseline_file)
        self.user = user
        self.no_agent = no_agent
        self.min_quality = min_quality
        self.lfhf_calib = lfhf_calib
        self.lfhf_smooth = 20.0
        self.spec_window = 60.0
        self.calib = False
        self.agent_dir = str(Path(__file__).resolve().parent.parent.parent
                             / "心理助手Agent-1" / "hrv_companion_agent(1)")
        self.csv = self.jsonl = None
        self.feedback_delay = 90


def run_scenario(scenario: str, norm: Baseline, use_norm: bool,
                 args, seed: int = 3, run_sec: int = RUN_SEC):
    """跑一个场景, 返回 [(truth_emo, pred_emo), ...] 与原始/归一化特征样本"""
    bridge = Bridge(args)
    bridge.spec = TachogramSpectrum(max_span_sec=args.spec_window,
                                    lfhf_calib=args.lfhf_calib)
    bridge.agent = AgentLink(Path(args.agent_dir), args.user, enabled=True)
    if not bridge.agent.ok:
        raise SystemExit("Agent 接入失败: %s" % bridge.agent.error)

    sim = FirmwareSim(scenario=scenario, seed=seed, seconds=run_sec)
    pairs = []
    feats_log = []
    n = 0
    while n < run_sec:
        line = sim.next_line()
        if line is None:
            break
        raw = json.loads(line)
        n += 1
        pkt, meta = bridge.build_packet(raw)
        feed = norm.normalize(pkt) if use_norm else pkt
        bridge.agent.tick(feed)
        st = bridge.agent.state() or {}
        if n > WARMUP_SEC and st.get("emotion") not in (None, "数据积累中"):
            pairs.append((TRUTH2EMO.get(meta["truth"], meta["truth"]),
                          st.get("emotion")))
            feats_log.append({**{k: feed[k] for k in
                                 ("hr", "sdnn", "rmssd", "lf_hf", "rr")},
                              "quality": pkt["sig_quality"],
                              "truth": meta["truth"]})
    bridge.agent.close()
    return pairs, feats_log


def fit_baseline(args, seconds: int = 120) -> Baseline:
    """用 calm 场景生成一份"个人静息基线"(等价于真机上 --baseline 的功能)"""
    bridge = Bridge(args)
    bridge.spec = TachogramSpectrum(max_span_sec=args.spec_window,
                                    lfhf_calib=args.lfhf_calib)
    bridge.agent = AgentLink(Path(args.agent_dir), args.user, enabled=False)
    sim = FirmwareSim(scenario="calm", seed=99, seconds=seconds)
    samples = []
    for _ in range(seconds):
        line = sim.next_line()
        if line is None:
            break
        pkt, meta = bridge.build_packet(json.loads(line))
        if pkt["sig_quality"] >= 0.5 and pkt["rmssd"] > 0 and meta["nn"] >= 20:
            samples.append(pkt)
    b = Baseline(Path(args.baseline_file), args.user)
    b.fit(samples)
    return b


def confusion(pairs):
    """返回 (总体准确率, 混淆矩阵 dict, 每类召回)"""
    if not pairs:
        return 0.0, {}, {}
    cm = defaultdict(lambda: defaultdict(int))
    for truth, pred in pairs:
        cm[truth][pred] += 1
    total = len(pairs)
    correct = sum(v[t] for t, v in cm.items() if t in v)
    rec = {}
    for t, row in cm.items():
        s = sum(row.values())
        rec[t] = row.get(t, 0) / s if s else 0.0
    return correct / total, cm, rec


def print_confusion(cm, title):
    labels = [v for v in TRUTH2EMO.values()]
    cols = [l for l in labels if any(l in cm[t] for t in cm)]
    print("\n  %s" % title)
    print("       " + "".join("%8s" % c for c in cols) + "%8s" % "召回")
    for t in labels:
        if t not in cm:
            continue
        row = cm[t]
        s = sum(row.values())
        line = "  %-5s" % t
        for c in cols:
            line += "%8d" % row.get(c, 0)
        line += "%8.0f%%" % (100.0 * row.get(t, 0) / s if s else 0)
        print(line)


def main():
    ap = argparse.ArgumentParser(description="全链路验收")
    ap.add_argument("--with-calib", action="store_true",
                    help="同时评估个人基线归一化的效果")
    ap.add_argument("--script", action="store_true",
                    help="改跑剧本场景, 打印情绪时间线")
    ap.add_argument("--baseline-file",
                    default=str(Path(__file__).resolve().parent.parent
                                / "baseline" / "_check_temp.json"))
    ap.add_argument("--run-sec", type=int, default=RUN_SEC)
    ap.add_argument("--scenario", default="stress_event",
                    help="--script 模式使用的剧本名")
    a = ap.parse_args()

    print("=" * 78)
    print(" 全链路验收: 固件仿真 → bridge → HeartCompanionSystem")
    print("=" * 78)

    args = _Args(a.baseline_file, user="check")
    print("Agent:", args.agent_dir)

    # ---------------------------------------------------------- 剧本回放
    if a.script:
        name = a.scenario if a.scenario in SEQUENCES else "stress_event"
        args.user = "check_script"
        bridge = Bridge(args)
        bridge.spec = TachogramSpectrum(max_span_sec=args.spec_window,
                                        lfhf_calib=args.lfhf_calib)
        bridge.agent = AgentLink(Path(args.agent_dir), args.user, enabled=True)
        base = fit_baseline(args) if a.with_calib else None
        if base:
            print("\n基线: HR %.1f SDNN %.1f RMSSD %.1f LF/HF %.2f  (来自 calm 场景)"
                  % (base.data["hr"], base.data["sdnn"], base.data["rmssd"],
                     base.data["lf_hf"]))
        sim = FirmwareSim(scenario=name, seed=3)
        print("\n剧本 %s: %s  (总 %ds)\n"
              % (name, SEQUENCES[name], sum(d for _, d in SEQUENCES[name])))
        print("  %-5s %-6s %-6s | %-6s %-6s %-6s %-6s | %-8s %s"
              % ("t", "HR", "RMSSD", "SDNN", "LF/HF", "呼吸", "Q",
                 "识别", "真值"))
        t = 0
        phase_name = SEQUENCES[name][0][0]
        recs = []            # (t, truth_key, pred_zh)
        while True:
            line = sim.next_line()
            if line is None:
                break
            t += 1
            pkt, meta = bridge.build_packet(json.loads(line))
            feed = base.normalize(pkt) if base else pkt
            bridge.agent.tick(feed)
            st = bridge.agent.state() or {}
            truth_zh = TRUTH2EMO.get(meta["truth"], meta["truth"])
            if t == 1 or meta["truth"] != phase_name:
                print("  ---- 进入「%s」 ----" % truth_zh)
                phase_name = meta["truth"]
            if t % 10 == 0:
                print("  %-5d %-6.1f %-6.1f | %-6.1f %-6.2f %-6.1f %-6.2f | "
                      "%-8s %s" % (t, pkt["hr"], pkt["rmssd"], pkt["sdnn"],
                                   pkt["lf_hf"], pkt["rr"], pkt["sig_quality"],
                                   st.get("emotion"), truth_zh))
            recs.append((t, meta["truth"], st.get("emotion")))
        bridge.agent.close()

        # ---- 分段收敛分析 -------------------------------------------------
        # 生理指标本身是"慢变量"（HRV 需 ≥30~60s 统计窗, LF/HF 还需额外平滑）,
        # 因此状态切换后必然有一段爬升期。这里把"收敛延迟"显式量出来,
        # 而不是把爬升期算作错误（那会严重低估系统实际能力）。
        segs = []
        for i, (_t, key, _p) in enumerate(recs):
            if not segs or segs[-1][0] != key:
                segs.append([key, i, i])
            else:
                segs[-1][2] = i
        print("\n" + "-" * 78)
        print("  分段收敛分析（延迟 = 进入该状态后首次达到 80% 一致率所需秒数）")
        print("  %-8s %-6s %-10s %-12s %-12s" %
              ("真值", "段长", "收敛延迟", "稳段一致率", "含爬升一致率"))
        steady_ok = steady_n = 0
        for key, s, e in segs:
            zh = TRUTH2EMO.get(key, key)
            n = e - s + 1
            hits = [1 if recs[i][2] == zh else 0 for i in range(s, e + 1)]
            delay = None
            for d in range(n):
                win = hits[d:]
                if win and sum(win) / len(win) >= 0.8 and len(win) >= 5:
                    delay = d
                    break
            warm = min(WARMUP_SEC, n - 1)
            st_hits = hits[warm:]
            st_rate = (sum(st_hits) / len(st_hits)) if st_hits else 0.0
            steady_ok += sum(st_hits)
            steady_n += len(st_hits)
            print("  %-8s %-6d %-10s %-12s %-12s" % (
                zh, n,
                ("%d s" % delay) if delay is not None else "未收敛",
                "%.0f%%" % (100 * st_rate),
                "%.0f%%" % (100 * sum(hits) / n)))
        if steady_n:
            print("\n  稳态总体一致率（各段剔除前 %ds 爬升期）: %.1f%%  (%d/%d)"
                  % (WARMUP_SEC, 100.0 * steady_ok / steady_n,
                     steady_ok, steady_n))
        print("\n  注意: 上表是**生理统计窗口带来的固有延迟**, 与系统计算延迟")
        print("        (端到端 <1s, 由固件 1 包/秒 + 规则层 O(1) 判定保证) 是两回事。")
        return 0

    # ---------------------------------------------------------- 无归一化
    base_none = Baseline(Path(a.baseline_file), "none")
    base_none.data = {}
    print("\n[1/2] 原始量纲（不做基线归一化）")
    res_raw = {}
    for key in SCENARIOS:
        args.user = "check_raw_" + key
        pairs, _ = run_scenario(key, base_none, False, args,
                                run_sec=a.run_sec)
        res_raw[key] = pairs
    all_raw = [p for v in res_raw.values() for p in v]
    acc_raw, cm_raw, _ = confusion(all_raw)
    print("  总体准确率: %.1f%%  (%d 个样本)" % (100 * acc_raw, len(all_raw)))
    print_confusion(cm_raw, "混淆矩阵 (行=真值, 列=识别)")

    if not a.with_calib:
        print("\n提示: 加 --with-calib 可对比「个人静息基线归一化」的效果。")
        return 0

    # ---------------------------------------------------------- 带归一化
    print("\n[2/2] 个人静息基线归一化（个人内标准化）")
    base = fit_baseline(args)
    print("  基线(由 calm 场景拟合): HR %.1f | SDNN %.1f | RMSSD %.1f | "
          "LF/HF %.2f | 呼吸 %.1f"
          % (base.data["hr"], base.data["sdnn"], base.data["rmssd"],
             base.data["lf_hf"], base.data["rr"]))
    print("  人群参考锚点        : HR %.1f | SDNN %.1f | RMSSD %.1f | LF/HF %.2f"
          % (67.5, 70.0, 55.0, 1.5))
    res_norm = {}
    for key in SCENARIOS:
        args.user = "check_norm_" + key
        pairs, _ = run_scenario(key, base, True, args, run_sec=a.run_sec)
        res_norm[key] = pairs
    all_norm = [p for v in res_norm.values() for p in v]
    acc_norm, cm_norm, _ = confusion(all_norm)
    print("  总体准确率: %.1f%%  (%d 个样本)" % (100 * acc_norm, len(all_norm)))
    print_confusion(cm_norm, "混淆矩阵 (行=真值, 列=识别)")

    print("\n" + "=" * 78)
    print(" 结论")
    print("=" * 78)
    print("  原始量纲          : %.1f%%" % (100 * acc_raw))
    print("  个人基线归一化    : %.1f%%   (Δ %+.1f 个百分点)"
          % (100 * acc_norm, 100 * (acc_norm - acc_raw)))
    print("\n  逐类对比(召回率):")
    _, _, rec_raw = confusion(all_raw)
    _, _, rec_norm = confusion(all_norm)
    for k in SCENARIOS:
        zh = TRUTH2EMO[k]
        print("    %-6s %5.0f%% → %5.0f%%" % (zh, 100 * rec_raw.get(zh, 0),
                                              100 * rec_norm.get(zh, 0)))
    print("\n  说明: 上述为**同源算法链路自洽性**验收, 非人体验证。")
    print("       真实指标须由 20 人以上被试实验给出（计划书 §5）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
