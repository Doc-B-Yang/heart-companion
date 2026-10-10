# -*- coding: utf-8 -*-
"""
真实数据训练管线 —— PAD 三维回归模型（PCA-SVR）
================================================
对应文献2《基于HRV特征和PCA-SVR的PAD维度情感预测分析》。

**为什么需要这个脚本**

``algorithms/pad_model.py`` 里的 ``PADRegressor`` 只负责"训练好之后怎么用"，
但**没有任何地方生产这个模型** —— 没有训练脚本，PAD 就永远只能走 RulePAD
规则映射，文献2 的 PCA-SVR 结论就落不了地。本脚本补上这一环。

**为什么不能直接把文献2 的模型拿来用**

文献2 的训练集是它自己的被试（特定采集条件、特定 HRV 提取流程）。
本项目用的是 MAX30102 指腹/腕上采集 + 自己的特征口径，特征分布不同，
直接迁移等于用别人的基线判断你的用户。因此需要**用自己的数据重训**。
个人基线相对化是本项目需要检验的设计选择；仿真准确率不证明真实效果。
文献2 使用压力脉搏传感器，本项目使用光电 PPG，不能直接迁移其指标。

**三种数据源**

1. ``--data pad.csv``  外部真实数据集（推荐，验收用）
2. ``--from-db``       本项目运行期数据（``data/memory.db``）
   ⚠️ 只能给出**代理标签**（由主观自评分数反推），仅供探索，
   **不得**用于验收报告 —— 脚本会强制要求 ``--allow-proxy`` 才继续。
3. ``--synthetic``     合成数据，仅用于验证流程跑得通（不产生可用模型）

**CSV 格式约定**（列名大小写不敏感，UTF-8 或 UTF-8-BOM）

    特征列（任意子集，缺失项由标准 HRV 关系式补齐，并在报告中列出）:
        mean_ibi_ms, sdnn_ms, hr_bpm, sd_hr_bpm, rmssd_ms, nn50,
        pnn50_pct, sdnn_index_ms, vlf_ms2, lf_ms2, hf_ms2, lf_nu,
        hf_nu, lf_hf, sd1_ms, sd2_ms, sampen, dfa_alpha1, dfa_alpha2

    标签列（二选一）:
      a) P / A / D        —— 直接给三维值（−4 ~ +4）
      b) v1 ~ v12         —— PAD 情绪量表 12 项原始分
                             （脚本按文献2 式(1) 折算成 P/A/D，
                               含 6 道反向题，不要手工算）

**输出**

    models/pad_svr.joblib            ← 系统实际加载路径（config.PAD_MODEL_PATH）
    models/pad_svr.<时间戳>.joblib   ← 旧模型自动备份
    models/pad_training_data.csv     ← 本次训练数据快照（可复现）

**用法**

    python models/train_pad_model.py --synthetic              # 流程自检
    python models/train_pad_model.py --data data/pad.csv      # 正式训练
    python models/train_pad_model.py --data data/pad.csv --cv 10 --out models/pad_svr.joblib
    python models/train_pad_model.py --from-db --allow-proxy  # 探索性（代理标签）
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 使 hrv_companion_agent 包可导入（无论从何处启动）
_PKG_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PKG_ROOT))

from hrv_companion_agent import config  # noqa: E402
from hrv_companion_agent.algorithms.pad_model import (  # noqa: E402
    PAD_FEATURE_NAMES, PADRegressor, pad_from_questionnaire)

# ---------------------------------------------------------------- 常量 ----
#: 文献2 按2:1划分训练/测试集的留出测试 CCC；不是分类准确率或交叉验证结果。
PAPER_REFERENCE_CCC = {"P": 0.78, "A": 0.46, "D": 0.30}

#: 归一化列名 → 内部特征名
_COL_ALIASES = {
    "mean_ibi": "mean_ibi_ms", "ibi": "mean_ibi_ms", "mean_ibi_ms": "mean_ibi_ms",
    "sdnn": "sdnn_ms", "sdnn_ms": "sdnn_ms",
    "hr": "hr_bpm", "heart_rate": "hr_bpm", "hr_bpm": "hr_bpm",
    "sd_hr": "sd_hr_bpm", "sd_hr_bpm": "sd_hr_bpm",
    "rmssd": "rmssd_ms", "rmssd_ms": "rmssd_ms",
    "nn50": "nn50", "nn50_pct": "nn50",
    "pnn50": "pnn50_pct", "pnn50_pct": "pnn50_pct",
    "sdnn_index": "sdnn_index_ms", "sdnn_index_ms": "sdnn_index_ms",
    "vlf_ms2": "vlf_ms2", "lf_ms2": "lf_ms2", "hf_ms2": "hf_ms2",
    "lf_nu": "lf_nu", "hf_nu": "hf_nu", "lf_hf": "lf_hf",
    "sd1": "sd1_ms", "sd1_ms": "sd1_ms", "sd2": "sd2_ms", "sd2_ms": "sd2_ms",
    "sampen": "sampen", "sample_entropy": "sampen",
    "dfa_alpha1": "dfa_alpha1", "dfa1": "dfa_alpha1",
    "dfa_alpha2": "dfa_alpha2", "dfa2": "dfa_alpha2",
}

#: 缺失特征的推导规则（由基础量按标准 HRV 关系式补齐）
_DERIVE_COLUMNS = ("mean_ibi_ms", "sd1_ms", "sd2_ms", "pnn50_pct")


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("-", "_")


def _f(v) -> Optional[float]:
    """宽松转 float（空串 / N/A / 非数字 → None）"""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip()
    if not s or s.upper() in ("NA", "N/A", "NAN", "NULL", "-", "—"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ------------------------------------------------------------ 特征补齐 ----
def fill_derived(feats: Dict[str, float]) -> List[str]:
    """按标准 HRV 关系式补齐缺失的衍生特征

    推导关系（与 train_emotion_model.py 保持一致，避免两个训练脚本
    对同一份数据算出两套特征）::

        mean_IBI = 60000 / HR
        SD1      = RMSSD / √2
        SD2      = √(2·SDNN² − SD1²)          （要求根号内为正）
        pNN50    ≈ 100·(1 − e^(−0.0129·RMSSD)) （经验拟合，非精确等式）

    Returns:
        被补齐的列名列表（供报告"哪些是推导来的"）
    """
    filled: List[str] = []
    hr = feats.get("hr_bpm")
    if not feats.get("mean_ibi_ms") and hr:
        feats["mean_ibi_ms"] = 60000.0 / hr
        filled.append("mean_ibi_ms")
    rmssd = feats.get("rmssd_ms")
    if rmssd:
        if not feats.get("sd1_ms"):
            feats["sd1_ms"] = rmssd / math.sqrt(2.0)
            filled.append("sd1_ms")
        if not feats.get("pnn50_pct"):
            feats["pnn50_pct"] = 100.0 * (1.0 - math.exp(-0.0129 * rmssd))
            filled.append("pnn50_pct")
    sdnn = feats.get("sdnn_ms")
    sd1 = feats.get("sd1_ms")
    if sdnn and sd1 and not feats.get("sd2_ms"):
        inside = 2.0 * sdnn * sdnn - sd1 * sd1
        if inside > 0:
            feats["sd2_ms"] = math.sqrt(inside)
            filled.append("sd2_ms")
    return filled


# ------------------------------------------------------------ 数据加载 ----
#: 三个数据源的统一返回: (特征矩阵, 标签矩阵, 说明行列表)
Dataset = Tuple[List[List[float]], List[List[float]], List[str]]


def load_csv(path: str) -> Dataset:
    """读 CSV → (X, y, notes)

    实现要点（v1 草稿曾在这里出过错，故显式写清）：
      特征与标签必须**在同一轮循环里成对产出**。若先收集所有特征行、
       再用 ``rows[:len(X)]`` 去取标签，一旦中间跳过若干行，
       X 与 y 就会整体错位 —— 而错位后的训练**不会报错**，
       只会静默地学出一堆噪声。这类 bug 在指标上看不出来，
       所以必须从结构上避免。
    """
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"❌ 文件为空: {path}")

    header = {_norm(k) for k in rows[0].keys()}
    has_pad = {"p", "a", "d"} <= header
    has_scale = {"v%d" % i for i in range(1, 13)} <= header
    if not (has_pad or has_scale):
        raise SystemExit(
            "❌ 缺少标签列。需要 P/A/D 三列，或 v1~v12 十二列（PAD 情绪量表）。\n"
            f"   实际读到: {sorted(header)}")
    label_src = "P/A/D 直接标签" if has_pad else "v1~v12 量表折算（文献2 式(1)）"

    X: List[List[float]] = []
    y: List[List[float]] = []
    skipped_feat = 0
    skipped_label = 0
    covered: Dict[str, int] = {n: 0 for n in PAD_FEATURE_NAMES}

    for r in rows:
        norm = {_norm(k): v for k, v in r.items()}
        # ---- 标签（先算，缺就不必处理特征）----
        if has_pad:
            p, a, d = _f(norm.get("p")), _f(norm.get("a")), _f(norm.get("d"))
            if p is None or a is None or d is None:
                skipped_label += 1
                continue
            label = [p, a, d]
        else:
            items = [_f(norm.get("v%d" % i)) for i in range(1, 13)]
            if any(v is None for v in items):
                skipped_label += 1
                continue
            pad = pad_from_questionnaire(items)  # type: ignore[arg-type]
            label = [pad["P"], pad["A"], pad["D"]]
        # ---- 特征（与标签同轮产出，保证对齐）----
        feats: Dict[str, float] = {}
        for col, val in norm.items():
            name = _COL_ALIASES.get(col)
            if not name:
                continue
            fv = _f(val)
            if fv is not None:
                feats[name] = fv
        fill_derived(feats)
        present = [n for n in PAD_FEATURE_NAMES if feats.get(n)]
        if len(present) < 3:
            skipped_feat += 1
            continue
        for n in present:
            covered[n] += 1
        X.append([float(feats.get(n, 0.0) or 0.0) for n in PAD_FEATURE_NAMES])
        y.append(label)

    if len(X) < 10:
        raise SystemExit(
            f"❌ 有效样本仅 {len(X)} 条，不足以训练（建议 ≥100）。\n"
            f"   跳过原因统计: 标签缺失 {skipped_label} 行 / 特征不足 {skipped_feat} 行")
    if skipped_label or skipped_feat:
        print(f"  ⚠️ 跳过 {skipped_label} 行（标签不全）+ {skipped_feat} 行"
              f"（有效特征 < 3 个）")

    # 特征覆盖率（让人一眼看出哪些列是推导出来的）
    derived = [n for n in PAD_FEATURE_NAMES if covered[n] == 0]
    notes = [f"源码: {path}",
             f"标签: {label_src}",
             f"有效样本: {len(X)}（原始 {len(rows)} 行）",
             "完全缺失的特征（训练中恒为 0，等价于不使用该特征）: "
             + (", ".join(derived) if derived else "无")]
    return X, y, notes


def load_from_db(db_path: str, allow_proxy: bool) -> Dataset:
    """从本项目运行期数据库构造训练集（**代理标签**）

    ⚠️ 这里的分工必须说清楚：
      * ``emotion_timeline`` 提供**特征**（真实采集的 HRV）；
      * ``self_report`` 提供**主观分数**（用户自己的 0~10 不适程度）。

    但 PAD 需要三维标签，而项目内的自评只采集了一个"不适程度"维度。
    因此这里只能用 ``P = −(score/10 − 0.5) × 8`` 反推一个**代理 P**，
    A 与 D 无法从单维自评得到 —— 脚本给它们填 0（即"不知道"）。

    后果：用这种标签训练，模型学不到 A/D 轴，等于只拟合了 P 的一小部分。
    所以它**只能用于验证流程，不能作为验收依据**。
    """
    if not allow_proxy:
        raise SystemExit(
            "❌ --from-db 只能给出代理标签（P 由主观分数反推，A/D 未知）。\n"
            "   这种标签不足以训练可用的 PAD 模型。\n"
            "   如确认只是探索流程，请加 --allow-proxy 重跑；\n"
            "   正式训练请用 --data <csv> 提供带 P/A/D 或 v1~v12 的真实标签。")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT t.ts, t.hr_bpm, t.sdnn_ms, t.rmssd_ms, t.lf_hf,"
            "       t.jitter, s.score"
            "  FROM emotion_timeline t"
            "  JOIN self_report s ON s.user_id = t.user_id"
            "   AND ABS(s.ts - t.ts) < 5"
            " WHERE s.score IS NOT NULL").fetchall()
    except sqlite3.Error as e:
        raise SystemExit(f"❌ 读取数据库失败（表结构不符？）: {e}")
    finally:
        conn.close()

    if len(rows) < 10:
        raise SystemExit(
            f"❌ 数据库里「情绪时间线 × 主观自评」配对的样本只有 {len(rows)} 条。\n"
            "   需要先让系统跑起来，并在对话中积累主观自评（self_report_tool）。")

    X: List[List[float]] = []
    y: List[List[float]] = []
    dropped = 0
    for r in rows:
        feats = {k: r[k] for k in ("hr_bpm", "sdnn_ms", "rmssd_ms", "lf_hf")
                 if r[k]}
        fill_derived(feats)
        if sum(1 for n in PAD_FEATURE_NAMES if feats.get(n)) < 3:
            dropped += 1
            continue
        X.append([float(feats.get(n, 0.0) or 0.0) for n in PAD_FEATURE_NAMES])
        p = -(float(r["score"]) / 10.0 - 0.5) * 8.0
        y.append([max(-4.0, min(4.0, p)), 0.0, 0.0])
    if len(X) < 10:
        raise SystemExit(f"❌ 可用样本不足（{len(X)} 条）。")

    notes = [f"源码: {db_path}",
             "标签: **代理**（P = 主观分数反推；A/D = 0，表示未知）",
             f"有效样本: {len(X)}（配对 {len(rows)} 条，丢弃 {dropped} 条特征不足）",
             "⚠️ 代理标签下模型学不到 A/D 轴，仅验证流程，不可作为验收依据"]
    return X, y, notes


def make_synthetic(n: int = 240, seed: int = 42) -> Dataset:
    """合成数据（仅用于验证流程可跑通）

    生成方式是"先造 PAD，再按生理学方向造 HRV"，因此 P/A 两轴模型必然学得会 ——
    这**只能**说明代码路径没坏，**不能**说明方法有效。

    D 轴刻意不接任何生理特征（真实数据里 D 也最弱，文献2 只有 ρc=0.30），
    它在这里充当**阴性对照**：既然特征里根本没有 D 的信息，
    交叉验证就应该给出 ρc(D) ≈ 0 —— 若跑出明显大于 0 的值，
    说明 K 折切分或预处理把训练信息漏进了验证集（数据泄漏），必须先修管线。
    """
    import random
    rng = random.Random(seed)
    X: List[List[float]] = []
    y: List[List[float]] = []
    for _ in range(n):
        P, A, D = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-3, 3))
        feats = {
            "hr_bpm": max(45.0, 70 + 12 * A + rng.gauss(0, 3)),
            "rmssd_ms": max(5.0, 45 - 8 * P + rng.gauss(0, 4)),
            "sdnn_ms": max(5.0, 50 - 8 * P + rng.gauss(0, 5)),
            "lf_hf": max(0.1, 1.5 + 0.5 * A - 0.2 * P + rng.gauss(0, 0.2)),
            "sd_hr_bpm": abs(rng.gauss(3, 1)),
            "sampen": 1.5 - 0.05 * P + rng.gauss(0, 0.1),
            "dfa_alpha1": 1.0 + 0.03 * P + rng.gauss(0, 0.05),
            "dfa_alpha2": 1.0 + rng.gauss(0, 0.05),   # 注意：不含 D，故意的
        }
        fill_derived(feats)
        X.append([float(feats.get(nm, 0.0) or 0.0) for nm in PAD_FEATURE_NAMES])
        y.append([P, A, D])
    notes = [f"源码: 合成数据（seed={seed}, n={n}）",
             "标签: P/A 由生理方向反推 → 必然学得会，只能证明代码可跑",
             "D 轴: 特征中**不含任何 D 信息**，作为阴性对照 —— "
             "预期 CV ρc(D) ≈ 0；若明显大于 0 说明交叉验证存在数据泄漏",
             "⚠️ 不可用于任何效果声明"]
    return X, y, notes


# ---------------------------------------------------------------- 报告 ----
def print_report(res: Dict, tag: str, notes: List[str]):
    print("\n" + "=" * 72)
    print(f"训练结果（{tag}）")
    print("=" * 72)
    for n in notes:
        print(f"  · {n}")
    print(f"  样本数 {res['n_samples']} | PCA {'启用' if res['use_pca'] else '关闭'}"
          f" | 指标口径: {'交叉验证' if res['ccc_source'] == 'cv' else '样本内（偏乐观）'}")
    print(f"\n  {'轴':<4}{'MSE':>10}{'ρ':>10}{'ρc(CCC)':>12}"
          f"{'文献2 ρc':>12}{'CV ρc':>10}")
    for axis in ("P", "A", "D"):
        m = res["metrics"][axis]
        cv = (res.get("cv_metrics") or {}).get(axis, {})
        cvv = cv.get("ccc") if cv else None
        print(f"  {axis:<4}{m['mse']:>10.4f}{m['rho']:>10.4f}{m['ccc']:>12.4f}"
              f"{PAPER_REFERENCE_CCC[axis]:>12.2f}"
              f"{(f'{cvv:.4f}' if cvv is not None else '—'):>10}")
    print(f"\n  平均 ρc = {res['mean_ccc']:.4f}（口径: {res['ccc_source']}）")
    print("  说明: 文献2 的ρc来自2:1留出测试，未声明交叉验证；CCC不是分类准确率。"
          "上表CV与论文划分口径不同，仅供方法参考；样本内指标也不能证明泛化效果。")


def main():
    ap = argparse.ArgumentParser(
        description="训练 PAD 三维回归模型（PCA-SVR，文献2）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", metavar="CSV", help="外部数据集（推荐）")
    src.add_argument("--from-db", action="store_true",
                     help="用本项目 memory.db（仅代理标签，须配 --allow-proxy）")
    src.add_argument("--synthetic", action="store_true",
                     help="合成数据（仅验证流程可跑通）")
    ap.add_argument("--db", default=str(config.MEMORY_DB_PATH),
                    help=f"memory.db 路径（默认 {config.MEMORY_DB_PATH}）")
    ap.add_argument("--out", default=str(config.PAD_MODEL_PATH),
                    help=f"模型输出路径（默认 {config.PAD_MODEL_PATH}）")
    ap.add_argument("--cv", type=int, default=5, help="交叉验证折数（默认 5）")
    ap.add_argument("--pca", choices=("auto", "on", "off"), default="auto",
                    help="是否启用 PCA（auto=按样本量，文献2 证明小样本下 PCA 掉分）")
    ap.add_argument("--allow-proxy", action="store_true",
                    help="允许使用代理标签（仅 --from-db 时有效）")
    ap.add_argument("--no-save", action="store_true", help="只训练不保存")
    args = ap.parse_args()

    print("=" * 72)
    print("PAD 三维回归模型训练（PCA-SVR · 文献2）")
    print("=" * 72)

    # 合成数据只是代码路径自检：模型与数据快照**都不落盘**。
    # 否则 models/ 里会出现一堆看起来像真实训练产物、实际毫无意义的文件。
    if args.synthetic:
        args.no_save = True
        print("\n  ⚠️ 合成数据仅用于验证代码路径可运行，训练出的模型没有任何实用价值。")
        print("     本次不保存模型、也不保存数据快照。要产出可用模型请用 --data 指向真实数据集。")

    # ---- 加载 ----
    if args.data:
        print(f"\n[1/4] 读取数据集: {args.data}")
        X, y, notes = load_csv(args.data)
        tag = Path(args.data).name
    elif args.from_db:
        print(f"\n[1/4] 从数据库构造: {args.db}")
        X, y, notes = load_from_db(args.db, args.allow_proxy)
        tag = "memory.db（代理标签）"
    else:
        print("\n[1/4] 生成合成数据（仅流程自检）")
        X, y, notes = make_synthetic()
        tag = "合成数据"

    # ---- 数据快照（可复现）----
    if not args.synthetic:
        snap = Path(args.out).parent / "pad_training_data.csv"
        try:
            snap.parent.mkdir(parents=True, exist_ok=True)
            with open(snap, "w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(list(PAD_FEATURE_NAMES) + ["P", "A", "D"])
                for row, lab in zip(X, y):
                    w.writerow([f"{v:.4f}" for v in row] + [f"{v:.4f}" for v in lab])
            print(f"      数据快照已保存: {snap}")
        except OSError as e:
            print(f"      ⚠️ 快照保存失败（不影响训练）: {e}")

    # ---- 训练 ----
    use_pca = {"auto": None, "on": True, "off": False}[args.pca]
    print(f"\n[2/4] 训练 PCA-SVR（{len(X)} 样本 × {len(PAD_FEATURE_NAMES)} 特征）")
    reg = PADRegressor(model_path=None)
    try:
        res = reg.fit(X, y, use_pca=use_pca, cv_folds=args.cv, verbose=True)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"❌ 训练失败: {e}")

    # ---- 报告 ----
    print("\n[3/4] 指标")
    print_report(res, tag, notes)

    # ---- 保存 ----
    if args.no_save:
        print("\n[4/4] 已按要求跳过保存。")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out.with_name(f"{out.stem}.{stamp}{out.suffix}")
        try:
            shutil.copy2(out, backup)
            print(f"\n[4/4] 备份旧模型 → {backup.name}")
        except OSError as e:
            print(f"\n[4/4] ⚠️ 旧模型备份失败（继续保存）: {e}")
    else:
        print("\n[4/4] 保存模型")
    reg.save(str(out))
    print(f"      ✅ 已保存: {out}")
    print(f"      系统将自动加载（config.PAD_MODEL_PATH）；"
          f"未加载时回退 RulePAD 规则映射。")
    print("\n  提醒: 报告里请写 **CV ρc**，不要写样本内 ρc。")
    print("        并注明训练数据来源与规模 —— 没有数据来源说明的指标没有意义。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
