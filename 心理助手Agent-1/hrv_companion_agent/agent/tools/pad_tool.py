# -*- coding: utf-8 -*-
"""
PAD 维度查询工具 (pad_tool)
============================
对应文献2《基于HRV特征和PCA-SVR的PAD维度情感预测分析》。

**为什么需要这个工具（而不是只用 5 类离散情绪）**

离散标签会把两种需要**完全不同干预**的状态塞进同一格：

    高唤醒 + 低愉悦（焦虑、愤怒、恐惧）  → 需要「降温」：呼吸、着陆、躯体安抚
    低唤醒 + 低愉悦（疲劳、低落、耗竭）  → 需要「激活」：起身活动、阳光、小任务

在 5 类体系里前者叫"高焦虑"、后者常被识别成"疲劳"，但情绪基调都偏负。
PAD 三维连续量把这两者分开 —— 文献2 用 19 维 HRV 特征做 PCA-SVR 回归，
P 轴 CCC 达 0.78（三轴中最好），A 轴 0.46，D 轴最差 0.30。

**本工具同时复用文献1 的「相对特征值法」做归因**

文献2 与文献1 的共识是：绝对阈值挡不住个体差异，必须相对个体平静基线。
因此本工具输出「哪些特征相对基线偏移最多」，让 Agent 的话有据可依
（可以说"你比平时更紧绷"，而不是编造一个数字）。

⚠️ 口径边界（必须向用户守住）：
  * D 轴（优势度）在 HRV 上天然不可靠，本工具会**主动标注低置信**；
  * PAD 是「倾向」不是「结论」，用于选择干预方向，不用于给用户定性。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ... import config
from ...algorithms.hrv_extractor import HRVExtractor
from ...algorithms.pad_model import estimate_pad, pad_to_emotion
from .base_tool import BaseTool

#: 维度 → 人类可读的两端含义（用于生成自然语言描述）
DIM_DESCRIPTIONS = {
    "P": ("低落 / 不愉快", "愉悦 / 满意"),
    "A": ("平静 / 低唤醒", "兴奋 / 高唤醒"),
    "D": ("被支配 / 无力", "主控 / 有掌控感"),
}

#: 相对特征 → 自然语言（供归因描述；不含数值）
FEATURE_PHRASE = {
    "hr_bpm": "心率",
    "rmssd_ms": "副交感张力（RMSSD）",
    "sdnn_ms": "整体变异性（SDNN）",
    "pnn50": "逐拍变异比例（pNN50）",
    "lf_hf": "交感-副交感平衡（LF/HF）",
    "sd_hr_bpm": "心率波动幅度",
    "sample_entropy": "信号复杂度",
    "dfa_alpha1": "短程分形标度",
}


def _band_phrase(value: float, low: float, high: float) -> str:
    """把 [-4,4] 的连续值转成三档语义"""
    if value <= low:
        return "明显偏低"
    if value >= high:
        return "明显偏高"
    return "处于中间"


class PADTool(BaseTool):
    name = "pad_tool"
    description = (
        "查询用户的 PAD 三维情感倾向（愉悦度P/唤醒度A/优势度D，范围 −4~+4），"
        "用于区分「需要降温」（高唤醒+低愉悦）与「需要激活」（低唤醒+低愉悦）。"
        "调用格式: [TOOL_CALL:pad_tool:dim=all] 或 [TOOL_CALL:pad_tool:why=1] "
        "参数 dim 可选 P/A/D/all（默认 all）；why=1 时额外给出驱动该结果的相对特征归因。"
        "返回含置信度 —— **置信度低时不要据此下判断**。")

    def __init__(self, state_provider: Callable[[], Optional[Dict]] = None,
                 baseline_provider: Callable[[], Optional[Dict]] = None,
                 regressor=None):
        """
        Args:
            state_provider: 无参回调 → 当前情绪/生理状态 dict（含 pad 或 feats）
            baseline_provider: 无参回调 → 个体平静期基线（用于相对化与规则映射）
            regressor: 已训练的 PADRegressor（可选；不可用时自动走 RulePAD）
        """
        self._state_provider = state_provider
        self._baseline_provider = baseline_provider
        self._regressor = regressor

    # ------------------------------------------------------------------
    def _get_state(self) -> Dict:
        if not self._state_provider:
            return {}
        try:
            return self._state_provider() or {}
        except Exception:  # noqa: BLE001 工具不应把异常抛给对话主循环
            return {}

    def _get_baseline(self) -> Optional[Dict]:
        if not self._baseline_provider:
            return None
        try:
            return self._baseline_provider() or None
        except Exception:  # noqa: BLE001
            return None

    def _resolve_pad(self, state: Dict) -> Dict:
        """优先复用 system 已算好的 PAD；没有则就地用规则模型补算"""
        if state.get('simulation') is False:
            return {}  # No independently validated hardware PAD regression.
        pad = state.get("pad")
        if pad and pad.get("P") is not None:
            return dict(pad)
        feats = state.get("feats") or {}
        if not feats:
            return {}
        baseline = self._get_baseline()
        result = estimate_pad(feats, baseline, self._regressor)
        return result.to_dict()

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        dim = str(params.get("dim") or params.get("input") or "all").strip().upper()
        want_why = str(params.get("why") or "").strip().lower() in ("1", "true", "yes")
        if dim not in ("P", "A", "D", "ALL"):
            dim = "ALL"

        state = self._get_state()
        if state.get('simulation') is False:
            return '当前硬件方案没有经独立标签验证的PAD模型，不能估计愉悦度或支配度；请以用户自述为准。'
        pad = self._resolve_pad(state)
        if not pad:
            return ("当前没有可用的生理特征，无法估计 PAD 维度。"
                    "请先确保传感器数据在积累（或直接依据用户自述陪伴）。")

        P = float(pad.get("P", 0.0))
        A = float(pad.get("A", 0.0))
        D = float(pad.get("D", 0.0))
        conf = float(pad.get("confidence", 0.0))
        d_conf = float(pad.get("d_confidence", 0.0))
        method = pad.get("method", "rule")

        lines = ["【PAD 维度倾向】（−4 ~ +4，0 = 个体基线附近）"]

        def _line(key: str, value: float, c: float) -> str:
            lo, hi = DIM_DESCRIPTIONS[key]
            side = hi if value >= 0 else lo
            return f"  {key}（{side}）: {value:+.2f}  {_band_phrase(value, -1.0, 1.0)}"

        if dim in ("P", "ALL"):
            lines.append(_line("P", P, conf))
        if dim in ("A", "ALL"):
            lines.append(_line("A", A, conf))
        if dim in ("D", "ALL"):
            lines.append(_line("D", D, d_conf))
            if d_conf < config.PAD_MIN_CONFIDENCE:
                lines.append("    ⚠️ D 轴置信度低（文献2 实测 ρc 仅 0.30），"
                             "**不要**据此判断用户“有没有掌控感”。")

        hint = pad.get("emotion_hint") or pad_to_emotion(P, A, D)
        lines.append(f"  综合倾向: 偏「{hint}」"
                     f"（方法: {method}，整体置信度 {conf:.2f}）")

        # ---- 干预方向建议（本工具的核心价值）----
        if A >= 0.8 and P <= 0:
            lines.append("  → **干预方向：降温**。用户处于高唤醒+不愉悦，"
                         "优先躯体安抚（延长呼气、生理性叹息、着陆技术），"
                         "**不要**在此刻讲道理或要求ta做复杂规划。")
        elif A <= -0.5 and P <= 0:
            lines.append("  → **干预方向：温和激活**。用户处于低唤醒+不愉悦，"
                         "适合极低门槛的小行动（起身喝水、晒太阳、伸展 2 分钟）、"
                         "以及不带压力的陪伴；避免提出“你应该振作”这类要求。")
        elif P >= 0.8:
            lines.append("  → **干预方向：巩固与放大**。愉悦度偏高，"
                         "适合做积极体验放大与资源记录"
                         "（“今天有什么让你觉得不错”）。")
        else:
            lines.append("  → **干预方向：维持陪伴**。各维度接近基线，"
                         "以倾听和开放式提问为主。")

        if conf < config.PAD_MIN_CONFIDENCE:
            lines.append("  ⚠️ 整体置信度偏低，PAD 仅作语气参考，"
                         "**以用户自述为准**。")

        if want_why:
            why = self._explain(state)
            if why:
                lines.append("")
                lines.append("【归因：相对个体平静基线的偏移（文献1 相对特征值法）】")
                lines.extend(why)
                lines.append("  说明: 以上仅描述「与平时相比的变化方向」，"
                             "请用自然语言表达（如“比平时更紧绷”），"
                             "**不要**向用户复述数值或特征名。")
        lines.append("")
        lines.append("表达纪律: 不要说“你的愉悦度是−1.2”，"
                     "只用来决定你说话的**方向与力道**。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _explain(self, state: Dict, top_n: int = 3) -> list:
        """相对基线的偏移归因（复用文献1 的相对特征值法）"""
        feats = state.get("feats") or {}
        baseline = self._get_baseline()
        if not feats or not baseline:
            return ["  （缺少个体基线，无法归因 —— 建议先完成一次静息基线采集）"]
        rel = HRVExtractor.relative_features(feats, baseline)
        pairs = []
        # relative_features 只返回真正完成相对化的键，因此这里**不需要**
        # 再按 RELATIVE_KEYS 或 feats 过滤 —— 缺基线的特征不会出现在 rel 中，
        # 也就不会被误当成"偏移量"（原始量纲混进比值空间曾导致
        # sampen=1.5 被读成"显著低于基线"并占据归因首位）。
        for k in HRVExtractor.RELATIVE_KEYS:
            if k not in rel:
                continue
            try:
                v = float(rel[k])
            except (TypeError, ValueError):
                continue
            if abs(v) < 1e-6:
                continue
            pairs.append((abs(v), k, v))

        if not pairs:
            return ["  各特征与基线基本一致（差异均在噪声范围内）。"]
        pairs.sort(reverse=True)
        lines = []
        for _, key, v in pairs[:top_n]:
            phrase = FEATURE_PHRASE.get(key, key)
            direction = "低于" if v > 0 else "高于"
            magnitude = "显著" if abs(v) >= 0.5 else ("轻度" if abs(v) >= 0.2 else "轻微")
            lines.append(f"  · {phrase}：{magnitude}{direction}平时基线")
        return lines
