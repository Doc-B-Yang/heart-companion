# -*- coding: utf-8 -*-
"""
主动触发决策模块
================
对应 PPT STEP 04 / 计划书 §4.2 算法子模块 3。

输入: 情绪识别结果 + 历史风险趋势 + 个体基线 + 距上次触发时间
输出: TriggerDecision{should_trigger, trigger_level, strategy, message, risk}

风险评分 R ∈ [0,100] 综合:
  1. 当前情绪风险（情绪基础分 × 生理偏离）
  2. 恶化趋势（最近 N 个样本风险斜率）
  3. 持续时长（连续高风险样本数）
  4. 个体基线偏离（与用户平静期基线比较）
  5. 冷却时间（距上次触发越久，容忍度越高 → 服务找人而非打扰）

触发分级（计划书 §4.2）:
  R>60 → low    轻微问候   |  R>80 → medium 主动对话
  R>90 → high   紧急关怀   |  R≥97 → emergency 危机干预协议
另含"陪伴式低频问候": 长时间无交互（默认2小时）时主动问候，体现主动性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config


@dataclass
class TriggerDecision:
    should_trigger: bool
    trigger_level: str        # none/low/medium/high/emergency/quiet_checkin
    strategy: str             # 干预策略名称（见 emotion_strategy.STRATEGIES）
    message: str              # 触发开场白模板（Agent 据此生成关怀）
    risk: float               # 综合风险评分
    reason: str = ""          # 决策依据（可解释性）
    detail: Dict = field(default_factory=dict)


# 触发等级 → 开场白微调（按当前情绪语气适配；未列出则用默认开场白）
EMOTION_OPENER_TWEAKS: Dict[str, Dict[str, str]] = {
    "疲劳": {
        "low": "感觉你今天有点累呢，先歇口气，还好吗？",
        "quiet_checkin": "好久没和你聊天啦，最近休息得怎么样？",
    },
    "低焦虑": {
        "low": "感觉你今天有点紧绷呢，是有什么事压在心上吗？",
    },
}


def _opener_for(level: str, emotion: str, default: str) -> str:
    """情绪适配开场白（默认开场白兜底）"""
    return EMOTION_OPENER_TWEAKS.get(emotion, {}).get(level, default)


class TriggerEngine:
    """主动触发决策引擎（含自适应个体阈值）"""

    def __init__(self,
                 low: float = None, medium: float = None,
                 high: float = None, emergency: float = None,
                 cooldown_sec: float = None, quiet_checkin_sec: float = None):
        self.low = low if low is not None else config.TRIGGER_LOW
        self.medium = medium if medium is not None else config.TRIGGER_MEDIUM
        self.high = high if high is not None else config.TRIGGER_HIGH
        self.emergency = (emergency if emergency is not None
                          else config.TRIGGER_EMERGENCY)
        self.cooldown_sec = (cooldown_sec if cooldown_sec is not None
                             else config.TRIGGER_COOLDOWN_SEC)
        self.quiet_checkin_sec = (quiet_checkin_sec if quiet_checkin_sec is not None
                                  else config.QUIET_CHECKIN_SEC)

        # 自适应个体阈值（STEP 06 反馈优化：随用户反馈微调）
        self.personal_offsets: Dict[str, float] = {}
        # 策略效果统计 {strategy: {"effective": n, "total": n}}
        self.strategy_stats: Dict[str, Dict[str, int]] = {}
        # v2: 连续满足触发条件的次数（抗单点误报，见 TRIGGER_MIN_CONSECUTIVE）
        self._consecutive_pass: int = 0
        # v2: 最近一次被质量门禁拦截的原因（便于诊断"为什么不触发"）
        self.last_gate_reason: str = ""

    # ------------------------------------------------------------ 风险计算
    def composite_risk(self, emotion_risk: float,
                       risk_history: List[float],
                       affect: Optional[Dict] = None) -> float:
        """
        综合风险 = 当前情绪风险 + 恶化趋势 + 持续时长 + 心境底色 + 情绪强度
                  （加法模型）

        说明: 当前情绪风险是主导项；趋势与持续项只做 0~15/0~10 的加成，
        保证 R ≥ 情绪风险分，与计划书 §4.2 的阈值语义一致。

        v2 新增两个**慢变量**项（依据文献3 的情感动力学思想）：
          * **心境底色**：瞬时风险一样，但"今天整体偏低落"的日子应当
            更容易被关怀。文献3 的心境累积 Mt 正是为此设计。
            仅当心境为**消极**时加分（积极心境不加分，避免"心情好反而
            风险高"的荒谬结论），上限 ``config.TRIGGER_MOOD_WEIGHT``。
          * **情绪强度**：同样是"低焦虑"，Et=0.2 与 Et=0.9 不应等量齐观。
            上限 ``config.TRIGGER_INTENSITY_WEIGHT``。
        """
        base = float(emotion_risk)

        trend = 0.0
        window = risk_history[-config.RISK_TREND_WINDOW:]
        if len(window) >= 3:
            # 简单线性斜率（最小二乘）
            m = len(window)
            x_mean = (m - 1) / 2.0
            y_mean = sum(window) / m
            num = sum((i - x_mean) * (window[i] - y_mean) for i in range(m))
            den = sum((i - x_mean) ** 2 for i in range(m))
            if den > 1e-9:
                slope = num / den                     # 每样本风险增量
                trend = max(0.0, min(15.0, slope * 1.5))

        persistence = 0.0
        if window:
            # 持续高风险 = 连续处于"紧急关怀"级以上的样本数
            high_count = sum(1 for r in window if r >= self.high)
            persistence = min(10.0, high_count * 2.0)

        # ---- v2: 心境底色（仅消极侧加分）----
        mood_bonus = 0.0
        intensity_bonus = 0.0
        if affect:
            mood = float(affect.get("mood") or 0.0)
            if mood < 0:
                mood_bonus = min(config.TRIGGER_MOOD_WEIGHT,
                                 -mood * config.TRIGGER_MOOD_WEIGHT)
            # 情绪强度：只有**负性**主导情绪的高强度才加分
            # （避免"愉悦强度高 → 风险高"这种方向性错误）
            dom = affect.get("dominant") or ""
            if dom and dom not in ("joy", "trust", "anticipation"):
                intensity = float(affect.get("dominant_intensity") or 0.0)
                intensity_bonus = min(config.TRIGGER_INTENSITY_WEIGHT,
                                      intensity * config.TRIGGER_INTENSITY_WEIGHT)

        r = base + trend + persistence + mood_bonus + intensity_bonus
        return round(min(100.0, max(0.0, r)), 1)

    # ------------------------------------------------------------ 触发决策
    def decide(self, emotion_result, risk_history: List[float],
               baseline: Optional[Dict[str, float]] = None,
               last_trigger_time: Optional[float] = None,
               last_interaction_time: Optional[float] = None,
               now: Optional[float] = None,
               affect: Optional[Dict] = None,
               quality: Optional[Dict] = None) -> TriggerDecision:
        """
        Args:
            emotion_result:  EmotionResult（含 emotion/confidence/risk_score）
            risk_history:    最近若干次 risk_score 列表（不含当前）
            baseline:        用户平静期基线 {hr_bpm, sdnn_ms, ...}（可选）
            last_trigger_time:   上次主动触发时间戳（可选）
            last_interaction_time: 上次用户交互时间戳（可选）
            now:             当前时间戳（测试注入用）
            affect:          情感动力学快照（``AffectState.to_dict()``），
                             提供时心境的慢变量会参与风险计算（v2 新增）
            quality:         信号质量报告（``HRVExtractor.quality_metrics()``），
                             提供时启用质量门禁（v2 新增）

        Returns:
            TriggerDecision
        """
        now = now if now is not None else time.time()
        current_risk = emotion_result.risk_score
        r = self.composite_risk(current_risk, risk_history, affect=affect)
        self.last_gate_reason = ""

        # ---- v2 质量门禁 ----
        # 依据实测：PPG 在接触不良时会出现「HR 正常但 HRV 是假的」
        # （指腹实测 RMSSD 130ms / jitter 0.455）。若在此条件下触发关怀，
        # 等于用一个噪声数字去打扰用户 —— 既不可靠也伤体验。
        # 例外：极高分（≥emergency 阈值）仍放行，宁可多关怀一次。
        if config.QUALITY_GATE_ENABLED and quality:
            q_ok = bool(quality.get("quality_ok", True))
            if not q_ok and r < self.emergency:
                self.last_gate_reason = (
                    f"信号质量不达标（{quality.get('quality_level', '?')}，"
                    f"抖动指数 {quality.get('jitter', 0):.3f}）→ 暂不触发")
                return TriggerDecision(
                    False, "none", "", "", r, self.last_gate_reason,
                    {"raw_risk": current_risk, "gate": "quality",
                     "quality": quality})

        # 个体基线偏离修正（应对 CH.02 个体差异挑战）
        if baseline and baseline.get("hr_bpm"):
            hr = float(getattr(emotion_result, "detail", {}).get("hr_bpm") or 0)
            if hr > 0:
                dev = abs(hr - baseline["hr_bpm"]) / baseline["hr_bpm"]
                r = round(min(100.0, r + max(0.0, dev - 0.15) * 15.0), 1)

        # 自适应个体阈值（每次反馈后微调，初始为 0）
        offset = self.personal_offsets.get("trigger", 0.0)
        low_t, med_t, high_t, emg_t = (self.low + offset, self.medium + offset,
                                       self.high + offset, self.emergency + offset)

        # ---- 危机干预协议（最高优先级，不受冷却限制） ----
        if r >= emg_t or r >= 99.0:
            hotline = config.CRISIS_HOTLINES[0] if config.CRISIS_HOTLINES else ("", "")
            emergency_msg = (
                "我注意到你现在可能非常难受。别担心，我在。我们先停下来，慢慢说，好吗？\n"
                f"如果此刻觉得撑不住，请一定先保证自己的安全——可以拨打"
                f"{hotline[0]} {hotline[1]}，或告诉身边信任的人。我一直在这里陪你。")
            return TriggerDecision(
                True, "emergency", "crisis_intervention",
                emergency_msg,
                r, "风险评分达到危机阈值", {"raw_risk": current_risk})

        # ---- 冷却时间检查 ----
        if last_trigger_time is not None:
            elapsed = now - last_trigger_time
            if elapsed < self.cooldown_sec:
                return TriggerDecision(False, "none", "", "", r,
                                       f"冷却中（剩余 {int(self.cooldown_sec - elapsed)}s）",
                                       {"raw_risk": current_risk})

        # ---- 陪伴式低频问候（长时间无交互 → 服务找人） ----
        if (last_interaction_time is not None
                and now - last_interaction_time >= self.quiet_checkin_sec
                and r < low_t):
            return TriggerDecision(
                True, "quiet_checkin", "quiet_greeting",
                _opener_for("quiet_checkin", getattr(emotion_result, "emotion", ""),
                            "好久没和你聊天啦，今天过得怎么样？"),
                r, "长时间无交互，低频陪伴问候", {"raw_risk": current_risk})

        # ---- v2 连续满足保护（抗单点误报）----
        # 单次误判就发起关怀会显著伤害体验（尤其在 PPG 信号不稳的场合），
        # 因此可要求风险连续 N 次越过阈值才触发。TRIGGER_MIN_CONSECUTIVE=1
        # 时该保护自动失效，与 v1 行为完全一致。
        need = max(1, int(config.TRIGGER_MIN_CONSECUTIVE))
        if need > 1:
            if r >= low_t:
                self._consecutive_pass += 1
            else:
                self._consecutive_pass = 0
            if r >= low_t and self._consecutive_pass < need:
                return TriggerDecision(
                    False, "none", "", "", r,
                    f"待确认（连续 {self._consecutive_pass}/{need} 次越过阈值）",
                    {"raw_risk": current_risk, "gate": "consecutive"})
        else:
            self._consecutive_pass = 0

        # ---- 分级触发 ----
        if r >= high_t:
            return TriggerDecision(
                True, "high", "breathing_guide",
                "你心跳得有点快，是紧张了吗？先别急，我们一起做几次深呼吸吧。",
                r, "风险评分达到紧急关怀阈值", {"raw_risk": current_risk})
        if r >= med_t:
            return TriggerDecision(
                True, "medium", "active_dialogue",
                "感觉你今天有点紧绷绷的，愿意和我聊聊吗？",
                r, "风险评分达到主动对话阈值", {"raw_risk": current_risk})
        if r >= low_t:
            return TriggerDecision(
                True, "low", "light_greeting",
                _opener_for("low", getattr(emotion_result, "emotion", ""),
                            "感觉你今天有点累呢，还好吗？"),
                r, "风险评分达到轻微问候阈值", {"raw_risk": current_risk})

        return TriggerDecision(False, "none", "", "", r, "风险低于触发阈值",
                               {"raw_risk": current_risk})

    # ------------------------------------------------------------ 反馈优化
    def record_feedback(self, strategy: str, effective: bool):
        """STEP 06: 记录干预效果，用于策略选择与阈值自适应"""
        stats = self.strategy_stats.setdefault(strategy, {"effective": 0, "total": 0})
        stats["total"] += 1
        if effective:
            stats["effective"] += 1
            # 策略有效 → 该用户触发阈值可略微上调（减少打扰）
            self.personal_offsets["trigger"] = self.personal_offsets.get("trigger", 0.0) + 0.5
        else:
            # 策略无效 → 阈值下调（更早介入）
            self.personal_offsets["trigger"] = self.personal_offsets.get("trigger", 0.0) - 1.0
        self.personal_offsets["trigger"] = max(-10.0, min(10.0,
                                                         self.personal_offsets["trigger"]))

    def strategy_effectiveness(self, strategy: str) -> float:
        """策略有效率（无数据时返回 0.5 中性值）"""
        stats = self.strategy_stats.get(strategy)
        if not stats or stats["total"] == 0:
            return 0.5
        return stats["effective"] / stats["total"]


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    from .emotion_recognizer import EmotionResult

    engine = TriggerEngine()
    hist = [30, 35, 40, 55, 60, 65, 70, 75]
    low = EmotionResult("低焦虑", 0.8, 72, "rule")
    high = EmotionResult("高焦虑", 0.85, 92, "rule")
    print(engine.decide(low, hist, now=1000.0))
    print(engine.decide(high, hist, now=1000.0))
    print(engine.decide(high, hist, last_trigger_time=999.0, now=1000.0))
