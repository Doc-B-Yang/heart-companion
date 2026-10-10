# -*- coding: utf-8 -*-
"""
情绪自适应策略库（v2 · 专业化增强版）
=====================================
对应 PPT INNOVATION 03（动态策略自适应机制）/ 计划书 §4.3 emotion_strategy.py。

v2 相对 v1 的三点增强
--------------------------------------------------------------------------------
1. **从「情绪名 → 策略名」升级为「情绪强度 + 心境 → 生成参数」**
   v1 只按离散情绪标签给一段风格文字，导致"有点焦虑"和"极度焦虑"得到
   完全相同的回应。v2 引入文献3（战勇胜等，计算机系统应用 2026, 35(9)）
   式(21)–(26) 的调制思想：由**情感强度 Et** 与**心境 Mt** 共同决定
   LLM 的**外部超参**（temperature / top_p / presence_penalty）。

   式(21)(22) 的直观含义是「**放大情绪对应的风格偏移**」::

       T'_i = T_i + (T_i − T_in) · E_t · f_i          # 情绪强度放大偏移
       T*_i = T'_i + (T'_i − T_in) · M_t · f_j        # 心境再叠加一层

   若情绪配置的温度高于中性（如「愉悦」0.85 > 中性 0.70），强度越大温度越高
   （更发散、更有生气）；若低于中性（如「高焦虑」0.50），强度越大温度越低
   （更聚焦、更稳定）。这个方向性与文献3 的表述一致：
   "积极↑温度=更发散，消极↓温度=更聚焦"。

2. **⚠️ 落地边界（必须写进说明书）**
   文献3 还把情感状态映射到 **Transformer 内部结构**（注意力头数、隐藏层维度、
   FFN 中间维度、层数 —— 式(27)–(33)）。本项目调用**云端 API 模型**，
   **无法**改动模型内部结构，因此只落地文献3 的"外部超参"部分（式21–26）。
   这一点在论文与答辩中须明确说明，不可含糊称"实现了 PWE 的参数化对话"。

3. **四种用户气质档位**（文献3 表2）
   ideal / indifferent / sensitive / persistent，见
   ``algorithms.emotion_dynamics.PERSONALITY_PROFILES``。

v1 的 ``STRATEGIES`` / ``EMOTION_STYLE_GUIDE`` / ``select_strategy`` /
``style_guide_for`` 接口**完全保留**，既有调用方无需改动。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .. import config

# ------------------------------------------------------------ 策略库 ----
# 每个策略: 适用情绪等级 / 干预强度 / 回复风格指引 / 推荐工具 / 开场白
STRATEGIES: Dict[str, dict] = {
    "light_greeting": {
        "name": "轻微问候",
        "levels": ["low"],
        "intensity": 1,
        "style": "轻松自然，像朋友打招呼，不给压力",
        "tools": [],
        "opener": "感觉你今天有点累呢，还好吗？",
    },
    "active_dialogue": {
        "name": "主动对话",
        "levels": ["medium"],
        "intensity": 2,
        "style": "先共情再邀请倾诉；用开放式问题引导表达",
        "tools": ["comfort_tool"],
        "opener": "感觉你今天有点紧绷绷的，愿意和我聊聊吗？",
    },
    "breathing_guide": {
        "name": "呼吸练习指导",
        "levels": ["high"],
        "intensity": 3,
        "style": "短句安抚 + 节奏明确的呼吸引导（4-4-6 或 4-7-8）",
        "tools": ["breathing_guide", "comfort_tool"],
        "opener": "你心跳得有点快，是紧张了吗？先别急，我们一起做几次深呼吸吧。",
    },
    "crisis_intervention": {
        "name": "危机干预协议",
        "levels": ["emergency"],
        "intensity": 4,
        "style": "安全第一：确认人身安全、陪伴安抚、提供热线与转介建议",
        "tools": ["crisis_tool", "breathing_guide"],
        "opener": "我注意到你现在可能非常难受。别担心，我在。我们先停下来，慢慢说，好吗？",
    },
    "quiet_greeting": {
        "name": "陪伴式低频问候",
        "levels": ["quiet_checkin"],
        "intensity": 1,
        "style": "久未联系时的自然问候，表达惦记而非监控",
        "tools": [],
        "opener": "好久没和你聊天啦，今天过得怎么样？",
    },
}

# 情绪 → 对话风格指引（注入 system prompt 的动态部分）
EMOTION_STYLE_GUIDE: Dict[str, str] = {
    "平静": "用户状态平稳。用轻松自然的语气聊天，可以聊聊日常，分享积极内容，巩固良好状态。",
    "愉悦": "用户心情好。积极回应、共情ta的快乐，强化积极体验，鼓励ta记住这份好心情。",
    "低焦虑": "用户有些紧张。先共情接纳，语气放慢，用开放式问题邀请倾诉；可轻量建议呼吸放松，不要过度严肃。",
    "高焦虑": "用户高度紧张。以安抚为第一优先级：短句、温柔、肯定感受；主动引导呼吸练习或落地技术，暂不追问压力源细节。",
    "疲劳": "用户疲惫。表达体谅，降低对话负担，建议休息与自我照顾；避免说教和密集提问。",
}


# ======================================================================
# 一、生成参数调制（文献3 式(21)–(26) 的落地）
# ======================================================================
#: 各情绪对应的"基础"生成参数（中性基线见 config.LLM_TEMP_NEUTRAL / TOPP_NEUTRAL）
#: 设计原则：
#:   * 高唤醒负性（高焦虑）→ 低温度 + 紧 top_p + 弱重复惩罚
#:     理由：需要稳定、可预测、少发散的输出，避免在用户紧张时火上浇油
#:     （presence_penalty 为负 = 抑制话题发散，保持聚焦）
#:   * 愉悦 → 较高温度 + 宽 top_p，让回应更有活力
#:   * 疲劳 → 中等偏低温度，避免让用户觉得"信息过载"
EMOTION_LLM_BASE: Dict[str, Dict[str, float]] = {
    "平静": {"temperature": 0.70, "top_p": 0.90, "presence_penalty": 0.0},
    "愉悦": {"temperature": 0.85, "top_p": 0.95, "presence_penalty": 0.1},
    "低焦虑": {"temperature": 0.65, "top_p": 0.88, "presence_penalty": 0.0},
    "高焦虑": {"temperature": 0.50, "top_p": 0.80, "presence_penalty": -0.1},
    "疲劳": {"temperature": 0.60, "top_p": 0.86, "presence_penalty": -0.05},
}

#: 情感强度/心境的放大因子（文献3 的 f_i / f_j）
#: ⚠️ 文献3 未公开 fi / fj 的取值，此处为本项目工程标定值。
INTENSITY_GAIN = 0.55      # f_i
MOOD_GAIN = 0.40           # f_j


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def modulate_llm_params(emotion: str,
                        intensity: float = 0.0,
                        mood: float = 0.0) -> Dict[str, float]:
    """情绪强度 + 心境 → LLM 外部生成参数（文献3 式(21)–(26)）

    Args:
        emotion: 5 类中文情绪
        intensity: 情感强度 Et ∈ [0, 1]（来自 EmotionDynamics）
        mood: 心境 Mt ∈ [−1, 1]（**带符号**；正值=积极心境）

    Returns:
        {"temperature": .., "top_p": .., "presence_penalty": ..}

    示例（为什么需要这一层）::

        同样判定为「高焦虑」：
          轻度(intensity=0.2) → temperature 0.48  （语气稳定但不过分拘谨）
          重度(intensity=0.9) → temperature 0.41  （非常聚焦、少发散）
        同样判定为「愉悦」：
          轻度(intensity=0.2) → temperature 0.87
          重度(intensity=0.9) → temperature 1.02  （更有活力，被钳位到上限）
    """
    base = dict(EMOTION_LLM_BASE.get(
        emotion, EMOTION_LLM_BASE["平静"]))
    t_in = float(config.LLM_TEMP_NEUTRAL)
    p_in = float(config.LLM_TOPP_NEUTRAL)
    e = _clamp(float(intensity or 0.0), 0.0, 1.0)
    m = _clamp(float(mood or 0.0), -1.0, 1.0)

    # ---- 式(21)(22) 温度：情绪强度放大偏移，心境再叠加 ----
    t1 = base["temperature"] + (base["temperature"] - t_in) * e * INTENSITY_GAIN
    t2 = t1 + (t1 - t_in) * m * MOOD_GAIN
    t_lo, t_hi = config.LLM_TEMP_RANGE
    out_t = _clamp(t2, t_lo, t_hi)

    # ---- 式(25) 核采样 top_p：同构调制 ----
    p1 = base["top_p"] + (base["top_p"] - p_in) * e * INTENSITY_GAIN
    p2 = p1 + (p1 - p_in) * m * MOOD_GAIN
    p_lo, p_hi = config.LLM_TOPP_RANGE
    out_p = _clamp(p2, p_lo, p_hi)

    # ---- 式(23)(24) 重复惩罚：防止在同一情绪下反复使用相同话术 ----
    r1 = base["presence_penalty"] * (1.0 + e * INTENSITY_GAIN)
    out_r = _clamp(r1, -0.5, 0.6)

    return {"temperature": round(out_t, 3),
            "top_p": round(out_p, 3),
            "presence_penalty": round(out_r, 3)}


def describe_affect(affect: Optional[Dict], emotion: str = "") -> str:
    """把情感动力学状态转成**自然语言**，供 Prompt 注入

    为什么不直接注入数字：模型对"强度 0.62"没有稳定语义，但对
    "你现在能感到ta的焦虑是明显而持续的，而且今天整体情绪底色偏沉"
    有准确的把握能力。文献3 也是把 Et / m / M / P(E) 四个量做语义化后
    才交给对话模型。
    """
    if not affect:
        return ""
    level_zh = {"unaware": "几乎察觉不到", "mild": "略微",
                "moderate": "明显", "intense": "强烈"}
    parts: List[str] = []
    lv = affect.get("level")
    if lv:
        parts.append(f"当前情绪强度: {level_zh.get(lv, lv)}")
    mood = float(affect.get("mood") or 0.0)
    if mood <= -0.35:
        parts.append("情绪底色: 明显偏低落（今天整体状态偏沉，回应可更耐心、"
                     "不要急于让ta振作）")
    elif mood <= -0.12:
        parts.append("情绪底色: 略偏低落")
    elif mood >= 0.35:
        parts.append("情绪底色: 明显偏积极（可以更放开地互动）")
    elif mood >= 0.12:
        parts.append("情绪底色: 略偏积极")
    else:
        parts.append("情绪底色: 中性平稳")

    durations = affect.get("durations") or {}
    dom = affect.get("dominant")
    if dom and durations.get(dom):
        d = float(durations[dom])
        if d > 20:
            parts.append("该情绪已持续较长时间，说明不是一时波动，"
                         "回应时可以温和地承认这一点（如\"感觉这份难受"
                         "好像已经跟着你一阵子了\"）")
    return "；".join(parts)


# ======================================================================
# 二、风格指引（v2 增强版，保留 v1 兼容签名）
# ======================================================================
#: 强度修饰语（按动力学级别给风格指引追加一句）
_INTENSITY_HINT: Dict[str, str] = {
    "unaware": "（当前强度很弱，正常聊天即可，不必刻意围绕情绪展开）",
    "mild": "（当前强度较轻，可轻描淡写地关心一句）",
    "moderate": "（当前强度明显，应当明确地回应情绪本身）",
    "intense": "（当前强度强烈，优先稳住情绪，暂缓探讨原因与建议）",
}


def style_guide_for(emotion: str, level: Optional[str] = None,
                    affect: Optional[Dict] = None) -> str:
    """情绪 → 风格指引（动态注入 Prompt）

    Args:
        emotion: 5 类中文情绪
        level: 动力学强度级别（unaware/mild/moderate/intense），可选
        affect: 完整动力学快照，提供时会补充"情绪底色/持续性"描述
    """
    base = EMOTION_STYLE_GUIDE.get(emotion, EMOTION_STYLE_GUIDE["平静"])
    parts = [base]
    if level:
        hint = _INTENSITY_HINT.get(level)
        if hint:
            parts.append(hint)
    extra = describe_affect(affect, emotion)
    if extra:
        parts.append(extra)
    return " ".join(parts)


def select_strategy(trigger_level: str,
                    engine=None,
                    user_preferences: Optional[dict] = None) -> dict:
    """
    策略选择（PPT STEP 04-3）:
    1. 按触发等级取候选策略
    2. 结合策略历史有效率（trigger_engine.strategy_stats）
    3. 考虑用户偏好（如"不喜欢被频繁打断"→ 降低强度）
    """
    candidates = [s for s in STRATEGIES.values() if trigger_level in s["levels"]]
    if not candidates:
        candidates = [STRATEGIES["light_greeting"]]

    def _score(s: dict) -> float:
        eff = engine.strategy_effectiveness(s["name"]) if engine else 0.5
        prefs = user_preferences or {}
        penalty = 0.0
        if prefs.get("少打扰") and s["intensity"] >= 3:
            penalty = 0.2
        return eff - penalty

    chosen = max(candidates, key=_score)
    return {**chosen, "key": next(k for k, v in STRATEGIES.items()
                                  if v["name"] == chosen["name"])}


def strategy_for_shift(previous: str, current: str) -> Tuple[str, str]:
    """情绪迁移 → 过渡建议（v2 新增）

    陪伴中最生硬的时刻是"用户情绪已经变了，AI 还在按上一轮的语气说话"。
    本函数给出**转场提示**，供 Prompt 注入，让切换显得自然。

    Returns:
        (transition_hint, advice)
    """
    if not previous or previous == current:
        return "", ""
    worsening = {("平静", "低焦虑"), ("平静", "高焦虑"), ("愉悦", "低焦虑"),
                 ("愉悦", "高焦虑"), ("低焦虑", "高焦虑")}
    improving = {("高焦虑", "低焦虑"), ("高焦虑", "平静"), ("低焦虑", "平静"),
                 ("低焦虑", "愉悦"), ("平静", "愉悦"), ("疲劳", "平静")}
    if (previous, current) in worsening:
        return (f"（注意：用户状态刚从「{previous}」转向「{current}」，"
                f"正在变差）",
                "不要突然升高紧张感，用平缓的方式自然转向安抚，"
                "避免让用户觉得被监视。")
    if (previous, current) in improving:
        return (f"（注意：用户状态刚从「{previous}」转向「{current}」，"
                f"正在好转）",
                "顺着好转的势头，肯定ta的自我调节，不要急着复盘刚才的难受。")
    return (f"（注意：用户状态从「{previous}」变为「{current}」）",
            "自然过渡即可，不要点破你察觉到了变化。")
