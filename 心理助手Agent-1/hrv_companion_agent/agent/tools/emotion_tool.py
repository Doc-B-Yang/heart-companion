# -*- coding: utf-8 -*-
"""
情绪查询工具 (emotion_tool)
============================
对应计划书 §4.3 tools/emotion_tool.py，v2 扩展为「状态全貌查询」。

**这个工具的作用不是让 Agent 复述数据，而是让 Agent 在不确定时先查一下。**

典型使用场景：
  * 用户说「我没事」，但系统感知到的状态与之不符 → 先查清楚再决定怎么接话；
  * 用户问「我现在的状态怎么样」→ 需要把生理判断转成自然语言；
  * Agent 想确认「现在是该安抚还是该激活」→ 查看 PAD 与质量。

**v2 相比 v1 多给的三类信息（都是为了让判断更稳）**

1. **信号质量**（jitter / 质量等级）—— 这是 v1 最关键的缺失。
   本项目实测已确认：指腹接触不良时会出现「HR 正常但 HRV 是假的」
   （RMSSD 130ms、pNN50 61~75%，而静息应 <50ms / <10%）。
   此时若不告诉模型"这次数据不可信"，模型会一本正经地基于噪声共情。
2. **PAD 三维** —— 区分「高唤醒的焦虑」与「低唤醒的耗竭」，
   两者的干预方向相反（降温 vs 激活）。
3. **情感动力学** —— 强度、心境底色、持续性。同样判定为「低焦虑」，
   「刚起」和「已经绷了半小时」需要不同的开场。

⚠️ 表达纪律（写入返回，约束模型）：这些数值是**给模型决定策略用的**，
   不是给用户看的。除非 config.EXPOSE_RAW_METRICS 打开，否则不得复述。
"""

from __future__ import annotations

import json
from typing import Callable, Dict, Optional

from ... import config
from .base_tool import BaseTool

EMOTION_LABELS = {
    "平静": "状态平稳放松",
    "愉悦": "心情愉快积极",
    "低焦虑": "有些紧张不安",
    "高焦虑": "高度紧张、压力明显",
    "疲劳": "疲惫、精力不足",
}

#: 信号质量等级 → 对模型的话
QUALITY_ADVICE = {
    "合格": "数据可用，可以采信生理判断。",
    "good": "数据可用，可以采信生理判断。",
    "勉强": "数据勉强可用，判断留有余地，不要说得太肯定。",
    "fair": "数据勉强可用，判断留有余地，不要说得太肯定。",
    "不可用": "**数据不可信** —— 必须完全以用户自述为准，"
              "不得依据生理数据做任何断言。",
    "poor": "**数据不可信** —— 必须完全以用户自述为准，"
            "不得依据生理数据做任何断言。",
    "不足": "采样不足，尚无有效判断，请以对话内容为主。",
}


class EmotionTool(BaseTool):
    name = "emotion_tool"
    description = (
        "查询用户当前的**完整状态**（情绪、置信度、信号质量、PAD 三维、"
        "情感动力学）。用于确认「现在该安抚还是该激活」以及"
        "「这次生理判断可不可信」。"
        "调用格式: [TOOL_CALL:emotion_tool:state] "
        "参数 mode=full(默认) / brief(仅核心)。")

    def __init__(self, state_provider: Callable[[], Optional[Dict]]):
        """
        Args:
            state_provider: 无参回调，返回当前情绪状态 dict 或 None
        """
        self._provider = state_provider

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_conf(v) -> str:
        try:
            return "%.2f" % float(v)
        except (TypeError, ValueError):
            return "N/A"

    def run(self, params: Dict) -> str:
        state = self._provider() if self._provider else None
        if not state:
            return ("当前没有可用的生理信号数据（传感器未连接或数据不足）。"
                    "请**只依据用户自述**陪伴，不要提及检测不到数据这件事。")

        if state.get('simulation') is False and not state.get('validated'):
            return ('当前生理情绪结果不确定，不能判断紧张程度或干预风险。'
                    '原因：' + str((state.get('detail') or {}).get('reason', '数据或模型尚未就绪')) +
                    '。请只依据用户自述陪伴，不要将缺少判断解释为放松或焦虑。')
        emotion = state.get("emotion") or "未识别"
        label = EMOTION_LABELS.get(emotion, "（未映射的标签）")
        risk = state.get("risk_score")
        conf = state.get("confidence")
        mode = str(params.get("mode") or "full").strip().lower()

        lines = ["【用户当前状态】",
                 "情绪: %s（%s）| 置信度 %s | 紧张评分 %s/100"
                 % (emotion, label, self._fmt_conf(conf),
                    ("%.1f" % float(risk)) if risk is not None else "N/A"),
                 "识别层: %s" % (state.get("layer") or "N/A")]

        # ---- 信号质量（最重要的一块）----
        quality = state.get("quality") or {}
        if quality:
            lv = quality.get("quality_level") or ""
            ok = quality.get("quality_ok")
            jitter = quality.get("jitter")
            lines.append("信号质量: %s%s"
                         % (lv or ("合格" if ok else "不可用"),
                            ("（jitter %s）" % jitter)
                            if jitter is not None else ""))
            advice = QUALITY_ADVICE.get(lv)
            if advice:
                lines.append("  → %s" % advice)
            elif ok is False:
                lines.append("  → **数据不可信** —— 必须完全以用户自述为准。")
            ect = quality.get("ectopic_ratio")
            if ect is not None and float(ect) > 0.1:
                lines.append("  · 伪迹比例偏高（%s），RR 序列经过较多剔除"
                             % ect)
        else:
            lines.append("信号质量: 未评估（可能尚无足够数据）")

        if mode == "brief":
            lines.append("")
            lines.append("提醒: 以上仅供你决定**说话方向**，"
                         "不要在对话里复述任何数值。")
            return "\n".join(lines)

        # ---- 情感动力学 ----
        affect = state.get("affect") or {}
        if affect:
            from ..emotion_strategy import describe_affect
            desc = describe_affect(affect)
            if desc:
                lines.append("")
                lines.append("情感动力学: %s" % desc)
            follow = affect.get("persistence") or affect.get("duration")
            if follow is not None:
                lines.append("  · 持续性: %s（用于判断「刚起」还是「已经绷很久」）"
                             % follow)

        # ---- PAD 三维 ----
        pad = state.get("pad") or {}
        if pad and pad.get("P") is not None:
            conf_pad = float(pad.get("confidence") or 0)
            lines.append("")
            lines.append("PAD 维度（−4~+4，0=个体基线附近）: "
                         "P %+.2f / A %+.2f / D %+.2f（方法 %s，置信 %.2f）"
                         % (float(pad.get("P") or 0),
                            float(pad.get("A") or 0),
                            float(pad.get("D") or 0),
                            pad.get("method") or "rule", conf_pad))
            if conf_pad < config.PAD_MIN_CONFIDENCE:
                lines.append("  → PAD 置信度偏低，仅作语气参考，"
                             "**不要**据此下判断。")
            else:
                A = float(pad.get("A") or 0)
                P = float(pad.get("P") or 0)
                if A >= 0.8 and P <= 0:
                    lines.append("  → 高唤醒 + 不愉悦 → **优先降温**"
                                 "（躯体安抚、延长呼气），不要此刻讲道理。")
                elif A <= -0.5 and P <= 0:
                    lines.append("  → 低唤醒 + 不愉悦 → **温和激活**"
                                 "（最低门槛的小行动），不要要求「振作」。")
                elif P >= 0.8:
                    lines.append("  → 愉悦度偏高 → 适合做积极体验放大")
            if float(pad.get("d_confidence") or 0) < config.PAD_MIN_CONFIDENCE:
                lines.append("  → D 轴（优势度）在 HRV 上天然不可靠"
                             "（文献2 实测 ρc 仅 0.30），不要用来判断"
                             "「用户有没有掌控感」。")

        # ---- 相对基线的偏移（文献1 相对特征值法）----
        rel = state.get("rel_features") or {}
        if rel:
            from ...algorithms.hrv_extractor import HRVExtractor
            items = []
            for k in HRVExtractor.RELATIVE_KEYS:
                v = rel.get(k)
                if v is None:
                    continue
                try:
                    items.append((abs(float(v)), k, float(v)))
                except (TypeError, ValueError):
                    continue
            items.sort(reverse=True)
            if items:
                lines.append("")
                lines.append("相对个体平静基线的最大偏移: "
                             + "、".join("%s %+.2f" % (k, v)
                                         for _, k, v in items[:3]))
                lines.append("  （正=低于基线，负=高于基线；"
                             "这是文献1 的相对特征值法，用于消除个体差异）")

        # ---- 原始指标（默认关闭）----
        if config.EXPOSE_RAW_METRICS:
            keep = {k: v for k, v in state.items()
                    if k not in ("ibi_list", "affect", "pad", "quality",
                                 "rel_features")}
            lines.append("")
            lines.append("[调试用原始值] "
                         + json.dumps(keep, ensure_ascii=False, default=str))

        lines.append("")
        lines.append("表达纪律（必须遵守）:")
        lines.append("  · 以上数值**只用来决定你说话的方向与力道**，"
                     "不要在回复里复述（不要说「你的心率是96」）;")
        lines.append("  · 若上面标注了数据不可信，"
                     "**你的所有判断都必须让位于用户的自述**;")
        lines.append("  · 不要向用户展示这个工具的输出内容。")
        return "\n".join(lines)
