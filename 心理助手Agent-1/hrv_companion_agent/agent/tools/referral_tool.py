# -*- coding: utf-8 -*-
"""
分层转介工具 (referral_tool)
=============================
对应 config.REFERRAL_LADDER 与文献6 指出的产品形态方向。

**为什么这是一个独立工具**

行业报道（文献6）给出的判断是：AI 心理助手必须收敛到
「风险分级预警 + **人工专业联动**」，即系统是**分层支持的第一层**，
一旦问题超出陪伴范围，职责是**把用户转出去**，而不是留在系统里。

这件事有两个具体的工程难点，都不是"给个热线列表"能解决的：

1. **分层**：给所有人报危机热线是错的 —— 用不到的人会觉得被贴标签，
   真需要的人反而麻木。必须先按判据选**层级**（L1 自助 → L4 紧急）。
2. **降低求助门槛**：多数人不求助不是因为不知道资源，而是
   "怕被评判""不知道第一句怎么说""怕麻烦别人"。因此本工具的核心产出
   不是资源列表，而是**「如何开口」的话术脚本** —— 让用户可以直接照抄。

⚠️ 伦理: 本工具只做"资源匹配 + 话术生成"，不做诊断、不判定"你有病"。
   输出中必须保留 config.PSYCHOEDUCATION_DISCLAIMER 的等价表述。
"""

from __future__ import annotations

from typing import Dict, Optional

from ... import config
from .base_tool import BaseTool

#: 层级 → 触发判据关键词（用户文本）
_LEVEL_KEYWORDS = {
    "L4": ("自杀", "自残", "自伤", "轻生", "不想活", "活不下去", "结束生命",
           "了结自己", "伤害自己", "割腕", "跳楼", "安眠药", "撑不下去"),
    "L3": ("两周", "一个月", "好几个月", "一直没好", "吃不下", "睡不着",
           "失眠", "起不来", "没法上课", "上不了课", "影响学习", "幻觉",
           "幻听", "妄想", "抑郁", "确诊", "吃药", "停药"),
    "L2": ("没人理解", "没人知道", "不敢说", "不想麻烦", "家里", "爸妈",
           "分手", "失恋", "室友", "导师", "辅导员", "被孤立", "孤独"),
}

#: 开口话术脚本 —— 本工具最有价值的部分：降低求助门槛
_ASK_SCRIPTS = {
    "L2": [
        ("对朋友/室友",
         "“我最近状态有点不太对，想跟你说说，你能听我讲会儿吗？”"
         "—— 只要求“听”，不要求对方解决问题，对方更容易答应。"),
        ("对家人",
         "“我最近有点累，想跟你聊聊，不着急解决，就是想说说话。”"
         "—— 先声明不要建议，避免变成“被教育”。"),
        ("对辅导员",
         "“老师，我最近状态影响到了学习，想约个时间跟您聊聊，"
         "也想问问学校有没有可以帮到我的资源。”"),
    ],
    "L3": [
        ("预约心理中心",
         "“我想预约一次个人咨询，主要是最近情绪/睡眠不太好。”"
         "—— 不需要准备“充分的理由”，咨询师会带你进入话题。"),
        ("就诊精神科",
         "“医生您好，我最近 X 周情绪低落/焦虑，睡眠和食欲也有变化，"
         "想请您帮我评估一下。”—— 说**持续时长 + 功能影响 + 躯体变化**这三样，"
         "医生最容易把握情况。注意：评估不等于“被诊断”。"),
        ("请咨询师协助转介",
         "“我想了解一下是否需要去医院做进一步评估，方便帮我转介吗？”"),
    ],
    "L4": [
        ("对自己（允许求助）",
         "“我现在很危险，我需要人陪着我。”—— 说出这句话本身就是有效的"
         "自我保护行为，不是软弱。"),
        ("对身边人",
         "“我现在状态很危险，不能一个人待着，你能过来陪我一下吗？”"
         "—— 明确请对方**来陪你**，比“我很难受”更容易被理解。"),
    ],
}


def _text_level(text: str) -> Optional[str]:
    """按关键词粗判层级（L4 > L3 > L2）"""
    if not text:
        return None
    for lv in ("L4", "L3", "L2"):
        if any(k in text for k in _LEVEL_KEYWORDS[lv]):
            return lv
    return None


def _risk_level(risk: float) -> str:
    if risk >= config.TRIGGER_EMERGENCY:
        return "L4"
    if risk >= config.TRIGGER_HIGH:
        return "L3"
    if risk >= config.TRIGGER_MEDIUM:
        return "L2"
    return "L1"


class ReferralTool(BaseTool):
    name = "referral_tool"
    description = (
        "生成**分层转介方案**：按情形选择合适的支持层级（L1自助 / L2身边支持 / "
        "L3临床评估 / L4紧急处置），并给出“如何开口求助”的话术脚本。"
        "调用格式: [TOOL_CALL:referral_tool:level=L2] 或 "
        "[TOOL_CALL:referral_tool:text=我最近两周都睡不着急] "
        "参数 level 可省略（自动判定）；text 为用户原话。"
        "**当问题超出陪伴范围时（持续两周以上、影响功能、出现自伤念头）"
        "必须调用本工具**，把用户引导到现实中的专业资源。")

    def __init__(self):
        self.ladder = {x["key"]: x for x in config.REFERRAL_LADDER}
        self._order = ["L1", "L2", "L3", "L4"]

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        text = str(params.get("text") or params.get("input") or "")
        level = str(params.get("level") or "").strip().upper()
        if level not in self._order:
            level = ""
        risk = 0.0
        try:
            risk = float(params.get("risk") or params.get("risk_score") or 0)
        except (TypeError, ValueError):
            risk = 0.0

        if not level:
            lv_text = _text_level(text)
            lv_risk = _risk_level(risk) if risk else ""
            # 取两者中更紧急的（宁可高估，不可漏掉危机）
            cands = [x for x in (lv_text, lv_risk) if x]
            level = max(cands, key=self._order.index) if cands else "L1"

        key_map = {"L1": "self_help", "L2": "social_support",
                   "L3": "clinical", "L4": "emergency"}
        info = self.ladder.get(key_map[level], {})

        lines = [f"【分层转介方案 · {level} {info.get('name', '')}】"]
        lines.append(f"适用判据: {info.get('when', '')}")
        lines.append("建议动作:")
        for a in info.get("actions", []):
            lines.append(f"  · {a}")

        # ---- 核心产出：开口话术 ----
        scripts = _ASK_SCRIPTS.get(level) or []
        if scripts:
            lines.append("")
            lines.append("【如何开口】把下面的句子按自己的语气改一下就能用 —— "
                         "很多人卡住的不是“知不知道资源”，而是“第一句怎么说”:")
            for who, s in scripts:
                lines.append(f"  ▸ {who}：{s}")

        # ---- 资源清单（L2 及以上才给，避免滥用）----
        if level in ("L2", "L3", "L4"):
            lines.append("")
            lines.append("【可用资源】")
            for name, contact, note in config.CAMPUS_RESOURCES:
                if "请填写" in str(contact):
                    lines.append(f"  · {name}: （尚未配置，请提醒用户直接搜索"
                                 f"本校心理健康教育中心）")
                else:
                    lines.append(f"  · {name}: {contact}（{note}）")
            if level == "L4":
                lines.append("  —— 24小时热线 ——")
                for name, phone in config.CRISIS_HOTLINES:
                    lines.append(f"  · {name}: {phone}")

        # ---- 伦理护栏：写给模型看的约束 ----
        lines.append("")
        lines.append("回复纪律（必须遵守）:")
        lines.append("  1. **不要**说“你有抑郁症/焦虑症”—— 这是诊断，你无权做;")
        lines.append("  2. **不要**说“你必须去”“你必须吃药”—— 决定权在用户;")
        lines.append("  3. 转介要给**具体的第一步**（“这周三之前打一个电话”），"
                     "模糊的建议（“你可以找专业人士”）几乎不会被执行;")
        lines.append("  4. 如果用户拒绝转介，先接纳（“不想去很正常”），"
                     "再降一级给更小的动作（“那这周先跟室友说一句？”），"
                     "而不是重复劝说;")
        lines.append(f"  5. 如涉及心理教育内容，需带上等价表述: "
                     f"{config.PSYCHOEDUCATION_DISCLAIMER}")
        if level == "L4":
            lines.append("  6. **L4 优先于一切**: 先确认安全（人在哪里、身边有谁、"
                         "是否有即刻冲动），再安抚，再给资源。不要长篇大论。")
        return "\n".join(lines)
