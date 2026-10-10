# -*- coding: utf-8 -*-
"""
危机干预工具 (crisis_tool)
===========================
对应计划书 §4.3 tools/crisis_tool.py，v2 由「单档协议」升级为
「意念分级 + 安全计划 + 明确禁止清单」。

**v1 的问题**

v1 只有一档协议（关键词命中 → 给出四步 + 热线列表）。这在专业上不够，
原因有三：

1. **风险不是二元的。** 被动意念（「活着没意思」）与有计划的意念
   （「想好了怎么做」）需要完全不同的应对强度。把两者一视同仁，
   前者会被过度反应吓到，后者会被安抚话术拖延。
2. **缺少「环境安全」这一步。** 国际通行的做法是：在提供资源之前，
   先降低**即刻可及的危险**（身边是否有药、有无高处环境）。
   这一步比说多少安慰的话都重要。
3. **缺少「不要做什么」。** 危机中真正会造成伤害的往往是好心的错误：
   承诺绝对保密、与当事人辩论「值不值得」、激将、把话题拉回自己身上。

**v2 的改动**

* **直接询问准则**：明确告知模型可以且应当**直接问**自杀意念。
  这是一个常被误解的点 —— 大量研究支持「直接询问不会诱导自杀」，
  而绕弯子（「你不会想不开吧？」）既无效又增加痛苦。不敢问，
  才是最大的风险。
* **意念分级**：passive（被动意念）/ active（主动意念）/ plan（有计划）/
  preparation（有准备行为），各级给出不同的第一步。
* **安全计划**：把「给热线」升级为可执行的六步安全计划，
  并明确这是**与用户共同填写**的东西，不是发给ta的一张清单。
* **禁止清单**：把最容易犯的错误写死在输出里。

⚠️ 伦理: 本工具只做「风险分层 + 陪伴 + 资源链接 + 安全计划」，
   不诊断、不评估"自杀风险等级"（那是临床工作），不承诺疗效。
   一旦识别到高危信号，目标只有两个：**保证安全**、**接到人手上**。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ... import config
from .base_tool import BaseTool

# ---------------------------------------------------------------- 关键词
#: 被动意念（活着没意思，但没有明确的行动意图）
#:
#: 注意语言变体：口语里「我不想活了」很少照着字面说，
#: 更多是「我不太想活了」「活着真的没意思」「睡一觉别再醒来」。
#: 只匹配字面写法会大面积漏检 —— 这是本类工具最容易犯的错。
PASSIVE_KEYWORDS = [
    "活着没意思", "活着好累", "活着没意义", "活着没劲", "活着真累",
    "不想活", "不太想活", "不想活着", "活不下去", "撑不下去", "熬不下去",
    "没意思", "没意义", "没希望", "没指望", "看不到意义", "看不到希望",
    "不如不存在", "不想存在", "想消失", "消失了算了", "人间蒸发",
    "睡不醒就好了", "希望一觉不醒", "睡过去别醒", "想解脱", "想结束",
    "别再醒来", "不再醒来", "醒不来就好", "一觉不醒", "想一睡不醒",
]

#: 主动意念（有想法，但未提及方式/计划）
ACTIVE_KEYWORDS = [
    "自杀", "自残", "自伤", "轻生", "结束生命", "了结自己", "了结生命",
    "伤害自己", "伤害我自己", "想死", "想死了", "好想死", "真的想死",
    "不想活了", "去死", "解脱了",
]

#: 计划 / 准备行为（最高危）
PLAN_KEYWORDS = [
    "割腕", "跳楼", "安眠药", "跳河", "上吊", "跳轨", "计划好了",
    "想好了怎么", "买好了", "攒了", "写好遗书", "交代后事", "安排好",
]

#: 分级 → 干预强度
LEVELS = {
    "passive": {
        "name": "被动意念",
        "first_step": "先**建立连接**：把你的担心直接说出来，"
                      "「你刚才那句话让我有点担心你」—— 承认你听到了，"
                      "而不是立刻转向解决方案。",
        "tone": "温和、不惊慌。不要立刻抬高风险等级，也不要轻描淡写。",
    },
    "active": {
        "name": "主动意念",
        "first_step": "**直接询问**：可以、也应当直接问「你是有过想结束"
                      "生命的念头吗？」—— 直接询问不会诱导自杀，"
                      "这是国际共识。绕弯子反而让ta更孤立。",
        "tone": "平静、直接、不评判。问完要**等**，沉默是允许的。",
    },
    "plan": {
        "name": "有计划（高危）",
        "first_step": "**不离开对话 + 立刻降低环境风险**：先确认ta此刻在哪里、"
                      "身边有没有人、有没有已经准备好的方式或物品。"
                      "然后请ta做一件具体的事：把东西交给别人、离开危险场所。",
        "tone": "坚定而温暖。此时**不要**用「我们聊聊」来代替具体行动。",
    },
    "preparation": {
        "name": "已采取准备行为（最高危）",
        "first_step": "**立即联系紧急服务**：引导ta拨打 110/120 并联系身边人，"
                      "不要试图独自处理。同时明确告诉ta需要马上有人陪在身边。",
        "tone": "简短、明确、指令化。不要长篇解释，不要讨论原因。",
    },
}

#: 安全计划（Stanley & Brown 安全计划简化六步）
SAFETY_PLAN_STEPS = [
    ("① 警示信号",
     "有哪些念头、画面、感觉出现时，说明「我该启动这个计划了」？"
     "（例：又开始觉得「都一样」，或者一整晚睡不着）"),
    ("② 自己能做的应对",
     "以前有没有哪一次，是你自己扛过来一点的？当时做了什么？"
     "（例：出门走一圈、冲个澡、把感受写下来）"),
    ("③ 能分散注意的人或场所",
     "哪些地方、哪些人，会让你稍微好一点，而且**不需要解释什么**？"
     "（例：去自习室、找个朋友待着、便利店坐一会儿）"),
    ("④ 可以联系的人",
     "至少写两个人名和电话 —— 如果只写得出一个也没关系，"
     "辅导员/宿管/家人任何一个都可以。"),
    ("⑤ 专业资源",
     f"24小时热线：{config.CRISIS_HOTLINES[0][1]}（{config.CRISIS_HOTLINES[0][0]}）；"
     "紧急情况 110 / 120；学校心理健康教育中心（工作日可预约）。"),
    ("⑥ 让环境更安全",
     "这一步最重要：把可能造成伤害的东西**交给别人保管**、"
     "或者放到自己拿不到的地方。不是「靠意志力控制」，是让环境帮你。"),
]

#: 明确禁止清单（危机中最容易造成伤害的"好心"）
FORBIDDEN = [
    ("不要承诺绝对保密",
     "「我保证不告诉任何人」是危险的 —— 它把你绑住，也让ta失去"
     "被你带去见真人的机会。正确说法：「我会一直在这里，但你的安全"
     "如果受到威胁，我需要帮你找到更多能帮上的人。」"),
    ("不要与ta辩论「值不值得」",
     "「你还有爸妈啊」「你成绩那么好」这类话，会把ta推入「我的痛苦"
     "不够正当」的二次伤害。痛苦不需要被证明。"),
    ("不要激将、不要讲道理",
     "「你要真想去早就去了」这类赌气式回应会直接造成伤害。"),
    ("不要用「你想多了」「都会过去的」收尾",
     "这两句话否定了ta当下的真实体验，是最常见的无效回应。"),
    ("不要把话题转到自己身上",
     "「我当年也……」在此刻是抢走了ta的空间。"),
    ("不要留ta独自面对",
     "识别到主动意念以上，必须推动「联系一个具体的人」这个动作，"
     "不能以「我陪着你」作为唯一方案 —— 你不是24小时可及的人。"),
    ("不要反复追问细节（方式、时间、地点）",
     "这既不是你的职责，也可能增加风险。你只需要知道「有没有」和"
     "「此刻安不安全」，细节交给专业人员。"),
]


def detect_level(text: Optional[str]) -> Optional[str]:
    """从用户文本判定危机分级（取最高危命中项）

    Returns:
        None / "passive" / "active" / "plan" / "preparation"
    """
    if not text:
        return None
    t = str(text)
    # 最高危优先
    if any(k in t for k in PLAN_KEYWORDS):
        # 「买好了药」这类准备行为 vs 「想到跳楼」这类被动意象
        if any(k in t for k in ("买好", "攒了", "写好", "交代后事", "安排好",
                                "计划好了", "想好了怎么")):
            return "preparation"
        return "plan"
    if any(k in t for k in ACTIVE_KEYWORDS):
        return "active"
    if any(k in t for k in PASSIVE_KEYWORDS):
        return "passive"
    return None


def level_from_risk(risk: float) -> Optional[str]:
    """从综合风险评分映射分级（生理侧无法直接测自杀风险，故只做弱映射）"""
    try:
        r = float(risk)
    except (TypeError, ValueError):
        return None
    if r >= config.TRIGGER_EMERGENCY:
        return "active"
    if r >= config.TRIGGER_HIGH:
        return "passive"
    return None


_ORDER = ["passive", "active", "plan", "preparation"]


class CrisisTool(BaseTool):
    name = "crisis_tool"
    description = (
        "危机干预协议（v2：意念分级 + 安全计划 + 禁止清单）。"
        "当用户文本出现自伤/自杀信号，或风险评分达到紧急阈值时调用。"
        "调用格式: [TOOL_CALL:crisis_tool:text=我想伤害自己] 或 "
        "[TOOL_CALL:crisis_tool:risk=99] 或 "
        "[TOOL_CALL:crisis_tool:level=active] "
        "返回: 分级后的分步干预协议、安全计划六步、专业求助资源、"
        "以及**必须避免的回应方式**。")

    def __init__(self):
        self.hotlines = config.CRISIS_HOTLINES

    # ------------------------------------------------------------------
    @staticmethod
    def detect_crisis_text(text: Optional[str]) -> bool:
        """兼容 v1 接口：文本是否含任何危机信号"""
        return detect_level(text) is not None

    def run(self, params: Dict) -> str:
        risk = 0.0
        try:
            risk = float(params.get("risk") or params.get("risk_score") or 0)
        except (TypeError, ValueError):
            risk = 0.0
        text = str(params.get("text") or params.get("input") or "")
        want = str(params.get("level") or "").strip().lower()

        lv_text = detect_level(text)
        lv_risk = level_from_risk(risk)
        cands = [x for x in (lv_text, lv_risk) if x]
        if want in LEVELS:
            cands.append(want)
        if not cands:
            return ("当前未达危机阈值。请保持陪伴：倾听、共情。"
                    "若用户情绪持续恶化，或出现「活着没意思」「想消失」"
                    "这类表达，立即再次调用本工具。"
                    "注意：**直接询问自杀意念不会诱导自杀** —— "
                    "有担心时宁可直接问，不要绕弯子。")

        level = max(cands, key=_ORDER.index)
        info = LEVELS[level]

        lines = ["⚠️ 危机干预协议已启动 · 分级: %s（%s）"
                 % (level, info["name"]), ""]
        lines.append("【第一步 · 优先于一切】")
        lines.append("  " + info["first_step"])
        lines.append("  语气: " + info["tone"])
        lines.append("")

        lines.append("【第二步 · 陪伴与稳定】")
        lines.append("  · 只说**短句**：「我在」「我听着」「你不用一个人扛」;")
        lines.append("  · 不要急着让ta「好起来」，陪伴本身就是干预;")
        lines.append("  · 如果ta愿意，可以做一次生理性叹息或"
                     "呼吸引导（调用 breathing_guide:type=calm）；"
                     "**不要**用含屏息的模式;")
        lines.append("  · 允许沉默。不追问、不填满每一个空白。")
        lines.append("")

        lines.append("【第三步 · 安全计划（与ta一起填，不是发一张清单）】")
        for title, content in SAFETY_PLAN_STEPS:
            lines.append("  %s: %s" % (title, content))
        lines.append("  填不完没关系 —— 能落实①和⑥就已经显著降低风险。")
        lines.append("")

        lines.append("【第四步 · 专业求助资源（24小时、免费、保密）】")
        for name, phone in self.hotlines:
            lines.append("  · %s: %s" % (name, phone))
        lines.append("  · 学校心理健康教育中心（工作日可预约，对学生免费）")
        lines.append("  转介话术与更高层级的资源匹配可调用 referral_tool。")
        lines.append("")

        lines.append("【必须避免的回应方式】")
        for title, why in FORBIDDEN:
            lines.append("  ✗ %s —— %s" % (title, why))
        lines.append("")

        lines.append("【直接询问准则（重要，常被误解）】")
        lines.append("  「直接问ta有没有自杀念头」**不会**诱导自杀 —— "
                     "这是国际自杀预防领域的共识。")
        lines.append("  正确的问法：「你是有过想结束生命的念头吗？」"
                     "「有想过具体怎么做吗？」")
        lines.append("  错误的问法：「你不会是想不开吧？」"
                     "「你没那么想吧？」—— 后者给了ta否认的台阶，"
                     "却让ta更孤立。")
        lines.append("")

        if level in ("plan", "preparation"):
            lines.append("【最高危附加要求】")
            lines.append("  · 不要以「我陪你聊」作为唯一方案，"
                         "必须推动「联系一个具体的、此刻能到场的人」;")
            lines.append("  · 若ta拒绝联系任何人，且存在即刻危险，"
                         "**明确告知**你会担心到需要请现实中的帮助介入"
                         "（这不是背叛，是保护）;")
            lines.append("  · 回复要短。长篇大论在此刻是负担。")

        lines.append("")
        lines.append("回复要点: 温暖、坚定、简短、具体。"
                     "优先保证安全，其次才有对话质量。"
                     "**本工具提供的是陪伴与资源链接，不能替代专业危机干预。**")
        return "\n".join(lines)
