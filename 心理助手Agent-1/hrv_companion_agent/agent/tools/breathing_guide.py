# -*- coding: utf-8 -*-
"""
呼吸练习指导工具 (breathing_guide)
===================================
对应计划书 §4.3 tools/breathing_guide.py，v2 扩展为「按生理目标选呼吸模式」。

**v2 为什么不是"再加两个呼吸法"**

呼吸是唯一既能**即时降唤醒**、又能**直接改变 HRV 读数**的自助手段 ——
这对本项目有双重意义：

1. 对用户：延长呼气提高迷走张力，是最快的主观放松路径；
2. 对系统：呼吸是 HRV 测量中最常见的**混杂因素**。用户在做深呼吸时，
   LF 与 HF 都会显著改变，此时"情绪识别"的结果反映的是呼吸不是情绪。
   因此本工具会在返回中提醒模型：**干预后的读数不能与干预前直接比较**。

**模式选择的生理依据**

    resonance  约 6 次/分（吸 4 呼 6，总周期 10 秒）
        —— 这一频率最接近多数人的压力感受器共振频率（约 0.1 Hz），
           能最大化 RSA（呼吸性窦性心律不齐）振幅，是提升 HRV 的
           标准做法。文献中常称"共振呼吸"或"频率呼吸"。
    calm       吸 4 呼 6（不屏息）
        —— 纯延长呼气，激活副交感，无需屏息，最安全的常规放松。
    box        4-4-4-4（含屏息）
        —— 屏息对焦虑有"抓手"作用，能提供控制感；
           ⚠️ 但会加重惊恐发作与过度换气者的症状，故列为禁忌。
    sleep      4-7-8
        —— 长屏息 + 极长呼气，偏助眠；
           ⚠️ 同样不适合惊恐/呼吸系统疾病者。

⚠️ 禁忌必须遵守：本工具会在返回中明确标注每组模式的禁忌，
   模型不得对不适用的人推荐含屏息的模式。
"""

from __future__ import annotations

from typing import Dict, List

from .base_tool import BaseTool

#: 呼吸模式库
GUIDES = {
    "resonance": {
        "name": "共振呼吸（约 6 次/分）",
        "phases": [("吸气", 4), ("呼气", 6)],
        "rounds": 6,
        "total_sec": 60,
        "goal": "提升心率变异性（最大化 RSA 振幅）",
        "note": "这一频率接近多数人的压力感受器共振点（约 0.1 Hz），"
                "是提升 HRV 的标准做法。需要连续做 5 分钟以上才有明显效果，"
                "建议作为**日常练习**而非急救手段。",
        "caution": "无禁忌。唯一要求是「不憋气、不勉强」——"
                   "气不够就缩短一点节奏，绝不要用力吸满。",
        "daily": True,
    },
    "calm": {
        "name": "放松呼吸（吸 4 呼 6）",
        "phases": [("吸气", 4), ("呼气", 6)],
        "rounds": 5,
        "total_sec": 50,
        "goal": "即时降唤醒（激活副交感）",
        "note": "呼气长于吸气是激活副交感的关键。不含屏息，"
                "因此对惊恐发作、哮喘、孕期、高血压人群都安全。",
        "caution": "无禁忌，是绝大多数场景的首选。",
        "daily": False,
    },
    "box": {
        "name": "方块呼吸（4-4-4-4）",
        "phases": [("吸气", 4), ("屏息", 4), ("呼气", 4), ("屏息", 4)],
        "rounds": 3,
        "total_sec": 48,
        "goal": "建立控制感（含屏息，提供明确的结构）",
        "note": "四段等长、结构清晰，对「脑子里一团乱」的人有安定作用。",
        "caution": "⚠️ **禁忌**: 惊恐发作、过度换气、哮喘或其它呼吸系统疾病、"
                   "孕晚期 —— 屏息会加重症状。上述情况改用 calm。",
        "daily": False,
    },
    "sleep": {
        "name": "4-7-8 助眠呼吸",
        "phases": [("吸气", 4), ("屏息", 7), ("呼气", 8)],
        "rounds": 4,
        "total_sec": 76,
        "goal": "助眠 / 深度放松",
        "note": "长屏息加极长呼气，降低整体代谢唤醒水平，适合睡前。",
        "caution": "⚠️ **禁忌**: 惊恐发作、呼吸系统疾病、心肺功能不佳者；"
                   "屏息 7 秒对不习惯的人偏长，可先减到 4-4-6 适应。",
        "daily": False,
    },
    "sigh": {
        "name": "生理性叹息（单次急救）",
        "phases": [("吸气", 3), ("补吸", 1), ("呼气", 6)],
        "rounds": 3,
        "total_sec": 30,
        "goal": "30 秒内最快降唤醒",
        "note": "双吸气复张塌陷肺泡 + 长呼气提高迷走张力，"
                "是目前起效最快的单次呼吸干预。适合情绪刚冲上来的那一下。",
        "caution": "无禁忌。注意第二口吸气要短、要轻，不要把肺撑到不适。",
        "daily": False,
    },
}

#: 情绪 / 目标 → 推荐模式（默认值，可被 params.type 覆盖）
EMOTION_DEFAULT = {
    "高焦虑": "calm",
    "低焦虑": "calm",
    "疲劳": "sleep",
    "平静": "resonance",
    "愉悦": "resonance",
}

#: 阶段 → 引导话术模板（自己实现而非依赖外部文案）
_PHASE_TEXT = {
    "吸气": "（吸气）用鼻子慢慢吸气，{sec} 秒……感受腹部慢慢鼓起。",
    "补吸": "（补吸）再短促地补一口，把肺轻轻填满。",
    "呼气": "（呼气）用嘴缓缓呼出，{sec} 秒……把紧张一点点送出去。",
    "屏息": "（屏息）轻轻停住，{sec} 秒……不用吸满，也不要憋紧。",
}


class BreathingGuideTool(BaseTool):
    name = "breathing_guide"
    description = (
        "生成带节奏标记的呼吸引导脚本，并按生理目标选择模式。"
        "调用格式: [TOOL_CALL:breathing_guide:type=calm] "
        "参数 type 可选: sigh(30秒急救) / calm(即时降唤醒·无禁忌·默认) / "
        "box(建立控制感·含屏息) / sleep(助眠·含屏息) / "
        "resonance(提升HRV·需长期练习)，默认 calm。"
        "返回的 caution 含**禁忌**，必须遵守：惊恐发作与呼吸系统疾病者"
        "不可用含屏息的 box / sleep 模式。"
        "脚本请以温柔、缓慢的节奏念给用户，一个节拍一个节拍地引导。")

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_type(params: Dict) -> str:
        kind = str(params.get("type") or "").strip().lower()
        if kind in GUIDES:
            return kind
        # 允许传情绪，自动选模式（v1 只接受 type，这里做向上兼容）
        emo = str(params.get("emotion") or "").strip()
        if emo in EMOTION_DEFAULT:
            return EMOTION_DEFAULT[emo]
        inp = str(params.get("input") or "").strip().lower()
        if inp in GUIDES:
            return inp
        return "calm"

    def run(self, params: Dict) -> str:
        kind = self._resolve_type(params)
        guide = GUIDES.get(kind) or GUIDES["calm"]

        lines = ["【%s】共 %d 轮，约 %d 秒"
                 % (guide["name"], guide["rounds"], guide["total_sec"])]
        lines.append("生理目标: %s" % guide["goal"])
        lines.append("引导话术:")
        lines.append("  先找一个舒服的姿势坐好或躺下，肩膀放松，闭上眼睛。")

        for r in range(1, guide["rounds"] + 1):
            lines.append("  —— 第 %d 轮 ——" % r)
            for phase, sec in guide["phases"]:
                tpl = _PHASE_TEXT.get(phase)
                if tpl is None:
                    continue
                lines.append("    " + tpl.format(sec=sec))

        lines.append("  很好，做得真棒。现在感觉身体有没有放松一点？")
        lines.append("")
        lines.append("[提示] %s" % guide["note"])
        lines.append("[禁忌] %s" % guide["caution"])

        # ---- 关键：干预会污染后续 HRV 读数，必须让模型知道 ----
        lines.append("")
        lines.append("[给模型的边界说明 —— 不要念给用户]")
        lines.append("  · 呼吸会直接改变 HRV 的 LF 与 HF 成分，因此**做练习期间**"
                     "采集到的情绪识别结果反映的是呼吸，不是情绪。"
                     "不要把练习前/后的读数当作「情绪变好了」的证据;")
        lines.append("  · 结束后如需继续判断状态，应静息 1-2 分钟再采信新的读数;")
        if guide.get("daily"):
            lines.append("  · 本模式属于**日常练习**，请说明需要连续做 5 分钟以上、"
                         "并坚持一段时间才有累积效应，避免用户因一次没感觉而放弃;")
        else:
            lines.append("  · 本模式属于**即时干预**，一次做完就够了，"
                         "不要要求用户反复做（过度换气反而会加重不适）;")
        lines.append("  · 引导节奏要慢，逐句停顿，不要一口气念完;"
                     "用户说头晕、胸闷要**立即停止**并让ta正常呼吸。")

        # 禁忌提醒：把含屏息模式明确标注出来
        if any(p == "屏息" for p, _ in guide["phases"]):
            lines.append("  · ⚠️ 本模式含屏息。若用户有惊恐发作、过度换气、哮喘、"
                         "心肺疾病或处于孕晚期，**改用 calm**，并解释"
                         "「换个不含屏气的版本，对你更合适」。")
        lines.append("[说明] 以上为缓解技巧，不构成治疗建议；"
                     "若胸闷心慌反复出现，应就医排查躯体原因。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    @staticmethod
    def available_types() -> List[str]:
        """全部可用模式（供 CLI/文档列出）"""
        return list(GUIDES.keys())
