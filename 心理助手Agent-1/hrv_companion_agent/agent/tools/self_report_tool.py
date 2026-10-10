# -*- coding: utf-8 -*-
"""
主观自评工具 (self_report_tool)
================================
本项目**唯一能提供情绪识别"真值"的通道**，同时承担三条职责。

**职责一：伦理 —— 用户对自身感受有最终解释权**

任何生理推断都可能出错（尤其本项目实测已确认：指腹 PPG 在接触不良时
会出现「HR 正常但 HRV 是假的」）。当用户说"我其实还好"而系统判定
"高焦虑"时，正确的做法不是坚持算法，而是**以用户为准并记录这次分歧**。

**职责二：科研 —— 为「情绪识别准确率 > 85%」提供真值**

这个指标是项目的核心验收项，但生理信号本身没有真值：
没有主观标注就无法计算准确率，只能用"自洽性"代替，而自洽性无法证伪。
本工具把「(客观判定, 主观评分)」成对落库（memory_manager.self_report），
使 20 人实验可以直接导出带标注的数据集，而不必事后补采。

**职责三：个性化 —— 校准"生理-主观"映射的个体偏差**

不同人对同一程度的紧张，其 HRV 变化的幅度并不相同（文献1 之所以要
用「相对特征值法」，就是在处理这件事）。系统记录的成对数据可以估出
"这位用户的生理读数要打多大折扣才等于ta的主观感受"。

⚠️ 表达纪律：主观评分是**用户说了算**的东西，Agent 绝不能反驳、
   "纠正"或暗示用户"其实你比你说的更难受"。
"""

from __future__ import annotations

from typing import Dict, Optional

from ... import config
from .base_tool import BaseTool

#: 评分区间 → 语义标签（用于让模型理解分数的含义，而非只看到数字）
SCORE_BANDS = (
    (9, "非常难受，已接近难以承受"),
    (7, "明显难受，影响到当下状态"),
    (4, "中等程度，有些不适但还能应对"),
    (2, "轻微，基本在可接受范围"),
    (0, "很平稳，几乎没影响"),
)

#: 常用情绪词 → 主观标签归一（用户说什么就是什么，这里只做归并）
_EMOTION_ALIASES = {
    "平静": "平静", "放松": "平静", "还好": "平静", "正常": "平静", "挺好": "平静",
    "开心": "愉悦", "高兴": "愉悦", "愉快": "愉悦", "兴奋": "愉悦", "不错": "愉悦",
    "有点紧张": "低焦虑", "紧张": "低焦虑", "担心": "低焦虑", "压力": "低焦虑",
    "焦虑": "高焦虑", "很焦虑": "高焦虑", "害怕": "高焦虑", "恐慌": "高焦虑",
    "烦躁": "高焦虑", "生气": "高焦虑", "愤怒": "高焦虑",
    "累": "疲劳", "疲惫": "疲劳", "困": "疲劳", "没力气": "疲劳", "疲惫不堪": "疲劳",
    "低落": "低焦虑", "难过": "低焦虑", "情绪低": "低焦虑",
}


def _band(score: float) -> str:
    for th, label in SCORE_BANDS:
        if score >= th:
            return label
    return SCORE_BANDS[-1][1]


class SelfReportTool(BaseTool):
    name = "self_report_tool"
    description = (
        "记录用户的**主观**情绪自评（0~10 不适程度 + 可选情绪词），"
        "并与系统当时的生理判定做一致/冲突核对。"
        "调用格式: [TOOL_CALL:self_report_tool:score=6] 或 "
        "[TOOL_CALL:self_report_tool:score=2,emotion=平静,note=只是有点困] "
        "**使用时机**：当用户用语言描述自己的感受时（“我最近还行”/“我快撑不住了”/"
        "“其实没那么严重”）。若用户的主观描述与系统判断不一致，"
        "**以用户为准**，本工具会告诉你是否需要降低对生理判断的采信度。")

    def __init__(self, memory_manager, user_id: str = None,
                 state_provider=None):
        """
        Args:
            memory_manager: MemoryManager 实例（落库 + 校准）
            user_id: 用户标识
            state_provider: 无参回调 → 当前客观侧状态 dict（用于配对存储）
        """
        self._memory = memory_manager
        self._user_id = user_id or getattr(
            memory_manager, "user_id", config.DEFAULT_USER_ID)
        self._state_provider = state_provider

    # ------------------------------------------------------------------
    def _state(self) -> Dict:
        if not self._state_provider:
            return {}
        try:
            return self._state_provider() or {}
        except Exception:  # noqa: BLE001
            return {}

    def _parse_score(self, params: Dict) -> Optional[float]:
        raw = params.get("score", params.get("value"))
        if raw is None:
            raw = params.get("input") if _looks_numeric(params.get("input")) else None
        if raw is None:
            return None
        try:
            score = float(raw)
        except (TypeError, ValueError):
            return None
        # 容错：用户/模型可能给 0-100 分制
        if score > 10:
            score = score / 10.0
        return max(0.0, min(10.0, score))

    @staticmethod
    def normalize_emotion(text: str) -> str:
        """把用户的自然说法归并到系统术语（**仅用于对齐，不改变用户原话**）"""
        if not text:
            return ""
        t = str(text).strip()
        if t in _EMOTION_ALIASES:
            return _EMOTION_ALIASES[t]
        for k, v in _EMOTION_ALIASES.items():
            if k in t:
                return v
        return t

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        if not config.SELF_REPORT_ENABLED:
            return "主观自评已关闭。"

        score = self._parse_score(params)
        emotion = self.normalize_emotion(
            str(params.get("emotion") or params.get("feeling") or ""))
        note = str(params.get("note") or "").strip()

        if score is None and not emotion:
            return ("请提供主观评分（0~10）或情绪词，例如 "
                    "[TOOL_CALL:self_report_tool:score=6,emotion=焦虑]。"
                    "**不要替用户打分** —— 必须来自用户的表达。")

        state = self._state()
        rid = 0
        try:
            rid = self._memory.save_self_report(
                score if score is not None else -1.0,
                emotion=emotion, note=note, objective=state)
        except Exception as e:  # noqa: BLE001 落库失败不应中断对话
            return f"（自评未能落库：{e}；但请继续按用户自述回应。）"

        lines = ["【已记录用户的主观自评】"]
        if score is not None:
            lines.append(f"  主观不适程度: {score:.0f}/10 —— {_band(score)}")
        if emotion:
            lines.append(f"  主观情绪词: {emotion}")
        if note:
            lines.append(f"  用户原话补充: {note}")

        # ---- 主客观对照（本工具的核心输出）----
        o_emo = state.get("emotion")
        o_risk = state.get("risk_score")
        o_conf = state.get("confidence")
        o_quality = (state.get("quality") or {}).get("quality_ok", True)
        if state.get('simulation') is False and not state.get('validated'):
            o_emo = None
            lines.append('当前没有可用于对照的生理分类，以用户自述为准；自评已保留。')
        if o_emo:
            lines.append("")
            lines.append("【与系统生理判定的对照（供你判断该信谁）】")
            lines.append(f"  系统判定: {o_emo}"
                         f"（置信度 {float(o_conf or 0):.2f}，"
                         f"紧张评分 {o_risk if o_risk is not None else 'N/A'}）"
                         f"  信号质量: {'合格' if o_quality else '不合格'}")

            conflict = False
            reason = ""
            if not o_quality:
                conflict, reason = True, "生理信号质量不合格，判定本身就不可信"
            elif score is not None and o_risk is not None:
                gap = abs(score * 10.0 - float(o_risk))
                if gap >= config.SELF_REPORT_CONFLICT_GAP * 10:
                    conflict = True
                    reason = f"主观与客观差距达 {gap:.0f}/100 分"
            if emotion and o_emo and emotion != o_emo:
                conflict = True
                reason = (reason + "；" if reason else "") + \
                    f"主观情绪词（{emotion}）与系统判定（{o_emo}）不一致"

            if conflict:
                lines.append(f"  ⚠️ **判定为不一致**（{reason}）")
                lines.append("  → 处置要求:")
                lines.append("     1. **以用户自述为准**，不要暗示ta判断错了，"
                             "也不要说“但数据显示你其实很焦虑”;")
                lines.append("     2. 生理判断在本轮的可信度打折"
                             f"（×{config.SELF_REPORT_CONFLICT_DISCOUNT}），"
                             "不要据此调整语气强度;")
                lines.append("     3. 可以把这种差异当作一个**温和的提问机会**："
                             "“我这边感觉到的和你说的不太一样，"
                             "你愿意多说两句吗？”—— 但不要反复追问;")
                lines.append("     4. 若用户主动说“我其实没那么严重”，"
                             "**接受它**，这就是最准确的信息。")
            else:
                lines.append("  ✅ 主客观一致，生理判断可作为参考。")

        # ---- 隐私与用途说明（必须主动告知）----
        cons = {}
        try:
            cons = self._memory.self_report_consistency()
        except Exception:  # noqa: BLE001
            pass
        if cons.get("n"):
            lines.append("")
            lines.append(f"  （累计已记录 {cons['n']} 次自评，"
                         f"其中主客观不一致 {cons['n_conflict']} 次。"
                         "这些数据保存在本机，用户可随时导出或删除。）")
        lines.append("")
        lines.append("回答纪律: **不要**把“已记录”这件事说得像在填表；"
                     "也不要让用户觉得被打分。自评是给ta自己用的镜子，"
                     "不是给你的评分卡。")
        return "\n".join(lines)


def _looks_numeric(v) -> bool:
    if v is None:
        return False
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
