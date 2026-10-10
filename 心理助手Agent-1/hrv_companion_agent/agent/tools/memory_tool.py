# -*- coding: utf-8 -*-
"""
记忆写入工具 (memory_tool)
===========================
对应计划书 §4.3 tools/memory_tool.py，v2 由「扁平事实表」升级为
「FTA 情境单元 + 时序链接」。

**v1 的问题：事实表记不下「事情」**

v1 把信息存成 `facts[fact_N] = "用户喜欢在晚上复习（2026-09-30）"`。
这种扁平的键值结构有三个硬伤：

1. **记不下时间指向。** 「上次面试砸了」和「下周三要面试」是同一句话的
   两种时间指向，但扁平存储后无法区分 —— 于是「后续追问」变成了随机行为。
2. **记不下未完结状态。** 用户说「我打算去找导师聊聊」，这件事当时没有
   结论。v1 会把它当成一条既成事实存下，导致之后无法自然地追问
   「后来联系上了吗」。
3. **记不下东西之间的关联。** 用户三天后说「导师说让我再等等」，
   这两条本该被连成一条时间线，v1 只能存成两条互不相干的事实。

**v2 的依据（文献5 FTA-Mem）**

文献5 的核心结论是：记忆应按**情境级**组织并携带五元组锚点

    ⟨ x_F 事实锚, x_T 时间锚, x_A 情感锚, e 证据指针, o 构造状态 ⟩

其中时间锚在消融实验中贡献最大、情感锚最不可靠。因此本工具：

* 写入时**必须**带上时间指向（past / present / future）；
* 当用户说的是**计划或未完结的事**时，标记 `lifecycle='pending'`，
  这样 recall_tool 才能把它挑出来做自然追问；
* 支持显式建立与已有记忆的**时序链接**（updates / contradicts /
  followup_of），把「事情的发展」串起来。

.. note::
   本项目为纯 Python 实现，**没有**文献5 原版的 LLM 边界检测器
   （用于从自由对话中自动切分情境单元）。这里采用规则版抽取，
   由模型在调用本工具时显式给出字段 —— 这是有意的取舍：
   规则版召回率低但可控，不会凭空造出用户没说过的情境。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from ... import config
from .base_tool import BaseTool

#: 时间锚的判定线索（按优先级）
_FUTURE_HINTS = ("打算", "准备", "计划", "想要", "下周", "明天", "后天",
                 "之后要", "接下来", "准备去", "要去", "报名", "约了",
                 "想试试", "希望")
_PAST_HINTS = ("上次", "昨天", "前天", "之前", "以前", "上周", "当时",
               "已经", "刚刚", "那天", "考完", "结束了")
_PRESENT_HINTS = ("最近", "这周", "这几天", "正在", "目前", "现在", "今天")

#: 未完结状态线索（决定 lifecycle）
_PENDING_HINTS = ("打算", "准备", "计划", "想要", "想试试", "约了",
                  "下周", "明天", "要去", "要跟", "要找", "还没",
                  "不知道怎么办", "在纠结")

#: 情境类型规则（规则版抽取，非文献5 的 LLM 检测器）
_TYPE_RULES = (
    ("risk_signal", ("焦虑", "紧张", "睡不着", "失眠", "睡不好", "睡得不好",
                     "崩溃", "难受", "压力", "慌", "害怕", "喘不上",
                     "撑不住", "哭", "吃不下", "提不起劲")),
    ("homework_or_plan", ("考试", "作业", "论文", "答辩", "开题", "绩点",
                          "挂科", "毕设", "ddl", "deadline", "实习",
                          "考研", "保研", "找工作", "面试", "报名",
                          "补考", "重修", "学习", "课题")),
    ("relationship_event", ("室友", "朋友", "同学", "导师", "老师", "辅导员",
                            "爸妈", "父母", "家里", "男朋友", "女朋友",
                            "分手", "失恋", "吵架", "冷战", "社团")),
    ("belief_candidate", ("我不行", "我很差", "我就是", "我总觉得", "我从来没",
                          "我是个", "我不配", "我没什么", "都是我的错")),
)

_SITUATION_ZH = {
    "risk_signal": "情绪紧张事件",
    "homework_or_plan": "学业/计划",
    "relationship_event": "人际关系",
    "belief_candidate": "自我看法",
    "event": "具体事件",
    "general_fact": "个人信息",
}


def infer_situation_type(text: str) -> str:
    """规则版情境类型推断（关键词命中，取最具体的一类）"""
    t = text or ""
    for stype, keys in _TYPE_RULES:
        if any(k in t for k in keys):
            return stype
    return "general_fact"


def infer_time_orientation(text: str) -> str:
    """规则版时间锚推断（future 优先于 past —— 计划比回忆更值得记）"""
    t = text or ""
    if any(k in t for k in _FUTURE_HINTS):
        return "future"
    if any(k in t for k in _PAST_HINTS):
        return "past"
    if any(k in t for k in _PRESENT_HINTS):
        return "present"
    return "unknown"


def infer_lifecycle(text: str) -> str:
    """是否属于「说了一半、还没有结论」的事

    取值必须落在 memory_manager.LIFECYCLE 内
    （pending / completed / contradicted / archived / unknown）——
    这里只用 pending 与 completed 两种，「未知」交给上游判断。
    """
    return "pending" if any(k in (text or "") for k in _PENDING_HINTS) \
        else "completed"


def _norm_participants(text: str) -> str:
    """粗抽参与者（人物词），用于结构化线索匹配"""
    found = []
    for k in ("室友", "朋友", "同学", "导师", "老师", "辅导员", "爸妈",
              "父母", "妈妈", "爸爸", "男朋友", "女朋友", "家人"):
        if k in (text or "") and k not in found:
            found.append(k)
    return ",".join(found)


class MemoryTool(BaseTool):
    name = "memory_tool"
    description = (
        "把用户提到的**情境**写入长期记忆（FTA 情境单元），供之后自然追问后续。"
        "调用格式: [TOOL_CALL:memory_tool:fact=用户下周三要跟导师谈开题] 或 "
        "[TOOL_CALL:memory_tool:fact=用户面试砸了,type=homework_or_plan,"
        "time=past,emotion=低焦虑] "
        "参数: fact(必填，用用户的原话或最少改动) / type(situation_type) / "
        "time(past|present|future) / emotion / participants / "
        "link_to(已有记忆id) / relation(updates|followup_of|contradicts) "
        "**建议在用户自然提到计划、事件、重要关系、自我看法时记录**；"
        "计划类（future）尤其值得记 —— 那是之后能追问后续的素材。")

    def __init__(self, memory_manager, user_id: str = None):
        self._memory = memory_manager
        self._user_id = user_id or getattr(
            memory_manager, "user_id", config.DEFAULT_USER_ID)
        self._recent_ids: List[int] = []

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        content = str(params.get("fact") or params.get("content")
                      or params.get("input") or "").strip()
        if not content:
            return "❌ 没有可记录的内容。"

        # ---- 更正路径：用户说「你记错了」----
        fix_id = params.get("fix_id") or params.get("unit_id")
        if fix_id:
            return self._correct(fix_id, content)

        stype = str(params.get("type") or params.get("situation_type") or "")
        if not stype:
            stype = infer_situation_type(content)
        time_or = str(params.get("time") or params.get("time_orientation") or "")
        if not time_or:
            time_or = infer_time_orientation(content)
        emotion = str(params.get("emotion") or "").strip()
        participants = str(params.get("participants") or "").strip() \
            or _norm_participants(content)
        intention = str(params.get("intention") or "").strip()

        # ---- 未开启 FTA 时退回 v1 扁平事实（保证向后兼容）----
        if not config.MEMORY_FTA_ENABLED:
            return self._flat_fact(content)

        # ---- 去重检查（FTA 与画像事实两侧都要查）----
        dup = self._find_duplicate(content)
        if dup:
            return ("📝 记忆库中已有相似记录，未重复保存。（%s）\n"
                    "如果用户是在**补充新进展**（同一件事的新情况），"
                    "请改用 link_to=<已有id> 建立 followup_of 链接，"
                    "而不是当成新记忆记录 —— 这样「后来怎么样了」才串得起来。"
                    % dup)

        lifecycle = "pending" if time_or == "future" else infer_lifecycle(content)
        state = self._provider_state()
        physio = {}
        if state:
            physio = {"emotion": state.get("emotion"),
                      "risk": state.get("risk_score"),
                      "quality": (state.get("quality") or {}).get("quality_level"),
                      "pad": state.get("pad")}

        link_to = params.get("link_to") or params.get("link")
        relation = str(params.get("relation") or "followup_of").strip()
        link_ids: List[int] = []
        if link_to:
            link_ids = [int(x) for x in re.findall(r"\d+", str(link_to))]

        try:
            unit_id = self._memory.save_fta_unit(
                fact_text=content,
                situation_type=stype,
                emotion=emotion,
                emotion_conf=float(state.get("confidence") or 0.0),
                time_orientation=time_or,
                participants=participants,
                intention=intention,
                physio=physio or None,
                completeness="partial" if lifecycle == "pending" else "complete",
                lifecycle=lifecycle,
                link_to=link_ids or None,
                link_relation=relation)
        except Exception as e:  # noqa: BLE001 落库失败不应中断对话
            return f"（记忆写入失败：{e}；请继续正常对话。）"

        if unit_id:
            self._recent_ids.append(int(unit_id))
            self._recent_ids = self._recent_ids[-5:]

        # ---- 双写：画像事实（供 build_memory_context 的「用户信息」段直接使用）----
        # 为什么要双写而不是只留 FTA：FTA 单元面向"检索与追问"，
        # 必须经过检索打分才能被用上；而画像事实是**每轮都注入**的
        # 轻量上下文（称呼、专业、偏好这类稳定信息）。
        # 两者用途不同，因此都保留 —— 但去重只做一次（见 _find_duplicate）。
        self._write_profile_fact(content)

        zh = _SITUATION_ZH.get(stype, stype)
        time_zh = {"past": "过去", "present": "眼下",
                   "future": "之后计划", "unknown": ""}.get(time_or, "")
        lines = ["📝 已记住（#%s）: %s" % (unit_id, content)]
        meta = ["类型=%s" % zh]
        if time_zh:
            meta.append("时间=%s" % time_zh)
        if emotion:
            meta.append("情绪=%s" % emotion)
        meta.append("状态=%s" % ("未完结（可追问后续）" if lifecycle == "pending"
                               else "已记录"))
        lines.append("  " + " | ".join(meta))
        if link_ids:
            lines.append("  已与记忆 %s 建立链接（%s）"
                         % (link_ids, relation))
        for t, r in self._conflict_check(content):
            lines.append("  ⚠️ 可能与此前的记录（#%s）%s: %s"
                         % (t, r, "请让用户确认哪个是真的，并更正旧的"))
        lines.append("")
        lines.append("回复纪律: **不要**说「已记录」「已存入记忆」这类系统口吻；"
                     "自然地体现你记住了就行（「那你下周三跟导师谈的时候，"
                     "想先说哪一部分？」）。")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _provider_state(self) -> Dict:
        provider = getattr(self, "_state_provider", None)
        if not provider:
            return {}
        try:
            return provider() or {}
        except Exception:  # noqa: BLE001 工具失败不应中断对话
            return {}

    def set_state_provider(self, provider):
        """注入状态回调（由 CompanionAgent 在装配时调用）"""
        self._state_provider = provider

    def _find_duplicate(self, content: str) -> str:
        """查重：画像事实 + 已存情境单元两侧都查

        Returns:
            命中的说明文字（未命中返回空串）

        为什么必须在**写入前**查：FTA 的 partial 融合只处理
        「未完结 → 补完」这一种情况；而"用户把同一件事又说了一遍"
        是另一种情况 —— 若不拦，记忆库会被同一件事灌满，
        检索时相似度全部趋同，反而找不到真正新的信息。
        """
        head = content[:20]
        # ① 画像事实
        try:
            facts = (self._memory.get_user_profile() or {}).get("facts") or {}
        except Exception:  # noqa: BLE001
            facts = {}
        for k, v in facts.items():
            sv = str(v)
            if head and (head in sv or sv[:20] in content):
                return str(k)
        # ② 情境单元（高相似度命中才算重复，避免误拦"补充进展"）
        try:
            for u in self._memory.retrieve_fta(content, top_k=3):
                old = str(u.get("fact_text") or "")
                if not old:
                    continue
                if head and (head in old or old[:20] in content):
                    return "#%s" % u.get("id")
        except Exception:  # noqa: BLE001
            pass
        return ""

    def _write_profile_fact(self, content: str):
        """把内容写入画像 facts（与 FTA 双写，见 run() 中的说明）"""
        import time
        try:
            profile = self._memory.get_user_profile() or {}
            facts = profile.setdefault("facts", {})
            facts["fact_%d" % (len(facts) + 1)] = "%s（%s）" % (
                content, time.strftime("%Y-%m-%d"))
            self._memory.update_user_profile({"facts": facts})
        except Exception:  # noqa: BLE001 失败不影响 FTA 已写入的事实
            pass

    def _flat_fact(self, content: str) -> str:
        """v1 路径：仅写入用户画像的 facts（FTA 关闭时的兜底）"""
        dup = self._find_duplicate(content)
        if dup:
            return "📝 记忆库中已有相似记录，未重复保存。（%s）" % dup
        self._write_profile_fact(content)
        return "📝 已记住：%s。（FTA 情境记忆未开启，本条存为画像事实）" % content

    def _correct(self, unit_id, content: str) -> str:
        """更正已有记忆（被纠正权 —— 用户说「你记错了」时的落地路径）"""
        try:
            ok = self._memory.update_fta_unit(
                int(re.findall(r"\d+", str(unit_id))[0]),
                fact_text=content, completeness="complete",
                lifecycle="settled")
        except Exception as e:  # noqa: BLE001
            return "（更正失败：%s）" % e
        if not ok:
            return "（未找到该记忆，可能已被删除。）"
        return ("✅ 已按你的说法更正记忆 #%s → %s\n"
                "回复时请**不要**辩解，也不要解释「系统之前判断错了」；"
                "一句「是我记错了，抱歉」就够了。"
                % (unit_id, content))

    def _conflict_check(self, content: str):
        """粗查是否与已有记忆矛盾（供模型决定是否请用户确认）"""
        out = []
        try:
            units = self._memory.retrieve_fta(content, top_k=3)
        except Exception:  # noqa: BLE001
            return out
        for u in units:
            old = u.get("fact_text") or ""
            if not old or old == content:
                continue
            # 简单反义线索（明确的否定翻转）
            if ("不" in content) != ("不" in old):
                score = u.get("score")
                if score is not None and float(score) >= 0.5:
                    out.append((u.get("id"), "表述方向相反", old))
        return out[:1]
