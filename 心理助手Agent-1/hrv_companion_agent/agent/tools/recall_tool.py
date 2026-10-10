# -*- coding: utf-8 -*-
"""
情境记忆检索与洞察工具 (recall_tool)
=====================================
对应文献5 FTA-Mem 与文献4 MemoryBank。

**为什么需要这个工具**

Agent 要说出"上次你说要联系导师，后来怎么样了？"这类话，光有"最近聊天记录"
是不够的 —— 那只能复述字面内容。文献5 的核心结论是：记忆应按**情境级**
（一次完整的事情）组织，并携带五元组锚点：

    ⟨ x_F 事实锚, x_T 时间锚, x_A 情感锚, e 证据指针, o 构造状态 ⟩

其中「时间锚」在消融实验中贡献最大（去掉后性能退化最显著），
「情感锚」最不可靠 —— 所以本工具的输出会显式标出**当时情绪的可信度**，
避免 Agent 拿一条低质量信号下的情绪判断去"关心"用户。

**四种工作模式**

    mode=recall   按当前话题检索相关情境（混合检索：语义 + 结构化线索）
    mode=pending  取出"说了一半"的事，用于自然追问后续
    mode=pattern  情绪模式洞察（高发时段、主导情绪、个体触发点）
    mode=progress 干预有效性回顾（哪些策略对这位用户真的有效）

``pattern``/``progress`` 的结论来自**这位用户自己的数据**，因此比通用建议
更有说服力 —— 这也是本工具与"通用安慰话术"的关键区别。

⚠️ 隐私: 只能检索当前 user_id 的记忆；输出给模型的内容不含原始生理数值。
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

from ... import config
from .base_tool import BaseTool

#: 情境类型 → 中文（与 system.py 的 _FTA_RULES 保持一致）
SITUATION_ZH = {
    "risk_signal": "情绪紧张事件",
    "homework_or_plan": "学业/计划",
    "relationship_event": "人际关系",
    "belief_candidate": "自我看法",
    "event": "具体事件",
    "general_fact": "个人信息",
    "emotion_snapshot": "情绪状态",
}

#: 时间锚 → 中文
TIME_ZH = {
    "past": "过去", "present": "眼下", "future": "计划/将来", "unknown": "未明确",
}

#: 低质量情绪可信度阈值（与信号质量门禁同一口径）
_LOW_AFFECT_CONF = 0.5


class RecallTool(BaseTool):
    name = "recall_tool"
    description = (
        "检索长期记忆中的**情境级**单元，回答“用户以前说过什么/后来怎么样了”，"
        "或给出基于其本人数据的情绪模式与干预有效性洞察。"
        "调用格式: [TOOL_CALL:recall_tool:mode=recall,query=实习] / "
        "[TOOL_CALL:recall_tool:mode=pending] / "
        "[TOOL_CALL:recall_tool:mode=pattern] / "
        "[TOOL_CALL:recall_tool:mode=progress] "
        "参数 mode 默认 recall；recall 需提供 query。"
        "返回的是**供你参考的背景**，请自然地用，不要逐条复述，也不要声称记住没记过的事。")

    def __init__(self, memory_manager, user_id: str = None):
        self._memory = memory_manager
        self._user_id = user_id or getattr(memory_manager, "user_id", config.DEFAULT_USER_ID)

    # ------------------------------------------------------------------
    def run(self, params: Dict) -> str:
        mode = str(params.get("mode") or "").strip().lower()
        # 兼容 [TOOL_CALL:recall_tool:实习] 这种只给 query 的写法
        raw = str(params.get("input") or "").strip()
        if not mode:
            if raw in ("pending", "pattern", "progress", "recall"):
                mode = raw
            else:
                mode = "recall"
        if mode == "recall" and not (params.get("query") or raw) and raw == mode:
            mode = "pending"

        if mode == "pending":
            return self._pending()
        if mode == "pattern":
            return self._pattern(params)
        if mode == "progress":
            return self._progress()
        query = str(params.get("query") or raw or "").strip()
        return self._recall(query, params)

    # ---------------------------------------------------------- recall
    def _recall(self, query: str, params: Dict) -> str:
        if not query:
            return ("需要提供 query（要检索的话题或关键词），"
                    "例如 [TOOL_CALL:recall_tool:mode=recall,query=实习]。")
        if not getattr(config, "MEMORY_FTA_ENABLED", True):
            return "情境记忆已关闭（config.MEMORY_FTA_ENABLED=0）。"
        try:
            top_k = int(params.get("top_k") or config.MEMORY_FTA_TOP_K)
        except (TypeError, ValueError):
            top_k = config.MEMORY_FTA_TOP_K

        try:
            units = self._memory.retrieve_fta(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001 工具失败不应中断对话
            return f"（记忆检索暂不可用：{e}）"

        if not units:
            return (f"关于「{query}」没有检索到相关记忆。"
                    "如果用户在聊这个，就当作新话题认真听，"
                    "**不要**假装记得（不允许编造记忆）。")

        lines = [f"关于「{query}」检索到 {len(units)} 条相关情境记忆"
                 f"（按相关度排序，供你参考，不要逐条复述）:"]
        ids = []
        for i, u in enumerate(units, 1):
            uid = u.get("id")
            if uid is not None:
                ids.append(uid)
            lines.append(f"\n{i}. {self._format_unit(u)}")
            # 关联记忆（时序链接）：让"后来怎么样了"能接上
            if uid is not None:
                try:
                    linked = self._memory.get_linked(uid, limit=2)
                except Exception:  # noqa: BLE001
                    linked = []
                for lk in linked:
                    rel = {"updates": "更新了", "contradicts": "与此前说法不一致",
                           "supports": "印证了", "followup_of": "是后续发展",
                           "same_episode_as": "同属一件事"}.get(
                               lk.get("relation"), lk.get("relation", "关联"))
                    lines.append(f"   ↳ 关联（{rel}）: {lk.get('fact', '')}")

        # 召回即强化（文献4：S += 1 并重置衰减计时）
        if ids:
            try:
                self._memory.touch_recall(ids)
            except Exception:  # noqa: BLE001
                pass

        lines.append("\n使用建议: 若用户正在聊这个话题，可以自然地问一句后续"
                     "（“后来联系上了吗？”）；**不要**一次抛出多条，"
                     "也不要让用户觉得你在“调档案”。")
        return "\n".join(lines)

    def _format_unit(self, u: Dict) -> str:
        """把 FTA 五元组渲染成给模型看的一行"""
        fact = u.get("fact_text") or u.get("fact") or ""
        parts = [fact]
        stype = SITUATION_ZH.get(u.get("situation_type", ""), u.get("situation_type", ""))
        if stype:
            parts.append(f"[{stype}]")
        t = TIME_ZH.get(u.get("time_orientation", ""), "")
        if t and t != "未明确":
            parts.append(f"（{t}）")
        created = u.get("created_ts")
        if created:
            try:
                parts.append(datetime.fromtimestamp(float(created)).strftime("（%m-%d）"))
            except (TypeError, ValueError, OSError):
                pass
        emo = u.get("emotion") or ""
        if emo:
            conf = u.get("emotion_conf")
            if conf is not None and float(conf) < _LOW_AFFECT_CONF:
                # 文献5 消融实验：情感锚最不可靠，低置信时必须标注
                parts.append(f"（当时生理提示「{emo}」，但信号可信度低，仅供参考）")
            elif conf is not None:
                parts.append(f"（当时情绪「{emo}」）")
            else:
                parts.append(f"（当时情绪「{emo}」）")
        if u.get("lifecycle") == "pending" or u.get("completeness") == "partial":
            parts.append("⚑ 这件事当时**没有结论**，适合问后续")
        ret = u.get("retention")
        if ret is not None and float(ret) < 0.5:
            parts.append("（记忆较久远，细节可能有偏差 —— 不确定时说“我可能记错了”）")
        return " ".join(p for p in parts if p)

    # --------------------------------------------------------- pending
    def _pending(self) -> str:
        try:
            units = self._memory.get_pending_units(limit=3)
        except Exception as e:  # noqa: BLE001
            return f"（读取未完结事项失败：{e}）"
        if not units:
            return ("没有记录在案的“未完结事项”。"
                    "如果用户提到某件还没结果的事，可以用 memory_tool 记下来，"
                    "并在之后自然地关心后续。")
        lines = ["用户提到过、但**没有结论**的事（适合用来自然追问后续）:"]
        for i, u in enumerate(units, 1):
            stype = SITUATION_ZH.get(u.get("situation_type", ""), "")
            lines.append(f"{i}. {u.get('fact_text', '')}"
                         + (f"  [{stype}]" if stype else ""))
        lines.append("\n用法: 挑**至多一条**与当前话题最相关的，用不经意的方式问，"
                     "例如“对了，你之前提过的那件事，后来怎么样了？”；"
                     "如果用户不想聊，立刻放下，不要追问。")
        return "\n".join(lines)

    # --------------------------------------------------------- pattern
    def _pattern(self, params: Dict) -> str:
        try:
            days = float(params.get("days") or 7)
        except (TypeError, ValueError):
            days = 7.0
        try:
            timeline = self._memory.get_emotional_timeline(days=days)
            dist = self._memory.emotion_distribution(days=days)
        except Exception as e:  # noqa: BLE001
            return f"（读取情绪时间线失败：{e}）"

        if not timeline:
            return (f"最近 {days:g} 天还没有足够的情绪记录可用于模式分析。"
                    "可以先正常陪伴，等数据积累后再做回顾。")

        lines = [f"【情绪模式洞察】基于这位用户最近 {days:g} 天、"
                 f"{len(timeline)} 个状态样本（其本人数据，非通用结论）"]

        # --- 主导情绪分布 ---
        total = sum(dist.values()) or 1
        top = sorted(dist.items(), key=lambda kv: -kv[1])[:4]
        lines.append("  · 情绪分布: " + "、".join(
            f"{k} {v / total * 100:.0f}%" for k, v in top if k))

        # --- 高发时段（把"什么时候更难受"说清楚）---
        buckets = Counter()
        risk_buckets = Counter()
        for r in timeline:
            ts = r.get("ts_epoch")
            if not ts:
                continue
            try:
                hour = datetime.fromtimestamp(float(ts)).hour
            except (TypeError, ValueError, OSError):
                continue
            slot = _slot_of(hour)
            buckets[slot] += 1
            try:
                if float(r.get("risk") or 0) >= config.TRIGGER_LOW:
                    risk_buckets[slot] += 1
            except (TypeError, ValueError):
                pass
        if risk_buckets:
            worst = sorted(risk_buckets.items(), key=lambda kv: -kv[1])[:2]
            lines.append("  · 高压时段（风险≥%d 的样本最集中）: %s"
                         % (config.TRIGGER_LOW,
                            "、".join(f"{k}（{v}次）" for k, v in worst)))
        elif buckets:
            busiest = sorted(buckets.items(), key=lambda kv: -kv[1])[:2]
            lines.append("  · 记录最集中的时段: "
                         + "、".join(f"{k}（{v}次）" for k, v in busiest))

        # --- 高风险占比 ---
        try:
            hi = sum(1 for r in timeline
                     if float(r.get("risk") or 0) >= config.TRIGGER_HIGH)
            if hi:
                lines.append(f"  · 达到紧急关怀线（≥{config.TRIGGER_HIGH}）的样本: "
                             f"{hi} 个（占 {hi / len(timeline) * 100:.0f}%）"
                             "—— 若集中出现，说明存在反复的高压情境")
        except (TypeError, ValueError):
            pass

        # --- 个体触发点（从情境记忆类型统计）---
        triggers = self._trigger_sources(days)
        if triggers:
            lines.append("  · 反复出现的困扰类型: "
                         + "、".join(f"{k}（{v}次）" for k, v in triggers))

        # --- 信号质量提醒（避免拿不可靠数据下结论）---
        bad = 0
        for r in timeline:
            lv = r.get("quality_level") or ""
            if lv in ("不可用", "poor"):
                bad += 1
        if bad:
            lines.append(f"  · 注意: {bad}/{len(timeline)} 个样本信号质量不可用，"
                         "上述统计已受其影响 —— 用时留有余地，别说得太肯定。")

        lines.append("\n表达方式: 用“我注意到”这类观察口吻，而不是下结论；"
                     "可以说“好像晚上你更容易紧绷”，然后**问**用户"
                     "“你觉得是这样吗？”—— 用户的感受永远优先于统计。")
        return "\n".join(lines)

    def _trigger_sources(self, days: float) -> List[tuple]:
        """从 FTA 单元统计反复出现的困扰类型"""
        try:
            rows = self._memory.retrieve_fta("", top_k=200, tau=0.0,
                                             expand_links=False)
        except Exception:  # noqa: BLE001
            return []
        counter: Counter = Counter()
        for u in rows:
            st = u.get("situation_type") or ""
            if st in ("homework_or_plan", "relationship_event", "risk_signal"):
                counter[SITUATION_ZH.get(st, st)] += 1
        return [kv for kv in counter.most_common(3) if kv[1] >= 2]

    # -------------------------------------------------------- progress
    def _progress(self) -> str:
        try:
            stats = self._memory.strategy_stats_persisted()
        except Exception as e:  # noqa: BLE001
            return f"（读取干预反馈失败：{e}）"
        if not stats:
            return ("还没有足够的干预反馈记录。"
                    "在做过呼吸/安抚等干预后，可以问一句“现在感觉好一点了吗”，"
                    "并根据回答用 feedback 记录有效性，之后就能看出哪些方法对ta有效。")

        from ..emotion_strategy import STRATEGIES
        rows = []
        for name, s in stats.items():
            tot = s.get("total", 0)
            if tot <= 0:
                continue
            rate = s.get("effective", 0) / tot
            label = (STRATEGIES.get(name) or {}).get("name", name)
            rows.append((rate, tot, label, name))
        rows.sort(reverse=True)
        lines = ["【干预有效性回顾】这位用户本人的反馈统计:"] 
        for rate, tot, label, name in rows:
            verdict = ("比较有效" if rate >= 0.6 else
                       "效果一般" if rate >= 0.35 else "效果不佳")
            lines.append(f"  · {label}（{name}）: {rate * 100:.0f}% 有效"
                         f"（{tot} 次）→ {verdict}")
        best = rows[0]
        if best[0] >= 0.6 and best[1] >= 2:
            lines.append(f"\n建议: 优先使用「{best[2]}」—— 对这位用户已验证有效。")
        weak = [r for r in rows if r[0] < 0.35 and r[1] >= 2]
        if weak:
            lines.append("可减少使用: " + "、".join(f"「{r[2]}」" for r in weak)
                         + "（此前效果欠佳，或换一种方式）")
        lines.append("\n注意: 这是相关性不是因果（用户状态本来就会自己波动），"
                     "把结论说成“之前试过这个，你好像感觉好一些”，"
                     "而不是“这个方法对你有用”。")
        return "\n".join(lines)


def _slot_of(hour: int) -> str:
    """把小时映射到人话时段（避免生硬地报"22点"）"""
    if 5 <= hour < 9:
        return "清晨(5-9)"
    if 9 <= hour < 12:
        return "上午(9-12)"
    if 12 <= hour < 14:
        return "午间(12-14)"
    if 14 <= hour < 18:
        return "下午(14-18)"
    if 18 <= hour < 22:
        return "晚间(18-22)"
    return "深夜(22-5)"
