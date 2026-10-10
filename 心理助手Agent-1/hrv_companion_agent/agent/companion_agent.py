# -*- coding: utf-8 -*-
"""
主动式情感陪伴 Agent 核心实现
==============================
对应计划书 §4.3 companion_agent.py。

继承关系（对齐 HelloAgents 框架章节7 MySimpleAgent 架构模式）:
  hello_agents.SimpleAgent（框架可用时继承）→ CompanionAgent
框架不可用时自动退化为独立实现（外部接口完全一致）。

功能:
  1. 感知用户实时情绪状态（[系统感知] 上下文注入）
  2. 情绪自适应回复策略（风格指引动态注入 system prompt）
  3. 长期记忆驱动的个性化陪伴（画像 + 最近互动 + 情绪趋势注入）
  4. 主动发起对话（proactive_trigger，非被动等待用户输入）
  5. **九大工具调用循环**（[TOOL_CALL:...] 协议，v2 扩展为分层工具矩阵）
  6. 危机文本检测 → 自动启用分级危机干预协议
  7. **v2 生成参数调制**（文献3 式(21)–(26)：情绪强度 + 心境 → temperature/top_p）

v2 的工具装配与 v1 的差别
--------------------------
v1 只装配 5 个无状态工具（情绪/安抚/呼吸/危机/记忆）。
v2 有 9 个，其中 4 个需要外部数据源，故装配时需要注入回调：

    emotion_tool / pad_tool   ← get_current_state（实时状态）
    pad_tool                  ← memory.get_baseline（个体基线）
    comfort_tool              ← 策略反馈统计 + 最近用过的技能（去重）
    recall_tool / memory_tool ← memory（FTA 情境记忆）
    self_report_tool          ← memory + get_current_state（主客观配对）

这样做的目的不是「工具越多越好」，而是让每个工具只负责一件
**它自己知道而模型不知道**的事（信号是否可信、这个技术对ta有没有效、
这件事当时有没有结论）。
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterator, List, Optional

from .. import config
from ..core.llm_client import CompanionLLM
from . import emotion_strategy, prompts
from .memory_manager import MemoryManager
from .tools import (BreathingGuideTool, ComfortTool, CrisisTool, EmotionTool,
                    MemoryTool, PADTool, RecallTool, ReferralTool,
                    SelfReportTool, SimpleToolRegistry, detect_level)

logger = logging.getLogger("companion.agent")

# ---- 框架兼容: hello_agents 可用则继承其 SimpleAgent ----
try:
    from hello_agents import SimpleAgent as _HelloSimpleAgent
    _HAS_HELLO_AGENTS = True
except ImportError:  # pragma: no cover
    _HelloSimpleAgent = object
    _HAS_HELLO_AGENTS = False

TOOL_CALL_PATTERN = re.compile(r"\[TOOL_CALL:([^:]+):([^\]]+)\]")


class CompanionAgent(_HelloSimpleAgent):
    """主动式情感陪伴 Agent —— 小暖"""

    def __init__(self, user_id: str = None, llm: Optional[CompanionLLM] = None,
                 memory: Optional[MemoryManager] = None,
                 trigger_engine=None, system_prompt: Optional[str] = None):
        self.user_id = user_id or config.DEFAULT_USER_ID
        self.name = config.COMPANION_NAME
        self.system_prompt = system_prompt or prompts.COMPANION_SYSTEM_PROMPT
        self.llm = llm or CompanionLLM()
        self.memory = memory or MemoryManager(self.user_id)
        self.trigger_engine = trigger_engine

        # 继承框架基类（可用时），保持与 HelloAgents 生态的兼容性
        if _HAS_HELLO_AGENTS:
            try:
                super().__init__(name=self.name, llm=self.llm,
                                 system_prompt=self.system_prompt)
            except Exception as e:  # noqa: BLE001 基类校验失败不致命
                logger.debug("hello_agents 基类初始化跳过: %s", e)
        self._history: List[Dict[str, str]] = []
        self._current_state: Optional[Dict] = None
        #: 最近用过的心理技能 id（comfort_tool 的新鲜度惩罚依据）
        self._recent_skills: List[str] = []
        #: 本次对话实际使用的技能 id（供 system.feedback 落库，形成技能级闭环）
        self._pending_skills: List[str] = []
        #: 工具实例引用（部分工具的状态需在回合间保留）
        self._tool_refs: Dict[str, object] = {}
        #: 本轮是否为危机回合（crisis_tool 被调用过）
        self._crisis_turn: bool = False
        #: 上一轮**回复时**的情绪（用于生成转场提示，见 run_with_emotion）
        self._prev_reply_emotion: str = ""
        #: 本轮的情绪转场提示（由 run_with_emotion 设置）
        self._transition_hint: str = ""

        # 工具注册（对齐计划书 §4.3 _setup_tools）
        self.tool_registry = self._create_registry()
        self.enable_tool_calling = True
        self._setup_tools()
        logger.info("✅ %s 初始化完成（框架: %s | LLM: %s | 工具: %s）",
                    self.name,
                    "hello_agents" if _HAS_HELLO_AGENTS else "内置",
                    self.llm.backend,
                    self.tool_registry.list_tools())

    # ------------------------------------------------------------ 工具
    @staticmethod
    def _create_registry():
        try:
            from hello_agents import ToolRegistry  # 优先框架注册表
            return ToolRegistry()
        except ImportError:  # pragma: no cover
            return SimpleToolRegistry()

    def _setup_tools(self):
        """装配 v2 工具矩阵（9 个，按 感知→干预→校准→记忆→安全 分层）"""
        # ---- 感知层 ----
        emotion_tool = EmotionTool(self.get_current_state)
        pad_tool = PADTool(state_provider=self.get_current_state,
                           baseline_provider=self.memory.get_baseline)
        self.tool_registry.register_tool(emotion_tool)
        self.tool_registry.register_tool(pad_tool)

        # ---- 干预层 ----
        comfort = ComfortTool(
            skill_stats_provider=self._skill_stats,
            recent_provider=lambda: list(self._recent_skills))
        breathing = BreathingGuideTool()
        self.tool_registry.register_tool(comfort)
        self.tool_registry.register_tool(breathing)

        # ---- 校准层（主观自评：大创实验真值来源）----
        self_report = SelfReportTool(self.memory, self.user_id,
                                     state_provider=self.get_current_state)
        self.tool_registry.register_tool(self_report)

        # ---- 记忆层 ----
        mem_tool = MemoryTool(self.memory, self.user_id)
        # 让 memory_tool 写入时能带上当时的生理锚（FTA 的 physio_json）
        mem_tool.set_state_provider(self.get_current_state)
        recall = RecallTool(self.memory, self.user_id)
        self.tool_registry.register_tool(mem_tool)
        self.tool_registry.register_tool(recall)

        # ---- 安全层 ----
        crisis = CrisisTool()
        referral = ReferralTool()
        self.tool_registry.register_tool(crisis)
        self.tool_registry.register_tool(referral)

        # 全部留引用：除了 comfort_tool 需要在回合间保留 last_ids，
        # 其余工具也要能被 API 层「单独问一次」（见 run_tool）。
        for t in (emotion_tool, pad_tool, comfort, breathing, self_report,
                  mem_tool, recall, crisis, referral):
            self._tool_refs[t.name] = t

    # ------------------------------------------------------------------
    def run_tool(self, name: str, params: Optional[Dict] = None) -> str:
        """直接执行一个工具，不经过 LLM 的工具调用协议（v2 新增）

        用途：APP / API 层需要「查一次能力」—— 列出匹配到的心理技能、
        取一段呼吸脚本、问一次转介阶梯 —— 而不想因此触发一整轮生成。

        v1 没有这个入口，调用方只能去摸私有属性 ``_tool_refs``，
        工具改名就会**静默**失效（拿到 None 而不是报错）。这里做一个
        公开且会显式失败的口子。

        抛 ``KeyError`` 表示工具名不存在（附上可用清单，便于定位）。
        """
        tool = self._tool_refs.get(name)
        if tool is None:
            # 兜底：hello_agents 的注册表接口与内置版不完全一致
            for getter in ("get_tool",):
                fn = getattr(self.tool_registry, getter, None)
                if callable(fn):
                    try:
                        tool = fn(name)
                    except Exception:  # noqa: BLE001
                        tool = None
                if tool is not None:
                    break
        if tool is None:
            raise KeyError(f"未装配工具 '{name}'，可用: {sorted(self._tool_refs)}")
        return tool.run(dict(params or {}))

    def run_self_report(self, params: Dict) -> str:
        """执行主观自评工具（``POST /self-report`` 的落地入口）"""
        return self.run_tool("self_report_tool", params)

    # ------------------------------------------------------------------
    def _skill_stats(self) -> Dict[str, dict]:
        """技能级反馈统计（供 comfort_tool 打分 · 文献4 反馈闭环在技能粒度的落地）

        数据来源是 ``memory.feedback`` 表。该表按 ``strategy`` 字段分组，
        本系统在两种粒度上**复用同一张表**：

          * 系统主动干预      → strategy = 策略名（active_dialogue / breathing_guide…）
          * 对话中用了心理技能 → strategy = 技能 id（decatastrophizing / grounding_54321…）

        技能 id 由 ``_execute_tool_call`` 在 comfort_tool 执行后收集，
        由 ``HeartCompanionSystem.feedback()`` 与策略一并落库。
        这里只挑出**确实属于技能库的 id**，避免把策略名误当成技能。
        """
        stats: Dict[str, dict] = {}
        try:
            raw = self.memory.strategy_stats_persisted() or {}
        except Exception:  # noqa: BLE001 统计失败不应影响对话
            return stats
        try:
            from .tools.comfort_tool import all_skill_ids
            known = set(all_skill_ids())
        except Exception:  # noqa: BLE001
            return stats
        for k, v in raw.items():
            if k in known:
                stats[k] = {"total": int(v.get("total") or 0),
                            "effective": int(v.get("effective") or 0)}
        return stats

    #: 最近一次对话中实际使用过的技能 id（供 system.feedback 落库）
    @property
    def pending_skill_ids(self) -> List[str]:
        return list(self._pending_skills)

    def clear_pending_skills(self):
        self._pending_skills = []

    def add_tool(self, tool):
        self.tool_registry.register_tool(tool)
        if not self.enable_tool_calling:
            self.enable_tool_calling = True
        logger.info("🔧 工具 '%s' 已添加", getattr(tool, "name", tool))

    def remove_tool(self, tool_name: str) -> bool:
        try:
            self.tool_registry.unregister(tool_name)
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_tools(self) -> list:
        try:
            return self.tool_registry.list_tools()
        except AttributeError:  # pragma: no cover
            return []

    # ------------------------------------------------------------ 状态
    def set_state(self, state: Optional[Dict]):
        """注入当前生理-情绪状态（由 HeartCompanionSystem 每周期调用）"""
        self._current_state = state

    def get_current_state(self) -> Optional[Dict]:
        return self._current_state

    @staticmethod
    def _has_meaningful_state(state: Optional[Dict]) -> bool:
        """判断状态是否值得注入 [系统感知]（避免纯文本聊天带 N/A 块）

        实现委托给 ``prompts.is_usable_state`` —— 该判据必须只有一处定义：
        调用方判一次、注入函数再判一次（见 prompts 的说明），
        两份逻辑漂移时会出现"外面以为不注入、里面照样注入"的错位。
        """
        return prompts.is_usable_state(state)

    # ------------------------------------------------------------ 对话入口
    def run_with_emotion(self, message: str,
                         emotion_state: Optional[Dict] = None) -> str:
        """带情绪感知的对话接口（计划书 §4.3 run_with_emotion）

        v2 新增「情绪转场提示」：陪伴中最生硬的时刻，是用户情绪已经变了、
        AI 还按上一轮的语气说话。当识别结果与**上一轮回复时**的情绪不同，
        这里会生成一条转场指引注入 Prompt（见 emotion_strategy.strategy_for_shift）。

        注意比较对象是"上一轮回复时的情绪"，而不是状态流里的上一帧 ——
        后者是 1 秒级的抖动，用它做转场会每轮都在转场。
        """
        if emotion_state is not None:
            self.set_state(emotion_state)
        state = self._current_state or {}
        self._crisis_turn = False

        cur_emotion = state.get("emotion") or ""
        self._transition_hint = ""
        if self._prev_reply_emotion and cur_emotion and \
                self._prev_reply_emotion != cur_emotion:
            try:
                self._transition_hint = prompts.build_transition_hint(
                    self._prev_reply_emotion, cur_emotion)
            except Exception as e:  # noqa: BLE001 转场提示失败不影响回复
                logger.debug("转场提示生成失败: %s", e)
                self._transition_hint = ""

        if self._crisis_text(message):
            reply = self._crisis_response(message, state)
        elif self._has_meaningful_state(state):
            enhanced = prompts.build_emotion_context(message, state)
            reply = self.run(enhanced)
        else:
            reply = self.run(message)

        if cur_emotion:
            self._prev_reply_emotion = cur_emotion
        self.memory.save_interaction(message, reply, state)
        return reply

    def proactive_trigger(self, emotion_state: Optional[Dict] = None,
                          decision=None) -> str:
        """主动触发对话（计划书 §4.3 proactive_trigger）"""
        if emotion_state is not None:
            self.set_state(emotion_state)
        state = self._current_state or {}
        trigger_msg = prompts.build_proactive_message(state, decision)
        reply = self.run(trigger_msg)
        self.memory.save_interaction("[系统主动触发]", reply, state)
        return reply

    @staticmethod
    def _crisis_text(message: str) -> bool:
        """是否含危机信号（v2 用分级检测，覆盖面比 v1 的关键词表更广）"""
        return detect_level(message) is not None

    def _crisis_response(self, message: str, state: Dict) -> str:
        """危机文本 → 直接执行分级危机干预协议

        与 v1 的差别：v1 只判断"是不是危机"，v2 会把**分级结果**
        一并交给协议（被动意念 / 主动意念 / 有计划 / 已准备），
        因为四者的第一步完全不同（见 crisis_tool.LEVELS）。
        """
        level = detect_level(message) or ""
        crisis = CrisisTool()
        protocol = crisis.run({"text": message,
                               "risk": state.get("risk_score", 0),
                               "level": level})
        emotion = state.get("emotion") or "高焦虑"
        # 危机场景下风格指引固定走"高焦虑"档：不随识别结果波动，
        # 避免在危机回合里因为一个不可靠的读数把语气调轻。
        style = emotion_strategy.style_guide_for("高焦虑")
        affect = state.get("affect") or {}
        affect_note = emotion_strategy.describe_affect(affect)
        sys_msg = (self.system_prompt
                   + f"\n\n[当前状态指引]\n{style}")
        if affect_note:
            sys_msg += f"\n[情感动力学]\n{affect_note}"
        sys_msg += f"\n\n[危机干预协议 · 分级 {level or 'active'}]\n{protocol}"
        msgs = [{"role": "system", "content": sys_msg}]
        msgs.extend(self._memory_messages())
        msgs.append({"role": "user", "content": message})
        reply = self.llm.invoke(msgs)
        self._history.extend([{"role": "user", "content": message},
                              {"role": "assistant", "content": reply}])
        self._crisis_turn = True
        return reply

    # ------------------------------------------------------------ 消息构建
    def _memory_messages(self) -> List[Dict[str, str]]:
        """记忆注入（画像 + 最近互动摘要），作为 system 补充内容"""
        profile = self.memory.get_user_profile()
        recent = self.memory.get_recent_interactions(limit=config.RECENT_CONTEXT_N)
        timeline = self.memory.get_emotional_timeline(days=1)
        ctx = prompts.build_memory_context(profile, recent, timeline)
        if not ctx:
            return []
        return [{"role": "system", "content": ctx}]

    def _enhanced_system_prompt(self, tools_visible: bool = True) -> str:
        """基础角色 + 当前情绪风格指引 + 情绪转场 + 工具说明

        风格指引按「情绪 + 动力学强度级别 + 心境底色」生成（v2）：
        同样判定为"低焦虑"，强度 0.2 与 0.9 的说话方式不该一样。
        """
        parts = [self.system_prompt]
        state = self._current_state or {}
        if state.get("emotion"):
            affect = state.get("affect") or {}
            try:
                guide = emotion_strategy.style_guide_for(
                    state["emotion"], level=affect.get("level"), affect=affect)
            except Exception as e:  # noqa: BLE001
                logger.debug("风格指引生成失败，退回基础档: %s", e)
                guide = emotion_strategy.style_guide_for(state["emotion"])
            parts.append("[当前状态指引]\n" + guide)
        if self._transition_hint:
            parts.append("[情绪转场]\n" + self._transition_hint)
        parts.append(self._tools_section(tools_visible))
        return "\n\n".join(p for p in parts if p)

    def _tools_section(self, show: bool = True) -> str:
        """工具说明与调用协议。仅在本轮可能需要工具时注入，
        避免大模型在普通闲聊中输出未执行的 [TOOL_CALL] 标记。"""
        if not self.enable_tool_calling or not show:
            return ""
        try:
            desc = self.tool_registry.get_tools_description()
        except Exception:  # noqa: BLE001
            return ""
        if not desc:
            return ""
        return ("## 可用工具\n" + desc
                + "\n\n## 工具调用格式\n"
                  "确有必要时输出: `[TOOL_CALL:工具名:参数]`\n"
                  "例如: `[TOOL_CALL:emotion_tool:state]` 或 "
                  "`[TOOL_CALL:breathing_guide:type=relax]`\n"
                  "普通聊天请直接回复用户，不要调用工具。"
                  "工具结果会自动注入对话，随后继续回答。")

    # ------------------------------------------------------------ 主循环
    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """对话主循环（含工具调用循环，协议对齐章节7 MySimpleAgent）"""
        tools_visible = self._likely_needs_tools(input_text)
        messages: List[Dict[str, str]] = [
            {"role": "system",
             "content": self._enhanced_system_prompt(tools_visible)}]
        messages.extend(self._memory_messages())
        for msg in self._history[-12:]:
            messages.append(msg)
        messages.append({"role": "user", "content": input_text})

        if not self.enable_tool_calling or not tools_visible:
            reply = self._strip_tool_markers(self.llm.invoke(messages))
            self._history.extend([{"role": "user", "content": input_text},
                                  {"role": "assistant", "content": reply}])
            return reply

        final_response = ""
        for _ in range(max_tool_iterations):
            response = self.llm.invoke(messages)
            tool_calls = self._parse_tool_calls(response)
            if not tool_calls:
                final_response = response
                break
            clean = response
            results = []
            for call in tool_calls:
                result = self._execute_tool_call(call["tool_name"], call["parameters"])
                results.append(result)
                clean = clean.replace(call["original"], "")
            messages.append({"role": "assistant", "content": clean})
            messages.append({"role": "user",
                             "content": "工具执行结果：\n" + "\n\n".join(results)
                                        + "\n\n请基于这些结果给出完整的回答。"})
        if not final_response:
            final_response = self.llm.invoke(messages)

        self._history.extend([{"role": "user", "content": input_text},
                              {"role": "assistant", "content": final_response}])
        return final_response

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        流式对话（供 WebSocket / CLI 打字效果）。
        工具感知: 按启发式判断本轮是否需要工具——需要时才暴露工具并预探测，
        执行 [TOOL_CALL] 后生成最终回复；普通闲聊直接流式（无标记泄露）。
        """
        tools_visible = self.enable_tool_calling and self._likely_needs_tools(input_text)
        messages: List[Dict[str, str]] = [
            {"role": "system",
             "content": self._enhanced_system_prompt(tools_visible)}]
        messages.extend(self._memory_messages())
        for msg in self._history[-12:]:
            messages.append(msg)
        messages.append({"role": "user", "content": input_text})

        if tools_visible:
            # 非流式探测一轮: LLM 是否需要工具
            probe = self.llm.invoke(messages)
            tool_calls = self._parse_tool_calls(probe)
            if tool_calls:
                clean = probe
                results = []
                for call in tool_calls:
                    results.append(self._execute_tool_call(call["tool_name"],
                                                           call["parameters"]))
                    clean = clean.replace(call["original"], "")
                messages.append({"role": "assistant", "content": clean})
                messages.append({"role": "user",
                                 "content": "工具执行结果：\n" + "\n\n".join(results)
                                            + "\n\n请基于这些结果给出完整的回答。"})
                final = self._strip_tool_markers(self.llm.invoke(messages))
                self._history.extend([{"role": "user", "content": input_text},
                                      {"role": "assistant", "content": final}])
                yield final
                return
            # 探测结果不含工具调用 → 该轮按普通聊天流式处理
            for chunk in [probe]:
                if chunk:
                    full = self._strip_tool_markers(chunk)
                    self._history.extend([{"role": "user", "content": input_text},
                                          {"role": "assistant", "content": full}])
                    yield full
                    return

        full = ""
        for chunk in self.llm.stream_invoke(messages):
            full += chunk
            yield chunk
        full = self._strip_tool_markers(full)
        self._history.extend([{"role": "user", "content": input_text},
                              {"role": "assistant", "content": full}])

    @staticmethod
    def _strip_tool_markers(text: str) -> str:
        """移除回复中残留的 [TOOL_CALL:...] 标记（兜底保险）"""
        return TOOL_CALL_PATTERN.sub("", text or "").strip()

    def _likely_needs_tools(self, input_text: str) -> bool:
        """启发式判断本轮是否需要工具（决定是否把工具说明注入 Prompt）

        为什么要做这个门控：模型一旦看到工具说明就容易"过度调用"——
        普通闲聊也输出 [TOOL_CALL]，既浪费一次往返，也可能把
        "我查一下你的状态"这种系统口吻带进对话。因此只在**可能真需要**
        的轮次暴露工具。

        v2 扩展了触发词：新增转介、自评、记忆检索、技能练习四类场景。
        """
        t = input_text or ""
        keywords = (
            # 情绪/生理
            "呼吸", "放松", "紧张", "焦虑", "安抚", "安慰", "情绪",
            "心率", "心跳", "心慌", "喘", "状态", "冥想", "难受",
            # 记忆
            "记得", "记住", "记录", "上次", "之前说", "以前说", "后来",
            "忘了", "提醒我",
            # 技能 / 方法
            "怎么办", "方法", "技巧", "练习", "冷静", "睡不着", "失眠",
            "拖延", "不想动", "崩溃", "撑不住",
            # 转介与安全
            "医院", "医生", "心理中心", "咨询", "看心理", "吃药", "抑郁",
            "两周", "很久了", "一直没好", "活不下去", "没意思",
            # 自评
            "打个分", "自评", "感觉还好", "其实还好", "没那么严重",
        )
        if any(k in t for k in keywords):
            return True
        state = self._current_state or {}
        return state.get("emotion") in ("高焦虑", "低焦虑", "疲劳")

    # ------------------------------------------------------------ 工具循环
    def _parse_tool_calls(self, text: str) -> List[dict]:
        calls = []
        for tool_name, parameters in TOOL_CALL_PATTERN.findall(text or ""):
            calls.append({"tool_name": tool_name.strip(),
                          "parameters": parameters.strip(),
                          "original": f"[TOOL_CALL:{tool_name}:{parameters}]"})
        return calls

    def _execute_tool_call(self, tool_name: str, parameters: str) -> str:
        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                return f"❌ 错误：未找到工具 '{tool_name}'"
            param_dict = self._parse_tool_parameters(tool_name, parameters)
            result = tool.run(param_dict)
            self._after_tool(tool_name, tool, result)
            return f"🔧 工具 {tool_name} 执行结果：\n{result}"
        except Exception as e:  # noqa: BLE001 工具异常不得中断对话
            logger.debug("工具 %s 执行失败: %s", tool_name, e)
            return f"❌ 工具调用失败：{str(e)}"

    def _after_tool(self, tool_name: str, tool, result: str):
        """工具执行后的副作用收集（v2）

        * comfort_tool  → 记录匹配出的技能 id，用于
          ①下一轮的新鲜度惩罚（避免反复推同一种技术）
          ②反馈落库后的技能级有效率统计（文献4 反馈闭环）
        * memory_tool   → 若本次写入建立了链接，把它记进历史，便于
          "追问后续"时优先带出这条线。
        """
        if tool_name == "comfort_tool":
            ids = list(getattr(tool, "last_ids", None) or [])
            if ids:
                self._recent_skills = (self._recent_skills + ids)[-6:]
                self._pending_skills = ids
        elif tool_name == "crisis_tool":
            # 危机工具被调用 → 标记本轮为危机回合，供外层决定是否降级后续触发
            self._crisis_turn = True

    @staticmethod
    def _parse_tool_parameters(tool_name: str, parameters: str) -> dict:
        """解析 [TOOL_CALL:工具名:参数] 的参数部分

        支持两种写法:
          * ``key=value`` 形式（推荐，可多个逗号分隔）——
            例: [TOOL_CALL:comfort_tool:emotion=高焦虑,arousal=high]
          * 单值简写 —— 例: [TOOL_CALL:breathing_guide:calm]
            简写按工具映射到它最常用的那个参数名，避免模型对
            "这个位置该填什么"产生歧义（v1 的简写只覆盖 4 个工具，
            v2 补齐 9 个）。
        """
        param_dict: Dict[str, str] = {}
        if "=" in parameters:
            for pair in parameters.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    param_dict[k.strip()] = v.strip()
            return param_dict

        raw = parameters.strip()
        if not raw:
            return param_dict
        # ---- 单值简写：按工具映射到主参数 ----
        simple = {
            "emotion_tool": "input",
            "pad_tool": "dim",
            "comfort_tool": "emotion",
            "breathing_guide": "type",
            "crisis_tool": "text",
            "referral_tool": "text",
            "memory_tool": "fact",
        }
        if tool_name == "recall_tool":
            # 简写既可能是模式名，也可能是检索词
            if raw.lower() in ("recall", "pending", "pattern", "progress"):
                param_dict = {"mode": raw.lower()}
            else:
                param_dict = {"query": raw}
        elif tool_name == "self_report_tool":
            # 简写可能是评分（数字）也可能是情绪词
            try:
                float(raw)
                param_dict = {"score": raw}
            except ValueError:
                param_dict = {"emotion": raw}
        else:
            param_dict = {simple.get(tool_name, "input"): raw}
        return param_dict

    # ------------------------------------------------------------ 其他
    def clear_history(self):
        self._history = []

    def get_history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def close(self):
        self.memory.close()
