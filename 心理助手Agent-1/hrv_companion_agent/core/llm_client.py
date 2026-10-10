# -*- coding: utf-8 -*-
"""
LLM 客户端（三层降级策略）
==========================
优先:  hello_agents.HelloAgentsLLM（项目指定框架，章节7 my_llm.py 同源）
备选:  openai SDK 直连（任意 OpenAI 兼容服务）
兜底:  离线规则话术库（无 API Key 时也能跑通完整闭环演示）

统一接口: invoke(messages) -> str / stream_invoke(messages) -> Iterator[str]
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

from .. import config
from ..agent.prompts import OfflineBrain

logger = logging.getLogger("companion.llm")


class CompanionLLM:
    """情感陪伴 Agent 的大脑（在线 LLM + 离线降级）"""

    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None):
        self.model = model or config.LLM_MODEL_ID
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url or config.LLM_BASE_URL
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)
        # 记住基准温度：clear_overrides() 要能把被调制过的温度复原，
        # 否则一次"高焦虑低温"会永久泄漏到之后所有无关对话。
        self._base_temperature = self.temperature
        self.max_tokens = max_tokens or config.LLM_MAX_TOKENS
        self.timeout = timeout or config.LLM_TIMEOUT

        self._backend = None          # 'hello_agents' | 'openai' | 'offline'
        self._client = None
        self._brain = OfflineBrain()  # 离线话术脑（带上下文，避免重复机械）
        # v2: 运行时生成参数覆盖（情绪强度 + 心境 → 动态调制，见
        # agent/emotion_strategy.modulate_llm_params）。为空时行为与 v1 完全一致。
        self._overrides: Dict[str, float] = {}
        # v2: 最近一次实际生效的生成参数（可观测性 —— 便于验证调制是否真的下发）
        self._last_params: Dict[str, float] = {}
        self._init_backend()

    # ------------------------------------------------- 运行时参数覆盖(v2)
    def set_overrides(self, **params) -> Dict[str, float]:
        """设置一次性生成参数覆盖（temperature / top_p / presence_penalty …）

        设计为"随用随设、用完即清"：调用方（HeartCompanionSystem）在每轮
        生成前按当前情绪设置，生成后 ``clear_overrides()``。
        这样既做到了逐轮自适应，又不会让参数"泄漏"到后续无关调用。
        """
        for k, v in params.items():
            if v is None:
                self._overrides.pop(k, None)
            else:
                self._overrides[k] = float(v)
        # hello_agents 后端只能通过实例属性传温度，故同步一次
        if "temperature" in self._overrides:
            try:
                self.temperature = float(self._overrides["temperature"])
            except (TypeError, ValueError):
                pass
        return dict(self._overrides)

    def clear_overrides(self):
        """清除覆盖，回到配置默认值（含把温度复原到基准值）"""
        self._overrides = {}
        self.temperature = self._base_temperature

    @property
    def last_params(self) -> Dict[str, float]:
        """最近一次实际下发的生成参数（供 /api/state 观测）"""
        return dict(self._last_params)

    def _effective_params(self) -> Dict[str, Optional[float]]:
        p: Dict[str, Optional[float]] = {
            "temperature": self.temperature,
            "top_p": None,
            "presence_penalty": None,
        }
        p.update(self._overrides)
        self._last_params = {k: v for k, v in p.items() if v is not None}
        return p

    # ------------------------------------------------------------ 初始化
    def _init_backend(self):
        if not self.api_key:
            self._backend = "offline"
            logger.info("未配置 LLM_API_KEY → 使用离线规则话术库（演示模式）")
            return
        # 首选 openai SDK 直连（输出干净、无调试打印）；
        # hello_agents.HelloAgentsLLM 会向 stdout 打印调试信息并重复输出，
        # 因此仅保留 hello_agents 作为 Agent 框架（SimpleAgent/ToolRegistry），
        # LLM 通道统一走 openai SDK。
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout)
            self._backend = "openai"
            logger.info("LLM 后端: openai SDK (%s)", self.model)
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("openai SDK 不可用(%s)，尝试 hello_agents", e)
        try:
            from hello_agents import HelloAgentsLLM
            self._client = HelloAgentsLLM(
                model=self.model, api_key=self.api_key, base_url=self.base_url,
                provider=config.LLM_PROVIDER or "auto",
                temperature=self.temperature, max_tokens=self.max_tokens,
                timeout=self.timeout)
            self._backend = "hello_agents"
            logger.info("LLM 后端: hello_agents.HelloAgentsLLM (%s)", self.model)
            return
        except Exception as e:  # noqa: BLE001
            logger.warning("hello_agents 初始化失败(%s) → 离线模式", e)
            self._backend = "offline"

    @property
    def online(self) -> bool:
        return self._backend != "offline"

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------ 健康检查
    def health_check(self) -> tuple:
        """
        返回 (状态说明, 详细信息)。做一次 1-token 真实对话探测，提前暴露
        模型名/Key 配置错误；不抛异常，始终可用。
        """
        if self._backend == "offline":
            return ("离线演示模式", "未检测到 API Key。可在 .env 中填写 LLM_API_KEY "
                                     "或 DEEPSEEK_API_KEY 后重启升级为真实大模型对话。")
        detail = f"模型 {self.model} @ {self.base_url}"
        probe = [{"role": "system", "content": "只回复一个字：好"},
                 {"role": "user", "content": "测试"}]
        try:
            out = self._raw_chat(probe)
        except Exception as e:  # noqa: BLE001
            if self._autocorrect_model(e):
                detail = f"模型已自动修正为 {self.model} @ {self.base_url}"
                try:
                    out = self._raw_chat(probe)
                except Exception as e2:  # noqa: BLE001
                    return ("在线但对话失败", detail + f"（{e2}）")
            else:
                return ("在线但对话失败", detail + f"（{e}；请检查 Key/模型名/网络）")
        if out:
            return ("在线", detail + "（连通性与模型验证 ✓）")
        return ("在线", detail + "（模型返回为空，请检查配置）")

    # ------------------------------------------------------------ 推理
    def _raw_chat(self, messages: List[dict], stream: bool = False):
        """真正调用 LLM 后端，失败时抛出异常（供健康检查/纠错使用）

        v2: 下发生成参数时会合并 ``set_overrides()`` 设置的覆盖值。
        若服务端不接受 top_p / presence_penalty（部分模型确有此限制），
        自动降级为"只带 temperature"重试一次 —— 调制是增强项，
        绝不能因为某个参数不被支持而让整轮对话失败。
        """
        if self._backend == "hello_agents":
            if stream:
                return self._client.stream_invoke(messages)
            return self._client.invoke(messages)

        params = self._effective_params()
        kw = {"model": self.model, "messages": messages,
              "temperature": params["temperature"],
              "max_tokens": self.max_tokens}
        for extra in ("top_p", "presence_penalty", "frequency_penalty"):
            if params.get(extra) is not None:
                kw[extra] = params[extra]
        try:
            if stream:
                return self._client.chat.completions.create(stream=True, **kw)
            resp = self._client.chat.completions.create(stream=False, **kw)
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            extra_keys = [k for k in ("top_p", "presence_penalty",
                                      "frequency_penalty") if k in kw]
            if not extra_keys:
                raise
            logger.info("服务端拒绝扩展参数 %s（%s）→ 降级为仅 temperature 重试",
                        extra_keys, e)
            for k in extra_keys:
                kw.pop(k, None)
            self._last_params = {k: v for k, v in kw.items()
                                 if k not in ("model", "messages", "stream")}
            if stream:
                return self._client.chat.completions.create(stream=True, **kw)
            resp = self._client.chat.completions.create(stream=False, **kw)
            return resp.choices[0].message.content or ""

    @staticmethod
    def _is_model_error(err: Exception) -> bool:
        s = str(err).lower()
        return ("model" in s and any(k in s for k in
                ("invalid", "not found", "not exist", "support", "unknown")))

    def _autocorrect_model(self, err: Exception) -> bool:
        """模型名配置错误时，按候选列表自动切换并重建客户端"""
        if not self._is_model_error(err):
            return False
        candidates = ["deepseek-v4-flash", "deepseek-v4-pro",
                      "deepseek-chat", "deepseek-reasoner"]
        for cand in candidates:
            if cand == self.model:
                continue
            old, self.model = self.model, cand
            try:
                self._init_backend()
                logger.warning("模型名 %s 无效，已自动切换为 %s", old, cand)
                return True
            except Exception:  # noqa: BLE001
                self.model = old
        return False

    def invoke(self, messages: List[dict]) -> str:
        """非流式推理，返回完整回复；失败自动纠错一次后降级离线话术"""
        if self._backend == "offline":
            return self._offline_invoke(messages)
        try:
            return self._raw_chat(messages)
        except Exception as e:  # noqa: BLE001
            if self._autocorrect_model(e):
                try:
                    return self._raw_chat(messages)
                except Exception as e2:  # noqa: BLE001
                    logger.error("LLM 调用失败(纠错后): %s → 本次降级离线回复", e2)
            else:
                logger.error("LLM 调用失败: %s → 本次降级离线回复", e)
            return self._offline_invoke(messages)

    def stream_invoke(self, messages: List[dict]) -> Iterator[str]:
        """流式推理（逐段 yield，供 WebSocket/前端打字效果）"""
        if self._backend == "offline":
            yield self._offline_invoke(messages)
            return
        try:
            if self._backend == "hello_agents":
                for chunk in self._client.stream_invoke(messages):
                    if chunk:
                        yield chunk
                return
            stream = self._raw_chat(messages, stream=True)
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001
            if self._autocorrect_model(e):
                try:
                    yield self._raw_chat(messages)   # 纠错后非流式兜底
                    return
                except Exception as e2:  # noqa: BLE001
                    logger.error("LLM 流式调用失败(纠错后): %s", e2)
            else:
                logger.error("LLM 流式调用失败: %s → 降级离线回复", e)
            yield self._offline_invoke(messages)

    # ------------------------------------------------------------ 离线兜底
    def _offline_invoke(self, messages: List[dict]) -> str:
        """离线模式: 取最后一条 user 消息 → 规则话术脑回复（带上下文）"""
        import re
        user_text = ""
        state = None
        is_proactive = False
        opener = ""
        for msg in reversed(messages):
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            role = msg.get("role", "") if isinstance(msg, dict) else ""
            if role == "user":
                # 注意: 标记为 "[系统主动触发 - 检测到情绪异常]"，只匹配前缀
                if "[系统主动触发" in content:
                    # 主动触发指令: 提取风险/情绪/开场白 → 走 proactive 话术
                    is_proactive = True
                    m_risk = re.search(r"风险评分:\s*([\d.]+)", content)
                    m_emo = re.search(r"情绪状态:\s*(\S+)", content)
                    m_opener = re.search(r"开场白参考:\s*(.+)", content)
                    state = {"risk_score": float(m_risk.group(1)) if m_risk else 0,
                             "emotion": m_emo.group(1) if m_emo else ""}
                    opener = m_opener.group(1).strip() if m_opener else ""
                elif "[系统感知]" in content:
                    # 情绪感知上下文: 提取紧张程度/情绪，剥离注入块取真实消息
                    m_risk = re.search(r"紧张程度:\s*([\d.]+)", content)
                    m_emo = re.search(r"情绪状态:\s*(\S+)", content)
                    state = {"risk_score": float(m_risk.group(1)) if m_risk else 0,
                             "emotion": m_emo.group(1) if m_emo else ""}
                    lines = content.split("\n")
                    user_text = next((l for l in reversed(lines)
                                      if l.strip() and not l.startswith("[")), "")
                else:
                    user_text = content
                break
        if is_proactive:
            return self._brain.proactive(state, opener)
        return self._brain.respond(user_text, state)
