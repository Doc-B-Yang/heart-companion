# -*- coding: utf-8 -*-
"""
主动式情感陪伴系统 —— 核心闭环编排（v2）
========================================
对应计划书 §2 三层闭环架构中的"分析处理层 + AI生成输出层"编排。

完整数据流（v2）
--------------------------------------------------------------------------------
::

  BLE 数据包
     ↓
  HRV 特征提取 (HRVExtractor)          ← 时域/频域/非线性 + 个体基线相对化
     ↓
  信号质量门禁 (quality_metrics)        ← 【v2】jitter/伪迹率，不合格则不参与情绪判定
     ↓
  三层情绪识别 (EmotionRecognizer)      ← 规则 → 模糊 → ML
     ↓
  PAD 维度回归 (estimate_pad)           ← 【v2】连续维度，补离散标签的粒度损失
     ↓
  情感动力学 (EmotionDynamics)          ← 【v2】强度/持续/衰减 + 心境累积，解决情感跳变
     ↓
  记忆 (FTA 情境单元 + 时间线 + 基线)
     ↓
  主动触发决策 (TriggerEngine)          ← 【v2】心境慢变量 + 质量门禁 + 连续确认
     ↓ (触发)
  CompanionAgent 主动关怀（按强度/心境调制生成参数）
     ↓
  TTS 语音合成 → 用户反馈 → 策略/阈值自适应优化

v2 的三个关键设计决定（都直接来自文献与实测结论）
--------------------------------------------------------------------------------
1. **质量门禁前置**。实测确认 PPG 存在「HR 正常但 HRV 是假的」这一隐蔽失效
   （指腹接触不良时 RMSSD 130ms / jitter 0.455）。若把这种数据放进情绪识别
   并触发关怀，等于用噪声去打扰用户。故质量不合格时**降低置信度并抑制触发**。
2. **离散标签 + 连续维度并存**。离散标签用于决策阈值（可解释、稳定），
   PAD 连续值用于对话风格调制（有粒度）。两者互不替代。
3. **状态层（state）承载一切**。动力学快照、PAD、质量报告都并入 state，
   使 ``/api/state``、CLI 与 Prompt 注入共享同一份事实，避免多处各算一遍。

对外接口（v1 全部保留，向后兼容）
--------------------------------------------------------------------------------
  ingest_sensor(packet) / check_trigger() / tick(packet) / chat(message)
  / stream_chat(message) / feedback(...) / set_manual_emotion(...)
  / get_state() / get_timeline(days) / get_profile() / close()

v2 新增
--------------------------------------------------------------------------------
  get_affect() / set_personality(key) / export_data() / purge_data()
  / update_daily_summary() / quality_report()
"""

from __future__ import annotations

import logging
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional

from .. import config
from ..agent.companion_agent import CompanionAgent
from ..agent.memory_manager import MemoryManager
from ..agent import emotion_strategy
from ..agent.emotion_strategy import select_strategy, modulate_llm_params
from ..agent.tts_service import synthesize_sync
from ..algorithms import (EmotionDynamics, EmotionRecognizer, HRVExtractor,
                          PADRegressor, TriggerEngine, estimate_pad)
from .llm_client import CompanionLLM
from .sensor_pipeline import SensorPipeline, finite
from .sensor_calibration import SensorCalibration
from ..algorithms.multimodal_emotion import MultimodalEmotionRecognizer

logger = logging.getLogger("companion.system")


#: 值得写入 FTA 情境记忆的内容类型触发词（规则版抽取；有 LLM 时可升级）
_FTA_RULES: Dict[str, tuple] = {
    "risk_signal": ("不想活", "自杀", "自残", "割腕", "活着没意思", "结束生命",
                    "死了算了", "没有意义", "撑不下去"),
    "homework_or_plan": ("我决定", "我打算", "我要去", "计划", "明天就",
                         "从今天开始", "我答应", "试试看", "约好了"),
    "relationship_event": ("和我妈", "和我爸", "和室友", "和老师", "和导师",
                           "和男朋友", "和女朋友", "和朋友", "分手", "吵架"),
    "belief_candidate": ("其实不是", "我担心", "我怕", "我觉得自己", "可能是因为"),
    "event": ("考试", "面试", "答辩", "比赛", "汇报", "约了", "去了"),
}


class HeartCompanionSystem:
    """生理感知 → 情绪识别 → 主动干预 → 长期陪伴 的完整闭环系统"""

    def __init__(self, user_id: str = None, llm: Optional[CompanionLLM] = None,
                 ml_model_path: Optional[str] = None,
                 pad_model_path: Optional[str] = None,
                 personality: Optional[str] = None, **kwargs):
        self.user_id = user_id or config.DEFAULT_USER_ID
        self.started_at = time.time()

        # ---------------- 算法层 ----------------
        self.extractor = HRVExtractor(window_sec=config.HRV_WINDOW_SEC)
        self.recognizer = EmotionRecognizer(ml_model_path)
        self.multimodal = MultimodalEmotionRecognizer()
        self.sensor_pipeline = SensorPipeline()
        self.sensor_calibration = SensorCalibration()
        self._sensor_lock = threading.RLock()
        self._last_sensor_received = None
        self._sensor_flow = None
        self._last_trigger_sensor = None
        self.trigger = TriggerEngine()
        # v2: PAD 维度回归（未训练模型时自动走 RulePAD 规则路径）
        self.pad_regressor = PADRegressor(
            pad_model_path or config.PAD_MODEL_PATH)
        # v2: 情感动力学状态机（PWE / Plutchik 情绪轮）
        self.dynamics = EmotionDynamics(
            personality=personality or config.DYNAMICS_PERSONALITY)

        # ---------------- AI 层 ----------------
        self.llm = llm or CompanionLLM()
        self.memory = MemoryManager(self.user_id)
        self.agent = CompanionAgent(user_id=self.user_id, llm=self.llm,
                                    memory=self.memory,
                                    trigger_engine=self.trigger)

        # ---------------- 运行状态 ----------------
        self._state: Optional[Dict] = None
        self._last_packet: Optional[Dict] = None
        self._last_trigger_time: Optional[float] = None
        self._last_interaction_time: Optional[float] = None
        self._pending_strategy: Optional[str] = None
        self._pending_before_risk: Optional[float] = None
        self._ibi_window: List[float] = []   # 后端累积的 IBI 窗口（BLE 每包仅带 3 拍）
        self._packet_count: int = 0
        # v2 新增运行状态
        self._quality: Optional[Dict] = None
        self._pad = None
        self._last_affect_update: Optional[float] = None
        self._prev_emotion: str = ""
        self._last_fta_ts: float = 0.0
        self._summary_day: str = ""
        # 个体基线读缓存（PAD 与相对特征化共用，见 _cached_baseline）
        self._baseline_cache: Optional[Dict] = None
        self._baseline_cache_ts: float = 0.0

        # 策略有效率从库中恢复（进程重启后不丢失，v2 改进）
        try:
            for name, st in self.memory.strategy_stats_persisted().items():
                self.trigger.strategy_stats[name] = st
        except Exception as e:  # noqa: BLE001
            logger.debug("策略统计恢复跳过: %s", e)

        logger.info("🚀 HeartCompanionSystem v3 初始化完成（用户: %s，硬件ML: %s，"
                    "PAD-SVR: %s，动力学人格: %s）",
                    self.user_id,
                    "已加载" if self.multimodal.available else "未加载",
                    "已加载" if self.pad_regressor.available else "规则映射",
                    self.dynamics.personality.get("zh", "理想型"))

    # ==================================================================
    # 数据入口
    # ==================================================================
    def ingest_sensor(self, packet: Dict) -> Dict:
        with self._sensor_lock:
            return self._ingest_sensor_locked(packet)

    def _ingest_sensor_locked(self, packet: Dict) -> Dict:
        """接收 1 包 BLE 数据 → 质量门禁 → 特征提取 → 情绪识别
        → PAD → 情感动力学 → 记忆记录"""
        self._packet_count += 1
        if not isinstance(packet, dict) or not (config.ALLOW_SIMULATION and packet.get('_simulation') is True):
            return self._ingest_multimodal(packet)
        # 累积 IBI 窗口（BLE 每包仅携带最近 3 个心跳间隔）
        ibi_in = [float(v) for v in (packet.get("ibi") or [])
                  if isinstance(v, (int, float))]
        self._ibi_window.extend(ibi_in)
        if len(self._ibi_window) > 300:  # ≈5 分钟 @60bpm
            del self._ibi_window[:len(self._ibi_window) - 300]

        feats = self.extractor.extract_from_ble(packet)
        # 固件特征缺失时，用后端累积窗口重算（双保险）
        if len(self._ibi_window) >= 5:
            computed = self.extractor.extract_from_ibi(list(self._ibi_window))
            for key in ("sdnn", "rmssd", "pnn50", "lf_hf", "sd1_ms", "sd2_ms",
                        "sd_ratio", "hr_bpm", "sdsd_ms", "sdnn_index_ms",
                        "sd_hr_bpm", "sampen", "dfa_alpha1", "jitter"):
                if feats.get(key) in (None, 0.0) and computed.get(key):
                    feats[key] = computed[key]
            feats["n_beats"] = len(self._ibi_window)
        # 回落计算会直接改写短名键, 需重新同步长名（sdnn_ms/rmssd_ms/pnn50_pct）,
        # 否则规则层仍会读到 0 并静默跳过 SDNN/RMSSD —— 见 HRVExtractor 类注释。
        self.extractor.add_aliases(feats)

        # ---------------- v2: 信号质量门禁 ----------------
        quality = self._assess_quality(feats)
        self._quality = quality

        # 心跳样本不足且无固件特征 → 不输出情绪（避免小样本误判）；
        # 仅当长时间只有心率数据时（≥10 包）才允许估算兜底。
        has_firmware = bool(packet.get("sdnn") or packet.get("rmssd"))
        if len(self._ibi_window) < 5 and not has_firmware and self._packet_count < 10:
            state = {"emotion": "数据积累中", "confidence": 0.0,
                     "risk_score": 0.0, "layer": "insufficient_data",
                     "detail": {"n_beats": feats.get("n_beats", 0)},
                     "hr_bpm": feats.get("hr_bpm"),
                     "quality": quality,
                     "timestamp": packet.get("t") or time.time()}
            self._state = state
            self._last_packet = packet
            self.agent.set_state(state)
            return state

        # ---------------- 三层情绪识别 ----------------
        result = self.recognizer.recognize(feats)

        # ---------------- v2: PAD 维度回归 ----------------
        # 基线带缓存：PAD 估计与相对特征化都要用它，而上报频率是 1 包/秒。
        # 不缓存的话每秒会对 SQLite 发起多次相同查询（长时间监护下无谓开销）。
        baseline = self._cached_baseline()

        pad_result = None
        if config.PAD_ENABLED:
            try:
                pad_result = estimate_pad(feats, baseline, self.pad_regressor)
                self._pad = pad_result
            except Exception as e:  # noqa: BLE001
                logger.debug("PAD 估计失败（不影响主链路）: %s", e)

        # ---------------- v2: 相对个体基线特征（文献1 相对特征值法）----
        # 相对化后的特征既供 pad_tool 做归因，也供调节策略判断个体偏离。
        # relative_features 现在只返回**真正完成相对化**的键（见其 docstring
        # 的 warning），因此这里直接透传即可，不再需要按 RELATIVE_KEYS 过滤。
        rel_feats: Dict = {}
        if baseline:
            try:
                rel_feats = HRVExtractor.relative_features(feats, baseline)
            except Exception as e:  # noqa: BLE001
                logger.debug("相对特征化失败（不影响主链路）: %s", e)
                rel_feats = {}

        # ---------------- v2: 情感动力学推进 ----------------
        # 用三层识别给出的概率分布驱动动力学；无概率时用 one-hot 兜底。
        prob_zh = self._probabilities_of(result)
        affect = self._advance_dynamics(prob_zh, packet)

        # 质量不达标 → 置信度打折（让下游"知道这个数不太可信"）
        confidence = result.confidence
        if config.QUALITY_GATE_ENABLED and not quality.get("quality_ok", True):
            confidence = round(confidence * 0.5, 3)

        state = {
            "emotion": result.emotion,
            "confidence": confidence,
            "risk_score": result.risk_score,
            "layer": result.layer,
            "detail": result.detail,
            "hr_bpm": feats.get("hr_bpm"),
            "sdnn_ms": feats.get("sdnn"),
            "rmssd_ms": feats.get("rmssd"),
            "lf_hf": feats.get("lf_hf"),
            "pnn50": feats.get("pnn50"),
            "sd_hr_bpm": feats.get("sd_hr_bpm"),
            "rr_bpm": feats.get("rr_bpm"),
            "sig_quality": feats.get("sig_quality", 1.0),
            "jitter": feats.get("jitter") or quality.get("jitter"),
            "quality": quality,
            "quality_level": quality.get("quality_level"),
            "pad": pad_result.to_dict() if pad_result else None,
            "affect": affect.to_dict() if affect else None,
            # v2: 原始特征 + 相对基线特征（供工具层归因；
            # 不进 Prompt —— prompts.build_emotion_context 不读取这两个键）
            "feats": dict(feats),
            "rel_features": rel_feats,
            "timestamp": packet.get("t") or time.time(),
            "simulation": True,
            "validated": False,
        }
        self._state = state
        self._last_packet = packet
        self.agent.set_state(state)
        self.memory.record_emotion_state(state)
        # 统一键名后更新个体基线（EMA），并受质量门槛保护
        self.memory.update_baseline({
            "hr_bpm": feats.get("hr_bpm"),
            "sdnn_ms": feats.get("sdnn"),
            "rmssd_ms": feats.get("rmssd"),
            "lf_hf": feats.get("lf_hf"),
            "pnn50_pct": feats.get("pnn50"),
            "sd_hr_bpm": feats.get("sd_hr_bpm"),
            "sdnn_index_ms": feats.get("sdnn_index_ms"),
        }, quality=quality)

        # 情绪显著变化 → 落一条"状态记忆"（不是每包都写，避免污染记忆库）
        self._maybe_record_state_fta(state, feats)
        return state

    def _ingest_multimodal(self, packet: Dict) -> Dict:
        flow = self.sensor_pipeline.process(packet)
        self._sensor_flow = flow
        if flow.get('ignored') and self._state:
            # Duplicate/out-of-order frames do not refresh liveness or re-run ML.
            return self._state
        if flow['accepted']:
            self._last_sensor_received = time.monotonic()
        packet = packet if isinstance(packet, dict) else {}
        session = flow['session']
        self.sensor_calibration.bind((session.get('device_id'), session.get('boot_id')))
        clock_ms = packet.get('device_ms') if flow['accepted'] else (self.sensor_pipeline.device_ms or 0)
        calibration = self.sensor_calibration.status(clock_ms)
        feats = self.sensor_calibration.transform(flow['features'])
        quality = flow['quality']
        self._quality = quality
        if not quality['quality_ok']:
            self.multimodal.reset()
            result = self.multimodal.abstain(quality['reason'], 'quality_blocked')
        elif packet.get('_replay'):
            self.multimodal.reset()
            result = self.multimodal.abstain('历史回放仅用于链路验证，不能视为当前情绪', 'replay')
        elif not calibration['contact_ready'] or not calibration['baseline_ready']:
            self.multimodal.reset()
            reason = ('请确认GSR电极佩戴' if not calibration['contact_confirmed'] else
                      'GSR电极稳定等待中' if not calibration['contact_ready'] else
                      '请在平静静息时采集本次佩戴基线')
            result = self.multimodal.abstain(reason, 'calibration_required')
        else:
            result = self.multimodal.recognize(feats, packet['device_ms']/1000,
                         packet['seq'], (packet['device_id'], packet['boot_id'], packet['segment_id']))
        validated = bool(result.detail.get('validated'))
        state = {'emotion': result.emotion, 'confidence': result.confidence,
                 'risk_score': result.risk_score if validated else None,
                 'layer': result.layer, 'detail': result.detail,
                 'quality': quality, 'quality_level': quality['quality_level'],
                 'validated': validated, 'simulation': False,
                 'calibration': calibration,
                 'historical': bool(packet.get('_replay')),
                 'features': feats, 'feats': feats, 'sensor': session,
                 'timestamp': time.time(), 'source_device_ms': packet.get('device_ms') if finite(packet.get('device_ms')) else None,
                 'sig_quality': packet.get('sig_quality') if finite(packet.get('sig_quality')) else None, 'jitter': feats.get('jitter'),
                 'hr_bpm': feats.get('hr_bpm'), 'sdnn_ms': feats.get('sdnn_ms'),
                 'rmssd_ms': feats.get('rmssd_ms'), 'pnn50': feats.get('pnn50_pct'),
                 'lf_hf': None, 'rr_bpm': None, 'pad': None, 'affect': None,
                 'rel_features': {}, 'model_available': self.multimodal.available}
        # No rule-PAD conversion: HRV/EDA cannot identify valence/dominance alone.
        if validated:
            affect = self._advance_dynamics(result.detail['probabilities'], {'t':state['timestamp']})
            state['affect'] = affect.to_dict() if affect else None
            self.memory.record_emotion_state(state)
            self._maybe_record_state_fta(state, feats)
        self._state, self._last_packet = state, packet
        self.agent.set_state(state)
        return state

    def calibrate_sensor(self, action):
        """Explicit wearer observations, bound to the current fresh hardware boot."""
        with self._sensor_lock:
            state = self.get_state() or {}
            if state.get('simulation') is not False or state.get('historical') or state.get('layer') == 'stale' or not (self._sensor_flow or {}).get('accepted'):
                raise ValueError('请先连接并接收新鲜硬件数据')
            now = self.sensor_pipeline.device_ms
            if action == 'confirm_contact':
                self.sensor_calibration.confirm_contact(now)
                self.sensor_pipeline.clear(now)
            elif action == 'capture_baseline':
                self.sensor_calibration.capture(self._sensor_flow, now)
            elif action == 'reset':
                self.sensor_calibration.reset()
                self.sensor_pipeline.clear(now)
            else:
                raise ValueError('校准操作必须为confirm_contact、capture_baseline或reset')
            self.multimodal.reset()
            self._sensor_flow = None
            self._state = {**state, 'emotion': '不确定', 'confidence': 0.0,
                           'risk_score': None, 'validated': False, 'layer': 'calibration_required',
                           'calibration': self.sensor_calibration.status(now),
                           'pad': None, 'affect': None,
                           'detail': {'reason': '校准已更新，等待下一包硬件数据'}}
            self.agent.set_state(self._state)
            return {'ok': True, 'calibration': self._state['calibration']}

    # ------------------------------------------------------------ v2 辅助
    def _cached_baseline(self, ttl: float = 60.0) -> Dict:
        """带 TTL 的个体基线缓存

        个体基线在 PAD 估计与相对特征化中都要用，而上报频率是 1 包/秒。
        每包都读一次 SQLite 在同一秒内会发起多次重复查询 —— 长时间监护
        （本项目目标是常驻运行）下这是纯粹的无谓开销。基线本身是慢变量
        （EMA，BASELINE_ADAPT_RATE=0.05），缓存 60 秒不会影响任何精度。

        注意：**只做读缓存**，基线更新仍由 ``memory.update_baseline`` 负责。
        """
        now = time.time()
        if self._baseline_cache is not None and \
                (now - self._baseline_cache_ts) < ttl:
            return self._baseline_cache
        try:
            self._baseline_cache = self.memory.get_baseline() or {}
        except Exception as e:  # noqa: BLE001
            logger.debug("读取个体基线失败: %s", e)
            self._baseline_cache = {}
        self._baseline_cache_ts = now
        return self._baseline_cache

    def _invalidate_baseline_cache(self):
        """基线被显式改写后调用（如基线采集/校准），保证下一次即读到新值"""
        self._baseline_cache = None
        self._baseline_cache_ts = 0.0

    def _assess_quality(self, feats: Dict) -> Dict:
        """信号质量评估（优先用 IBI 窗口精算，退化到固件上报的 jitter）"""
        if not config.QUALITY_GATE_ENABLED:
            return {"quality_ok": True, "quality_level": "未启用", "jitter": 0.0}
        if len(self._ibi_window) >= 8:
            try:
                return self.extractor.quality_metrics(self._ibi_window, feats)
            except Exception as e:  # noqa: BLE001
                logger.debug("质量评估异常: %s", e)
        # 退化路径：只有固件给的 jitter
        jit = float(feats.get("jitter") or 0.0)
        if jit <= config.QUALITY_JITTER_GOOD:
            level, ok = "合格", True
        elif jit <= config.QUALITY_JITTER_POOR:
            level, ok = "勉强", True
        else:
            level, ok = "不可用", False
        return {"quality_ok": ok, "quality_level": level,
                "jitter": round(jit, 4), "source": "firmware"}

    @staticmethod
    def _probabilities_of(result) -> Dict[str, float]:
        """从识别结果取 5 类概率分布（缺 FLOAT 概率时用 one-hot 兜底）"""
        probs = (result.detail or {}).get("probabilities")
        if isinstance(probs, dict) and probs:
            return {k: float(v) for k, v in probs.items()}
        return {result.emotion: 1.0}

    def _advance_dynamics(self, prob_zh: Dict[str, float], packet: Dict):
        """推进情感动力学状态机（按真实经过时间 dt）"""
        if not config.DYNAMICS_ENABLED:
            return None
        now = float(packet.get("t") or time.time())
        dt = 1.0
        if self._last_affect_update is not None:
            dt = now - self._last_affect_update
            # 时钟跳变/离线补数据保护：钳到 [0.2, 60]
            if dt <= 0 or dt > 600:
                dt = 1.0
        self._last_affect_update = now
        return self.dynamics.update_from_zh(prob_zh, dt=dt, now=now)

    def _maybe_record_state_fta(self, state: Dict, feats: Dict):
        """情绪显著变化时写入一条情境记忆（含生理锚）"""
        if not config.MEMORY_FTA_ENABLED:
            return
        emotion = state.get("emotion", "")
        risk = float(state.get("risk_score") or 0)
        changed = bool(self._prev_emotion) and emotion != self._prev_emotion
        significant = risk >= config.TRIGGER_MEDIUM
        # 距上次写入至少 10 分钟，避免高频重复
        if not (changed or significant):
            return
        if time.time() - self._last_fta_ts < 600:
            return
        try:
            self.memory.save_fta_unit(
                fact_text=f"（生理记录）{datetime.now():%m-%d %H:%M} 前后"
                          f"模型估计为「{emotion}」，工程触发评分 {risk:.0f}/100",
                situation_type=("risk_signal" if risk >= config.TRIGGER_HIGH
                                else "current_state"),
                emotion=emotion,
                emotion_conf=float(state.get("confidence") or 0.0),
                time_orientation="present",
                physio={"rmssd_ms": feats.get("rmssd_ms", feats.get("rmssd")),
                        "sdnn_ms": feats.get("sdnn_ms", feats.get("sdnn")),
                        "lf_hf": feats.get("lf_hf"),
                        "jitter": feats.get("jitter"),
                        "quality": (state.get("quality") or {}).get("quality_level")})
            self._last_fta_ts = time.time()
        except Exception as e:  # noqa: BLE001
            logger.debug("状态 FTA 写入跳过: %s", e)

    # ==================================================================
    # 触发决策
    # ==================================================================
    def check_trigger(self) -> Optional[Dict]:
        """检查是否应主动触发；触发则调用 Agent 生成关怀"""
        self.get_state()  # invalidate stale hardware inference before any trigger
        if not self._state:
            return None
        if self._state.get('simulation') is False and not self._state.get('validated'):
            self.trigger._consecutive_pass = 0
            return {'should_trigger': False, 'reason': self._state.get('detail', {}).get('reason', '分类未验证'),
                    'state': self._state}
        if self._state.get('simulation') is False:
            with self._sensor_lock:
                sensor = self._state.get('sensor') or {}
                identity = (sensor.get('device_id'), sensor.get('boot_id'), sensor.get('seq'))
                if identity == self._last_trigger_sensor:
                    return {'should_trigger': False, 'reason': '本包已检查，等待新的硬件数据', 'state': self._state}
                self._last_trigger_sensor = identity
        if self._state.get("emotion") == "数据积累中":
            return {"should_trigger": False, "reason": "数据积累中",
                    "state": self._state}
        history = self.memory.get_risk_history()
        baseline = self.memory.get_baseline()
        affect = self._state.get("affect")
        decision = self.trigger.decide(
            emotion_result=self._emotion_result_view(),
            risk_history=history,
            baseline=baseline,
            last_trigger_time=self._last_trigger_time,
            last_interaction_time=self._last_interaction_time,
            affect=affect,
            quality=self._quality)

        if not decision.should_trigger:
            return {"should_trigger": False, "decision": decision,
                    "state": self._state,
                    "gate": self.trigger.last_gate_reason or None}

        strategy = select_strategy(decision.trigger_level, self.trigger,
                                   self.memory.get_user_profile().get("preferences"))
        reply = self._generate(lambda: self.agent.proactive_trigger(self._state,
                                                                   decision))
        audio = None
        if config.TTS_ENABLED:
            audio = synthesize_sync(reply)

        self._last_trigger_time = time.time()
        self._pending_strategy = strategy["key"]
        self._pending_before_risk = decision.risk

        return {
            "should_trigger": True,
            "trigger_level": decision.trigger_level,
            "strategy": strategy,
            "risk": decision.risk,
            "reason": decision.reason,
            "message": reply,
            "tts_audio": audio,
            "state": self._state,
            "llm_params": self.llm.last_params,
        }

    def tick(self, packet: Dict) -> Optional[Dict]:
        """1 秒主循环: 数据进入 + 触发检查"""
        self.ingest_sensor(packet)
        return self.check_trigger()

    # ==================================================================
    # 生成参数调制（v2 核心：情绪强度 + 心境 → LLM 外部超参）
    # ==================================================================
    def _generate(self, fn):
        """在"情绪自适应的生成参数"下执行一次生成，完成后恢复默认

        ⚠️ 必须恢复：否则一次高焦虑的低温度会泄漏到后续所有无关对话。
        """
        if not config.LLM_PARAM_MODULATION:
            return fn()
        st = self._state or {}
        emotion = st.get("emotion", "平静")
        affect = st.get("affect") or {}
        intensity = float(affect.get("dominant_intensity") or 0.0)
        mood = float(affect.get("mood") or 0.0)
        params = {}
        try:
            params = modulate_llm_params(emotion, intensity, mood)
            self.llm.set_overrides(**params)
            # 主动记录"本轮本应下发的参数"：离线模式下不会走 _raw_chat，
            # 若不在这里写，演示时看不到调制效果，也无法验证是否真的生效。
            self.llm._last_params = dict(params)   # noqa: SLF001
            return fn()
        except Exception as e:  # noqa: BLE001
            logger.debug("参数调制失败，回退默认参数: %s", e)
            return fn()
        finally:
            self.llm.clear_overrides()

    # ==================================================================
    # 用户对话
    # ==================================================================
    def chat(self, message: str) -> Dict:
        """用户主动对话（带当前情绪上下文 + 参数调制 + 记忆检索）"""
        self.get_state()
        self._last_interaction_time = time.time()
        fta_id = self._capture_fta(message)
        reply = self._generate(lambda: self.agent.run_with_emotion(message,
                                                                   self._state))
        audio = None
        if config.TTS_ENABLED:
            audio = synthesize_sync(reply)
        state = self._state
        return {"response": reply,
                "emotion_adapted": bool(self.agent._has_meaningful_state(state)),  # noqa: SLF001
                "state": state,
                "tts_audio": audio,
                "llm_backend": self.llm.backend,
                "llm_params": self.llm.last_params,
                "fta_id": fta_id}

    def stream_chat(self, message: str):
        """流式对话生成器（WebSocket/CLI 用）"""
        self.get_state()
        self._last_interaction_time = time.time()
        state = self._state or {}
        if self.agent._crisis_text(message):  # noqa: SLF001
            reply = self.agent._crisis_response(message, state)  # noqa: SLF001
            self.memory.save_interaction(message, reply, state)
            yield reply
            return
        if self.agent._has_meaningful_state(state):  # noqa: SLF001
            from ..agent.prompts import build_emotion_context
            input_text = build_emotion_context(message, state)
        else:
            input_text = message
        full = ""
        for chunk in self.agent.stream_run(input_text):
            full += chunk
            yield chunk
        self.memory.save_interaction(message, full, state)

    # ------------------------------------------------------------ FTA 抽取
    def _capture_fta(self, message: str) -> Optional[int]:
        """从用户消息中识别值得长期记住的情境，落成 FTA 记忆单元

        ⚠️ 这是**规则版**抽取（关键词匹配），不是文献5 的 LLM 边界检测器。
        它只做"值不值得记"的粗筛，宁缺毋滥：只有命中风险信号、计划承诺、
        关系事件等明确类型才写入。有 LLM 时可在 ``agent`` 层升级为
        模型抽取（文献5 的 BWS 边界保持窗口切分），此处保留接口。
        """
        if not config.MEMORY_FTA_ENABLED or not message:
            return None
        msg = message.strip()
        if len(msg) < 6:
            return None
        stype = "general_fact"
        for t, kws in _FTA_RULES.items():
            if any(k in msg for k in kws):
                stype = t
                break
        if stype == "general_fact":
            return None          # 普通闲聊不建记忆，避免记忆库被噪声填满
        st = self._state or {}
        try:
            return self.memory.save_fta_unit(
                fact_text=msg[:120],
                situation_type=stype,
                emotion=st.get("emotion", ""),
                emotion_conf=float(st.get("confidence") or 0.0),
                time_orientation=("future" if stype == "homework_or_plan"
                                  else "present"),
                physio={"rmssd_ms": st.get("rmssd_ms"), "sdnn_ms": st.get("sdnn_ms"),
                        "lf_hf": st.get("lf_hf"), "jitter": st.get("jitter")},
                evidence_span=msg[:200])
        except Exception as e:  # noqa: BLE001
            logger.debug("FTA 抽取跳过: %s", e)
            return None

    # ==================================================================
    # 手动情绪
    # ==================================================================
    def set_manual_emotion(self, emotion: str, risk: Optional[float] = None,
                           hr: Optional[float] = None,
                           rr: Optional[float] = None) -> Dict:
        """
        手动注入情绪状态（无硬件时体验情绪自适应对话）:
        供 CLI /心情 命令与 API /api/emotion/manual 使用。
        """
        from ..algorithms.emotion_recognizer import EMOTION_BASE_RISK
        emotion = emotion.strip()
        valid = {"平静", "愉悦", "低焦虑", "高焦虑", "疲劳"}
        if emotion not in valid:
            return {"ok": False, "error": f"情绪需为 {sorted(valid)} 之一"}
        risk = float(risk) if risk is not None else EMOTION_BASE_RISK[emotion]
        risk = max(0.0, min(100.0, risk))
        # 手动模式下给一个中等强度，让动力学不至于永远停在 unaware
        affect = self.dynamics.update_from_zh({emotion: 0.85}, dt=1.0) \
            if config.DYNAMICS_ENABLED else None
        state = {
            "emotion": emotion,
            "confidence": 0.9,
            "risk_score": round(risk, 1),
            "layer": "manual",
            "simulation": True,
            "validated": False,
            "detail": {"manual": True},
            "hr_bpm": float(hr) if hr else None,
            "rr_bpm": float(rr) if rr else None,
            "pad": None,
            "affect": affect.to_dict() if affect else None,
            "quality": {"quality_ok": True, "quality_level": "手动注入",
                        "jitter": 0.0},
            "timestamp": time.time(),
        }
        self._state = state
        self.agent.set_state(state)
        self._quality = state['quality']
        self._sensor_flow = None
        self.multimodal.reset()
        self.memory.record_emotion_state(state)
        return {"ok": True, "state": state,
                "hint": f"已注入情绪「{emotion}」（风险 {risk:.0f}/100）。"
                        "现在和小暖对话，她会按这个状态调整语气与回应。"}

    # ==================================================================
    # 反馈闭环
    # ==================================================================
    def feedback(self, effective: Optional[bool] = None,
                 after_risk: Optional[float] = None) -> Dict:
        """STEP 06 反馈优化: 干预效果 → 更新策略有效率与个体阈值

        v2 在**两个粒度**上记录（复用同一张 feedback 表，靠 strategy 字段区分）：

          * **策略级** —— 系统主动触发用到的策略（active_dialogue 等），
            驱动触发阈值自适应；
          * **技能级** —— 对话中实际用过的心理技能（comfort_tool 的
            last_ids），驱动 comfort_tool 的匹配打分。

        为什么要加技能级：v1 只知道"呼吸引导有没有用"，却不知道
        "去灾难化 / 认知解离 / 行为激活"里哪一种对这位用户有效 ——
        而这恰恰是舒适工具能不能真正做个性化的前提（文献4 的反馈闭环）。

        兼容性: 无任何待评估记录时仍返回 ``recorded=False``（v1 行为）。
        """
        skills = []
        agent = getattr(self, "agent", None)
        if agent is not None:
            try:
                skills = list(agent.pending_skill_ids)
            except Exception:  # noqa: BLE001
                skills = []

        if not self._pending_strategy and not skills:
            return {"recorded": False, "reason": "无待评估的主动干预"}

        before = self._pending_before_risk or 0
        after = after_risk if after_risk is not None else (
            (self.get_state() or {}).get("risk_score"))
        if effective is None and after is None:
            return {'recorded': False, 'reason': '当前风险未知，请提供主观干预反馈'}
        if after is None:
            after = before
        if effective is None:
            effective = (after < before - 5)  # 风险下降5分以上视为有效

        out: Dict = {"recorded": False,
                     "before_risk": before, "after_risk": after,
                     "effective": bool(effective)}

        if self._pending_strategy:
            self.trigger.record_feedback(self._pending_strategy, bool(effective))
            self.memory.record_feedback(self._pending_strategy, before, after,
                                        bool(effective))
            out.update({
                "recorded": True,
                "strategy": self._pending_strategy,
                "trigger_threshold_offset":
                    self.trigger.personal_offsets.get("trigger", 0.0),
            })

        # ---- 技能级反馈（v2）----
        # 注意: 技能的 before/after 用同一次会话的风险值，因此
        # 「有效」判据在技能粒度上并不可靠（没有独立的干预前基线）。
        # 这里如实记录，但**不**据此调整触发阈值 —— 只在 comfort_tool
        # 打分时作为弱先验使用（权重 2.5，可被情绪/唤醒维度盖过）。
        if skills:
            for sid in skills:
                self.memory.record_feedback(sid, before, after, bool(effective))
            out["skills_recorded"] = skills
            out["recorded"] = True
            if agent is not None:
                agent.clear_pending_skills()

        self._pending_strategy = None
        self._pending_before_risk = None
        return out

    # ==================================================================
    # 状态查询
    # ==================================================================
    def get_state(self) -> Optional[Dict]:
        with self._sensor_lock:
            if (self._state and self._state.get('simulation') is False
                    and self._last_sensor_received is not None
                    and time.monotonic()-self._last_sensor_received > config.SENSOR_STALE_SEC
                    and self._state.get('layer') != 'stale'):
                self.sensor_pipeline.clear(self.sensor_pipeline.device_ms or 0)
                self.sensor_calibration.reset()
                self._sensor_flow = None
                self.multimodal.reset()
                self._quality = {'quality_ok':False, 'quality_level':'不可用',
                                 'reason':'传感器数据已过期，请重新连接', 'jitter':0}
                self._state = {**self._state, 'emotion':'不确定', 'confidence':0,
                               'risk_score':None, 'validated':False, 'layer':'stale',
                               'quality':self._quality, 'quality_level':'不可用', 'pad':None, 'affect':None,
                               'features': {}, 'feats': {}, 'calibration': self.sensor_calibration.status(0),
                               'detail':{'reason':self._quality['reason']}}
                self.agent.set_state(self._state)
            return self._state

    def get_affect(self) -> Optional[Dict]:
        """情感动力学快照（v2）"""
        if not config.DYNAMICS_ENABLED:
            return None
        state = self.get_state() or {}
        if state.get('simulation') is False:
            return state.get('affect') if state.get('validated') else None
        return self.dynamics.snapshot().to_dict()

    def quality_report(self) -> Dict:
        """最近一次信号质量报告（v2）"""
        self.get_state()
        return dict(self._quality or {})

    def set_personality(self, key: str) -> Dict:
        """切换用户气质档位（文献3 表2：理想型/冷漠型/敏感型/执着型）"""
        from ..algorithms.emotion_dynamics import PERSONALITY_PROFILES
        if key not in PERSONALITY_PROFILES:
            return {"ok": False,
                    "error": f"未知档位，可选 {sorted(PERSONALITY_PROFILES)}"}
        self.dynamics = EmotionDynamics(personality=key)
        return {"ok": True, "personality": key,
                "zh": PERSONALITY_PROFILES[key]["zh"]}

    def get_timeline(self, days: float = 7) -> List[Dict]:
        return self.memory.get_emotional_timeline(days)

    def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict]:
        """按查询检索 FTA 情境记忆（供界面"小暖记得的事"展示）"""
        return self.memory.retrieve_fta(query, top_k=top_k)

    def update_daily_summary(self, day: Optional[str] = None) -> Dict:
        """生成/更新当日摘要（文献4 MemoryBank 的分层摘要中层）

        规则版实现：按情绪分布与风险峰值生成结构化摘要，
        不经 LLM（省调用、可离线）；有 LLM 时可替换为模型摘要。
        """
        day = day or datetime.now().strftime("%Y-%m-%d")
        dist = self.memory.emotion_distribution(days=1)
        units = self.memory.retrieve_fta("", top_k=5) if config.MEMORY_FTA_ENABLED else []
        total = sum(dist.values()) or 1
        top = sorted(dist.items(), key=lambda x: -x[1])[:3]
        dominant = "、".join(f"{k}({v/total*100:.0f}%)" for k, v in top)
        summary = f"当日主要情绪: {dominant or '无数据'}。"
        if units:
            summary += "值得记住: " + "；".join(u["fact"][:30] for u in units[:3])
        self.memory.upsert_daily_summary(day, summary,
                                         {"distribution": dist})
        return {"day": day, "summary": summary}

    def get_profile(self) -> Dict:
        self.get_state()
        ml_src = None
        if self.recognizer.ml.available:
            _b = self.recognizer.ml._bundle or {}
            _meta = _b.get("metadata") or {}
            ml_src = _meta.get("source") or (
                "MAD 数据集" if "classes_zh" not in _b else "自定义")
        return {"user_id": self.user_id,
                "profile": self.memory.get_user_profile(),
                "portrait": self.memory.get_portrait(),
                "baseline": self.memory.get_baseline(),
                "stats": self.memory.stats(),
                "llm_backend": self.llm.backend,
                "llm_params": self.llm.last_params,
                "ml_loaded": self.multimodal.available,
                "ml_source": '独立自评+按受试者分组验证的多通道模型' if self.multimodal.available else None,
                "demo_ml_loaded": self.recognizer.ml.available,
                "demo_ml_source": ml_src,
                "calibration": self.sensor_calibration.status(self.sensor_pipeline.device_ms or 0),
                "pad_loaded": self.pad_regressor.available,
                "personality": self.dynamics.personality.get("zh"),
                "quality": self._quality,
                "uptime_sec": round(time.time() - self.started_at, 1)}

    # ==================================================================
    # 数据权利（文献5 伦理要求）
    # ==================================================================
    def export_data(self) -> Dict:
        """导出用户全部数据（数据可携带权）"""
        return self.memory.export_user_data()

    def purge_data(self, keep_profile: bool = True) -> Dict:
        """清除用户记忆数据（被遗忘权）"""
        return self.memory.purge_user_data(keep_profile=keep_profile)

    def _emotion_result_view(self):
        """将当前状态封装为 TriggerEngine 所需视图（无需构造 EmotionResult）"""
        from ..algorithms.emotion_recognizer import EmotionResult
        return EmotionResult(
            emotion=self._state["emotion"],
            confidence=self._state["confidence"],
            risk_score=self._state["risk_score"],
            layer=self._state.get("layer", ""),
            detail={"hr_bpm": self._state.get("hr_bpm", 0),
                    **self._state.get("detail", {})})

    def close(self):
        self.memory.close()
