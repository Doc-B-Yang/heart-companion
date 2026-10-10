# -*- coding: utf-8 -*-
"""
API 路由层（v2）
================
端点设计与请求/响应格式对齐《项目开发计划书》§4.4 与 §5.2。

v2 相对 v1 的三处结构性变化（都源于算法层与工具层的能力扩张）
--------------------------------------------------------------------------------
1. **状态接口承载完整感知链**。v1 的 ``/emotion/state`` 只回离散情绪 + 一张
   按标签查表算出来的 detail 视图；v2 增加 PAD 三维、情感动力学快照、
   信号质量报告，且 detail 视图优先由 PAD 连续值换算（情绪内才有区分度）。
2. **新增"校准类"端点**。``/self-report`` 把用户主观评分落库并回传主客观
   对照 —— 这是让「情绪识别准确率 > 85%」这个指标可被**真值**验证的唯一途径
   （此前只有客观判定，没有真值，指标无法举证）。
3. **新增"能力查询类"端点**。技能库（22 项循证技术）、分层转介、
   呼吸引导脚本由后端统一给出，避免 APP / 语音端各写一份规则而互相漂移。

隐私约定
--------------------------------------------------------------------------------
``state.feats`` / ``state.rel_features`` 属于内部归因数据，**不进入 Prompt、
也不默认下发**。只有在 ``EXPOSE_RAW_METRICS=1`` 时才通过 ``metrics`` 字段返回，
供调试与论文绘图使用。默认对外是语义化区间（见 ``backend/views.public_state``）。
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from ..core.label_store import LabelStore, SelfReportStore

from hrv_companion_agent import config
from hrv_companion_agent.core.system import HeartCompanionSystem
from .views import detail_view as _detail_view
from .views import public_state as _public_state

router = APIRouter()

# ---------------------------------------------------------------- 会话管理
_systems: Dict[str, HeartCompanionSystem] = {}
_lock = threading.Lock()
_labels = LabelStore()
_experiment_reports = SelfReportStore()


class RawSensorData(BaseModel):
    user_id: str = Field(default='user001', min_length=1, max_length=64)
    packet: Dict


class LabelRequest(BaseModel):
    user_id: str = Field(default='user001', min_length=1, max_length=64)
    label: str
    note: str = Field(default='', max_length=1000)
    arousal_score: Optional[int] = Field(default=None, ge=1, le=9, strict=True)
    valence_score: Optional[int] = Field(default=None, ge=1, le=9, strict=True)


class CalibrationRequest(BaseModel):
    user_id: str = Field(default='user001', min_length=1, max_length=64)
    action: str


@router.post('/sensor/calibration')
def sensor_calibration(data: CalibrationRequest):
    try:
        return get_system(data.user_id).calibrate_sensor(data.action)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post('/sensor/raw')
def sensor_raw(data: RawSensorData):
    if data.packet.get('_simulation') or data.packet.get('_replay'):
        raise HTTPException(status_code=422,detail='硬件入口不接受模拟包')
    if len(data.user_id)>64 or not data.user_id:
        raise HTTPException(status_code=422,detail='用户标识无效')
    system = get_system(data.user_id)
    with system._sensor_lock:
        state = system.ingest_sensor(data.packet)
        flow = system._sensor_flow or {}
        diagnostics = dict(flow)
        if not config.EXPOSE_RAW_METRICS:
            diagnostics.pop('features', None)
        return {'accepted':bool((system._sensor_flow or {}).get('accepted')),
                'state':_public_state(state),'trigger':{'should_trigger':False},
                'diagnostics':diagnostics}


@router.post('/dataset/label')
def dataset_label(data: LabelRequest):
    try:
        return _labels.append(get_system(data.user_id), data.label, data.note,
                              data.arousal_score, data.valence_score)
    except ValueError as exc:
        raise HTTPException(status_code=409,detail=str(exc)) from exc


@router.post('/dataset/self-report')
def experiment_self_report(data: LabelRequest):
    try:
        _experiment_reports.record(data.user_id, data.label, data.note,
                                   data.arousal_score, data.valence_score)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        linked = _labels.append(get_system(data.user_id), data.label, data.note,
                                data.arousal_score, data.valence_score)
    except ValueError as exc:
        return {'saved': True, 'label': data.label, 'training_sample_saved': False,
                'training_reason': str(exc), 'next_label_after_sec': 0}
    return {**linked, 'training_sample_saved': True}


def get_system(user_id: str) -> HeartCompanionSystem:
    with _lock:
        if user_id not in _systems:
            _systems[user_id] = HeartCompanionSystem(user_id=user_id)
        return _systems[user_id]


# ---------------------------------------------------------------- 请求模型
class SensorData(BaseModel):
    """ESP32 上报的生理数据包（1 包/秒，契约见《硬件到Agent对接说明》）"""
    user_id: str
    device_id: Optional[str] = None
    timestamp: Optional[float] = None
    hr: float
    ibi: Optional[list] = None
    sdnn: Optional[float] = None
    rmssd: Optional[float] = None
    pnn50: Optional[float] = None
    lf_hf: Optional[float] = None
    rr: Optional[float] = None
    sig_quality: Optional[float] = None


class ChatRequest(BaseModel):
    user_id: str
    message: str
    emotion_state: Optional[dict] = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


class FeedbackRequest(BaseModel):
    user_id: str
    effective: Optional[bool] = None
    after_risk: Optional[float] = None


class ManualEmotionRequest(BaseModel):
    user_id: str
    emotion: str
    risk: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None


class SelfReportRequest(BaseModel):
    """用户主观自评 —— 训练标签与验收真值的唯一来源。

    ``score`` 支持 0~10（自动按百分制 ×10）或 0~100 两种口径，
    由 ``agent/tools/self_report_tool`` 统一归一，此处不重复校验，
    以免前后端两套规则算出不同的值。
    """
    user_id: str
    score: float
    emotion: Optional[str] = ""
    note: Optional[str] = ""


class PersonalityRequest(BaseModel):
    user_id: str
    personality: str


class PurgeRequest(BaseModel):
    user_id: str
    keep_profile: bool = True


class CrisisCheckRequest(BaseModel):
    text: str


class PADQuery(BaseModel):
    user_id: str
    dim: Optional[str] = "all"
    why: Optional[bool] = False


# ---------------------------------------------------------------- 视图函数
# 视图换算已抽到 backend/views.py（纯函数、不依赖 Web 框架，便于单测）：
# 那里集中了「PAD 连续量 → 焦虑/压力/疲劳」的映射与「哪些数值可对外」的裁剪规则。
# 别名加下划线前缀，保持与 v1 的模块内调用名一致。


def _tool_text(system: HeartCompanionSystem, name: str, params: Dict) -> str:
    """调用 Agent 的某个工具并取回文本，失败时给出可诊断的 HTTP 错误"""
    try:
        system.get_state()
        return system.agent.run_tool(name, params)
    except KeyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"工具 '{name}' 执行失败: {e}") from e


# ================================================================== 基础端点
@router.get("/health")
def health():
    infos = {}
    with _lock:
        for uid, sys_ in list(_systems.items())[:20]:
            infos[uid] = {"llm": sys_.llm.backend,
                          "ml": sys_.multimodal.available,
                          "pad": sys_.pad_regressor.available}
    return {"status": "ok", "version": "v3", "systems": len(_systems),
            "details": infos, "time": time.time(),
            "features": {
                "quality_gate": config.QUALITY_GATE_ENABLED,
                "pad": config.PAD_ENABLED,
                "dynamics": config.DYNAMICS_ENABLED,
                "fta_memory": config.MEMORY_FTA_ENABLED,
                "self_report": config.SELF_REPORT_ENABLED,
                "expose_raw_metrics": config.EXPOSE_RAW_METRICS,
                "privacy_notice": config.PRIVACY_NOTICE_VERSION,
            }}


# ================================================================== 生理入口
@router.post("/sensor/data")
def sensor_data(data: SensorData):
    """接收 ESP32 上传的生理数据（计划书 §5.2 ESP32→后端格式）"""
    system = get_system(data.user_id)
    packet = {
        "t": data.timestamp or time.time(),
        "hr": data.hr,
        "ibi": data.ibi or [],
        "sdnn": data.sdnn,
        "rmssd": data.rmssd,
        "pnn50": data.pnn50,
        "lf_hf": data.lf_hf,
        "rr": data.rr,
        "sig_quality": data.sig_quality if data.sig_quality is not None else 0.0,
    }
    state = system.ingest_sensor(packet)
    triggered = {'should_trigger':False,'reason':'旧接口缺少schema=2证据，请用/api/sensor/raw'}
    return {"accepted": False, "state": _public_state(state),
            "trigger": triggered}


@router.get("/emotion/state")
def emotion_state(user_id: str):
    """获取用户当前情绪状态（计划书 §5.2 后端→APP格式）"""
    system = get_system(user_id)
    state = system.get_state()
    if not state:
        return {"user_id": user_id, "emotion": None,
                "detail": _detail_view(None), "message": "暂无传感器数据"}
    return {"user_id": user_id, **_public_state(state)}


@router.get("/emotion/quality")
def emotion_quality(user_id: str):
    """信号质量报告（v2）

    单独暴露的原因：APP 在展示任何情绪结论前，应先看这个端点决定
    "这次读数能不能作为依据"。实测已确认指腹接触不良时 HR 仍显示正常
    而 HRV 完全不可信 —— 只看 HR 会给出错误的安全感。
    """
    system = get_system(user_id)
    return {"user_id": user_id, **system.quality_report()}


@router.get("/emotion/affect")
def emotion_affect(user_id: str):
    """情感动力学快照（v2：PWE 情绪轮 + 强度/持续/心境累积）"""
    system = get_system(user_id)
    affect = system.get_affect()
    if not affect:
        return {"user_id": user_id, "affect": None,
                "message": "情感动力学未启用"}
    return {"user_id": user_id, "affect": affect}


@router.get("/emotion/pad")
def emotion_pad(user_id: str, dim: str = "all", why: bool = False):
    """PAD 三维读数的文字化解释（v2）

    返回的是 pad_tool 生成的**自然语言说明**而不是裸数值 ——
    数值属于内部量纲，直接展示给用户没有意义，且容易过度解读。
    """
    system = get_system(user_id)
    return {"user_id": user_id,
            "text": _tool_text(system, "pad_tool",
                               {"dim": dim, "why": "1" if why else ""})}


@router.post("/emotion/manual")
def manual_emotion(req: ManualEmotionRequest):
    """手动注入情绪状态（无硬件联调时体验情绪自适应对话）"""
    system = get_system(req.user_id)
    return system.set_manual_emotion(req.emotion, risk=req.risk,
                                     hr=req.hr, rr=req.rr)


# ================================================================== 对话
@router.post("/chat/message")
def chat_message(req: ChatRequest):
    """对话交互（计划书 §5.2 APP→后端 / 后端→APP格式）"""
    system = get_system(req.user_id)
    if req.emotion_state:
        base = system.get_state() or {}
        system.agent.set_state({**base, **(req.emotion_state or {})})
    result = system.chat(req.message)
    return {
        "response": result["response"],
        "emotion_adapted": result["emotion_adapted"],
        "state": _public_state(result["state"]),
        "llm_params": result.get("llm_params"),
        "fta_id": result.get("fta_id"),
        "tts_audio_url": None,  # 由 /api/tts/audio/{file} 提供
    }


@router.websocket("/chat/stream")
async def chat_stream(ws: WebSocket):
    """流式对话（实时打字效果）"""
    await ws.accept()
    user_id = "user001"
    try:
        while True:
            msg = await ws.receive_json()
            user_id = msg.get("user_id", user_id)
            system = get_system(user_id)
            await ws.send_json({"type": "state",
                                "state": _public_state(system.get_state())})
            for chunk in system.stream_chat(msg.get("message", "")):
                await ws.send_json({"type": "chunk", "content": chunk})
            await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "content": str(e)})
        except Exception:  # noqa: BLE001
            pass


# ================================================================== 触发与反馈
@router.get("/trigger/check")
def trigger_check(user_id: str):
    """主动检查和触发"""
    system = get_system(user_id)
    result = system.check_trigger() or {"should_trigger": False}
    return result


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    """干预效果反馈（STEP 06 反馈优化闭环）

    v2 会同时记录**策略级**（驱动触发阈值）与**技能级**
    （驱动舒适工具的匹配打分）两个粒度，见 ``system.feedback`` 的说明。
    """
    system = get_system(req.user_id)
    return system.feedback(effective=req.effective, after_risk=req.after_risk)


# ================================================================== 主观自评
@router.post("/self-report")
def self_report(req: SelfReportRequest):
    """提交主观自评（v2）

    这是"情绪识别准确率 > 85%"这一指标**唯一**的可验证真值来源：
    没有用户自评，所有准确率都只是在拿算法比算法。
    """
    if not config.SELF_REPORT_ENABLED:
        raise HTTPException(status_code=503, detail="主观自评未启用")
    system = get_system(req.user_id)
    return {"user_id": req.user_id,
            "text": system.agent.run_self_report(
                {"score": req.score, "emotion": req.emotion or "",
                 "note": req.note or ""})}


@router.get("/self-report/consistency")
def self_report_consistency(user_id: str, limit: int = 20):
    """主客观一致性统计（v2）

    返回 conflict_rate 与 mean_gap。这两个数在论文里的用法是：
    冲突率高不一定说明算法差 —— 也可能说明**用户不愿如实自评**，
    或生理指标捕捉的是"未意识到"的那部分状态。两种解释都需要后续访谈区分，
    因此接口只给数，不给结论。
    """
    system = get_system(user_id)
    return {"user_id": user_id,
            "consistency": system.memory.self_report_consistency(),
            "recent": system.memory.get_self_reports(limit=limit)}


# ================================================================== 能力查询
@router.get("/skills")
def list_skills(emotion: str = "", arousal: str = "",
                intensity: float = 0.0, top_k: int = 3):
    """查询匹配到的心理技能（v2：22 项循证技术库）

    暴露给 APP 的价值：干预卡片可以直接用这里返回的 ``script`` 与
    ``caution``，不必在客户端重写一遍禁忌规则（两处规则必然漂移）。
    """
    from hrv_companion_agent.agent.tools import match_skills
    picked = match_skills(emotion=emotion, arousal=arousal,
                          intensity=intensity, top_k=max(1, min(top_k, 5)))
    return {"emotion": emotion, "arousal": arousal, "intensity": intensity,
            "skills": picked}


@router.get("/skills/breathing")
def breathing_guide(emotion: str = "", type: str = ""):  # noqa: A002
    """呼吸引导脚本（v2）

    ⚠️ 返回文本里含一条给模型的边界说明：呼吸会直接改变 HRV 的
    LF/HF 比值，练习期间采集到的读数反映的是呼吸本身而非情绪。
    """
    system = get_system(config.DEFAULT_USER_ID)
    return {"emotion": emotion, "type": type,
            "text": _tool_text(system, "breathing_guide",
                               {"emotion": emotion, "type": type})}


@router.get("/referral")
def referral(level: str = "", text: str = "", risk: float = 0.0):
    """分层转介指引（v2）

    四档 L1~L4（自助 / 社会支持 / 临床 / 紧急），核心产出是
    「如何向他人开口」的话术脚本 —— 这是文献6 指出产品最该做、
    而多数系统做得最含糊的一环。
    """
    system = get_system(config.DEFAULT_USER_ID)
    return {"level": level,
            "text": _tool_text(system, "referral_tool",
                               {"level": level, "text": text, "risk": risk})}


@router.post("/crisis/check")
def crisis_check(req: CrisisCheckRequest):
    """危机文本分级检测（v2：被动意念 / 主动意念 / 计划 / 准备）

    ⚠️ 调用方必须把 ``is_crisis=True`` 当**最高优先级**处理：
    不做其他功能引导、不追问细节、直接给出对应的第一步。
    """
    from hrv_companion_agent.agent.tools import detect_level
    level = detect_level(req.text or "")
    return {"is_crisis": level is not None, "level": level,
            "first_step": _crisis_first_step(level)}


def _crisis_first_step(level: Optional[str]) -> Optional[str]:
    """把分级映射到"现在该做的第一件事"（与 crisis_tool 的四档一致）"""
    from hrv_companion_agent.agent.tools.crisis_tool import LEVELS
    if not level:
        return None
    return (LEVELS.get(level) or {}).get("first_step")


# ================================================================== 记忆
@router.get("/memory/recall")
def memory_recall(user_id: str, q: str = Query("", alias="q"),
                  top_k: int = 5):
    """检索 FTA 情境记忆（v2：五元组 + 混合检索 + 时序链接展开）"""
    system = get_system(user_id)
    if not config.MEMORY_FTA_ENABLED:
        return {"user_id": user_id, "units": [], "message": "FTA 记忆未启用"}
    return {"user_id": user_id, "query": q,
            "units": system.memory.retrieve_fta(q, top_k=top_k)}


@router.get("/memory/pending")
def memory_pending(user_id: str, limit: int = 3):
    """待跟进的事项（v2：lifecycle=pending 的情境单元）

    这是"主动陪伴"真正区别于"被动问答"的地方 —— 系统记得上次没做完的事。
    """
    system = get_system(user_id)
    return {"user_id": user_id,
            "units": system.memory.get_pending_units(limit=limit)}


@router.get("/memory/export")
def memory_export(user_id: str):
    """导出用户全部数据（v2：文献5 的数据可携带权）"""
    system = get_system(user_id)
    return {**system.export_data(), 'multimodal_labels': _labels.export(user_id),
            'experiment_self_reports': _experiment_reports.export(user_id)}


@router.post("/memory/purge")
def memory_purge(req: PurgeRequest):
    """清除用户记忆数据（v2：文献5 的被遗忘权）

    ⚠️ 不可逆。调用方需在客户端做二次确认。
    """
    system = get_system(req.user_id)
    result = system.purge_data(keep_profile=req.keep_profile)
    result['multimodal_labels_removed'] = _labels.purge(req.user_id)
    result['experiment_self_reports_removed'] = _experiment_reports.purge(req.user_id)
    return result


# ================================================================== 画像与统计
@router.get("/user/profile")
def user_profile(user_id: str):
    """获取用户画像和情绪趋势（计划书 §4.4 /api/user/profile）"""
    system = get_system(user_id)
    return {
        **system.get_profile(),
        "timeline_7d": system.get_timeline(days=7),
        "recent_interactions": system.memory.get_recent_interactions(limit=10),
    }


@router.post("/user/personality")
def set_personality(req: PersonalityRequest):
    """切换用户气质档位（文献3 表2：理想型/冷漠型/敏感型/执着型）

    不同档位改变情绪衰减速率与心境累积系数 —— 同一个生理信号，
    对"敏感型"用户应该比"冷漠型"更容易被判定为显著困扰。
    """
    system = get_system(req.user_id)
    return system.set_personality(req.personality)


@router.get("/stats/strategy")
def strategy_stats(user_id: str):
    """策略与技能的历史有效率统计（v2）"""
    system = get_system(user_id)
    return {"user_id": user_id,
            "strategy_stats": system.memory.strategy_stats_persisted(),
            "emotion_distribution_7d": system.memory.emotion_distribution(7)}


@router.post("/summary/daily")
def daily_summary(user_id: str, day: Optional[str] = None):
    """生成/更新当日摘要（文献4 分层摘要的中层）"""
    system = get_system(user_id)
    return system.update_daily_summary(day)


# ================================================================== 语音
@router.post("/tts/synthesize")
async def tts_synthesize(req: TTSRequest):
    """TTS 语音合成（Edge-TTS）"""
    from hrv_companion_agent.agent.tts_service import synthesize
    path = await synthesize(req.text, req.voice)
    if not path:
        raise HTTPException(status_code=503,
                            detail="TTS 不可用（未启用或未安装 edge-tts）")
    return {"audio_path": path, "text": req.text}
