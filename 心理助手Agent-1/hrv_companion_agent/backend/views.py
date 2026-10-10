# -*- coding: utf-8 -*-
"""
状态视图换算层（v2）
====================
把内部状态字典换算成对外可见的 APP 视图。

**为什么单独成模块**：这段逻辑里藏着两个会在论文里被追问的判断 ——
① PAD 连续量如何映射成焦虑/压力/疲劳三个"程度"字段；
② 哪些生理数值可以对外暴露、哪些必须收在调试开关后面。
它们是纯函数，不该被绑死在 FastAPI 上，否则没有 Web 依赖的环境
（比如只装了 numpy + sklearn 的实验机）就没法对它们做单元测试。
"""

from __future__ import annotations

from typing import Dict, Optional

from .. import config

#: PAD 三维的量纲上限（文献2 的量表为 7 级、居中后约 ±3，本项目取 ±4 留余量）
PAD_RANGE = 4.0

#: 离散情绪 → (anxiety, stress, fatigue, valence) 的兜底查表
#: 仅在 PAD 不可用（未启用 / 估计失败 / 手动注入）时使用。
#: ⚠️ 它的固有缺陷是"同一情绪里的所有人都拿到同一个分数"，没有情绪内区分度，
#:    因此 v2 把它降级为兜底而非主路径。
EMOTION_TABLE = {
    "平静": (10, 15, 15, 0.65), "愉悦": (5, 10, 10, 0.85),
    "低焦虑": (62, 55, 25, 0.38), "高焦虑": (88, 82, 35, 0.22),
    "疲劳": (25, 30, 75, 0.35),
}

#: 风险分数的语义区间（**定义在 config**，与 prompts.py 共用同一份，
#: 避免"前端显示偏高、模型被告知中等"这种跨层矛盾）
RISK_BANDS: tuple = tuple((lo, label) for lo, label, _hint in config.RISK_BANDS)

#: PAD 三轴 → 三个"程度"字段的仿射权重（归一化后的轴上）
#: 格式 ``(P, A, D)``，最终得分 = 50 + 25 × Σ(权重 × 轴)
#: ⚠️ 这些是**工程取值**，只保证单调性与值域正确，不代表各轴对构念的
#:    因果贡献；真实系数需靠 ``/self-report`` 累积的真值做回归标定。
_DETAIL_WEIGHTS = {
    # 焦虑: 效价↓ 唤醒↑                （经典二维画像）
    "anxiety_level": (-1.20, 1.00, 0.00),
    # 压力: 效价↓ 唤醒↑ 支配↓          （失控感加重负荷）
    "stress_level": (-1.00, 1.00, -0.40),
    # 疲劳: 效价↓ 唤醒↓ 支配↓          （低能量 + 无力感）
    "fatigue_level": (-0.80, -1.00, -0.60),
}

_SCALE = 25.0        # 归一化轴 × 权重 × 25 → 50 分中心 ± 若干
_CENTER = 50.0


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def band(score: Optional[float]) -> str:
    """把 0~100 的风险分数折成语义区间（**短标签**，适合界面展示）"""
    try:
        v = float(score or 0)
    except (TypeError, ValueError):
        return "未知"
    for lo, label in RISK_BANDS:
        if v >= lo:
            return label
    return RISK_BANDS[-1][1]


def detail_from_pad(pad: Dict) -> Dict:
    """PAD 连续维度 → APP 的 anxiety / stress / fatigue / valence 视图

    这三个"程度"字段原本是按离散标签查表得到的常量，问题在于：两位用户
    都被判为「高焦虑」时，APP 上显示的紧张值完全相同，**情绪内毫无区分度**；
    而 PAD 是逐人逐秒回归出来的连续量，恰好补上这个粒度。

    换算是一次仿射映射（PAD 归一化到 ±1 后按各轴权重加权）：

      * 紧张（anxiety）随 唤醒↑、效价↓  上升 —— 焦虑的经典二维画像；
      * 压力（stress） 随 唤醒↑、效价↓、支配↓ 上升 —— 失控感加重负荷；
      * 疲劳（fatigue）随 唤醒↓、效价↓、支配↓ 上升 —— 低能量 + 无力感；
      * 效价（valence）直接由 P 线性映射到 0~1。

    权重见 ``_DETAIL_WEIGHTS``。设计上保证：满量程组合落到 0 / 100 两端、
    中性点（P=A=D=0）恰好回到 50 / 0.5，因此即使权重标定不准，
    输出也不会越界或跳变。
    """
    p = float(pad.get("P", 0.0)) / PAD_RANGE
    a = float(pad.get("A", 0.0)) / PAD_RANGE
    d = float(pad.get("D", 0.0)) / PAD_RANGE

    def score(key: str) -> float:
        wp, wa, wd = _DETAIL_WEIGHTS[key]
        return _CENTER + _SCALE * (wp * p + wa * a + wd * d)

    return {
        "anxiety_level": round(clamp(score("anxiety_level")), 1),
        "stress_level": round(clamp(score("stress_level")), 1),
        "fatigue_level": round(clamp(score("fatigue_level")), 1),
        "valence": round(clamp(0.5 + 0.5 * p, 0.0, 1.0), 2),
        "source": "pad",
        "pad_confidence": pad.get("confidence"),
        "d_confidence": pad.get("d_confidence"),
    }


def detail_from_table(state: Dict) -> Dict:
    """查表兜底（PAD 不可用时）"""
    emotion = state.get("emotion", "")
    a, s, f, v = EMOTION_TABLE.get(emotion, (30, 30, 30, 0.5))
    conf = float(state.get("confidence") or 0.5)
    return {
        "anxiety_level": round(min(100, a * (0.7 + 0.3 * conf)), 1),
        "stress_level": round(min(100, s * (0.7 + 0.3 * conf)), 1),
        "fatigue_level": round(min(100, f * (0.7 + 0.3 * conf)), 1),
        "valence": round(v, 2),
        "source": "table",
    }


def detail_view(state: Optional[dict]) -> dict:
    """情绪状态 → 计划书 §5.2 的 detail 多维视图"""
    if not state:
        return {"anxiety_level": 0, "stress_level": 0, "fatigue_level": 0,
                "valence": 0.5, "source": "none"}
    if state.get('simulation') is False:
        return {'anxiety_level':None, 'stress_level':None, 'fatigue_level':None,
                'valence':None, 'source':'classification_only' if state.get('validated') else 'unknown',
                'raw_risk':state.get('risk_score') if state.get('validated') else None}
    pad = state.get("pad")
    out = (detail_from_pad(pad) if isinstance(pad, dict) and pad
           else detail_from_table(state))
    out["raw_risk"] = state.get("risk_score", 0)
    return out


def public_state(state: Optional[Dict]) -> Optional[Dict]:
    """对外状态视图：语义化为主，原始生理值仅调试模式可见

    这是 v2 的**有意设计**（与 v1 行为相反）：把 hr_bpm / rmssd_ms 这类数值
    直接交给上层或模型，会诱导出"心率 92 所以你很焦虑"式的伪精确表达，
    而实测已确认 PPG 存在「HR 可信但 HRV 是假的」这一隐蔽失效。
    默认只给语义区间 + 质量等级，让上层知道"这个数可不可信"。
    """
    if not state:
        return None
    out = {
        "timestamp": state.get("timestamp"),
        "emotion": state.get("emotion"),
        "confidence": state.get("confidence"),
        "risk_score": state.get("risk_score"),
        "risk_kind": 'engineering_trigger_score',
        "risk_band": band(state.get("risk_score")) if state.get('risk_score') is not None else '未知',
        "layer": state.get("layer"),
        "detail": detail_view(state),
        "quality": state.get("quality"),
        "quality_level": state.get("quality_level"),
        "jitter": state.get("jitter"),
        "pad": state.get("pad"),
        "affect": state.get("affect"),
        "privacy_notice": config.PRIVACY_NOTICE_VERSION,
        "validated": state.get('validated', False),
        "simulation": state.get('simulation', state.get('layer') == 'manual'),
        "sensor": state.get('sensor'),
        "model_available": state.get('model_available'),
        "calibration": state.get('calibration'),
        "reason": (state.get('detail') or {}).get('reason'),
    }
    if config.EXPOSE_RAW_METRICS:
        out["metrics"] = {
            "hr_bpm": state.get("hr_bpm"),
            "sdnn_ms": state.get("sdnn_ms"),
            "rmssd_ms": state.get("rmssd_ms"),
            "pnn50": state.get("pnn50"),
            "lf_hf": state.get("lf_hf"),
            "sd_hr_bpm": state.get("sd_hr_bpm"),
            "sig_quality": state.get("sig_quality"),
            "rel_features": state.get("rel_features"),
        }
    return out
