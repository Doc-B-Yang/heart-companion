# -*- coding: utf-8 -*-
"""
算法层：HRV 特征提取 / 情绪识别 / PAD 维度 / 情感动力学 / 主动触发决策
对应 PPT STEP 02-04、计划书 §4.2

模块划分（v2 专业化扩展）:
  * ``hrv_extractor``    HRV 特征提取（时域/频域/非线性 + 个体基线相对化 + 质量门禁）
  * ``emotion_recognizer`` 三层情绪识别（规则 → 模糊 → ML 随机森林）
  * ``pad_model``        PAD 三维连续情感回归（规则映射 + PCA-SVR）
  * ``emotion_dynamics`` PWE 情感动力学状态机（情绪强度/持续/衰减 + 心境累积）
  * ``trigger_engine``   主动触发决策（综合风险 + 分级 + 冷却 + 反馈自适应）
"""
from .hrv_extractor import HRVExtractor
from .emotion_recognizer import EmotionRecognizer, EmotionResult
from .pad_model import (PADRegressor, PADResult, RulePAD, estimate_pad,
                        pad_from_questionnaire, pad_to_emotion)
from .emotion_dynamics import (AffectState, EmotionDynamics,
                               PERSONALITY_PROFILES, WHEEL_EMOTIONS)
from .trigger_engine import TriggerEngine, TriggerDecision

__all__ = [
    "HRVExtractor",
    "EmotionRecognizer", "EmotionResult",
    "PADRegressor", "PADResult", "RulePAD", "estimate_pad",
    "pad_from_questionnaire", "pad_to_emotion",
    "AffectState", "EmotionDynamics", "PERSONALITY_PROFILES", "WHEEL_EMOTIONS",
    "TriggerEngine", "TriggerDecision",
]
