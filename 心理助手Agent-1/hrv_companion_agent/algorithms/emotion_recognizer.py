# -*- coding: utf-8 -*-
"""
情绪状态识别引擎（三层架构）
============================
对应 PPT STEP 03 / 计划书 §4.2 算法子模块 2。

Layer 1 — 规则状态机: HRV 阈值区间硬边界判断（快速，适合边缘部署）
Layer 2 — 模糊推理:     梯形隶属度函数处理边界模糊区（平滑概率输出）
Layer 3 — ML 分类器:    MAD 数据集训练的随机森林（高精度，需 sklearn/joblib）

情绪分类（5 类，与计划书 §4.2 规则表一致）:
  平静 / 低焦虑 / 高焦虑 / 愉悦 / 疲劳

与 mad_toolkit（7 类 MAD 标签）的映射（Layer 3 使用）:
  Neutral→平静, Happy→愉悦, Anxiety→高焦虑, Fearful→高焦虑,
  Sad→低焦虑, Angry→高焦虑, Disgust→低焦虑
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .. import config


# ------------------------------------------------------------------ 结果
@dataclass
class EmotionResult:
    """情绪识别结果（对应计划书 §4.2 输出格式）"""
    emotion: str            # 平静/低焦虑/高焦虑/愉悦/疲劳
    confidence: float       # 置信度 0-1
    risk_score: float       # 风险评分 0-100
    layer: str              # rule / fuzzy / ml / rule(fallback)
    detail: Dict = field(default_factory=dict)


# --------------------------------------------------------- Layer 1 规则层
# 参考区间来源: 计划书 §4.2 情绪分类规则参考表（需 20+ 人校准实验个性化）
EMOTION_RULES: Dict[str, Dict[str, tuple]] = {
    "平静": {"hr_bpm": (60, 75), "sdnn_ms": (50, 90), "rmssd_ms": (40, 70),
             "lf_hf": (1.0, 2.0), "rr_bpm": (12, 16)},
    "低焦虑": {"hr_bpm": (75, 90), "sdnn_ms": (30, 50), "rmssd_ms": (25, 40),
              "lf_hf": (2.0, 3.5), "rr_bpm": (16, 20)},
    "高焦虑": {"hr_bpm": (90, 110), "sdnn_ms": (10, 30), "rmssd_ms": (8, 25),
              "lf_hf": (3.5, 6.5), "rr_bpm": (20, 26)},
    "愉悦": {"hr_bpm": (65, 80), "sdnn_ms": (40, 60), "rmssd_ms": (35, 50),
            "lf_hf": (1.5, 2.5), "rr_bpm": (14, 18)},
    "疲劳": {"hr_bpm": (55, 70), "sdnn_ms": (35, 45), "rmssd_ms": (20, 30),
            "lf_hf": (1.0, 1.5), "rr_bpm": (10, 15)},
}

FEATURE_WEIGHTS = {"hr_bpm": 1.0, "sdnn_ms": 1.2, "rmssd_ms": 1.0,
                   "lf_hf": 0.8, "rr_bpm": 0.7}

# 情绪基础风险（用于风险评分）
EMOTION_BASE_RISK = {"平静": 10, "愉悦": 15, "疲劳": 40, "低焦虑": 70, "高焦虑": 90}


class RuleBasedClassifier:
    """Layer 1: 规则状态机 —— 加权区间匹配打分"""

    def classify(self, feats: Dict[str, float]) -> tuple:
        scores: Dict[str, float] = {}
        for emotion, rules in EMOTION_RULES.items():
            score = 0.0
            total_w = 0.0
            for feat_name, (lo, hi) in rules.items():
                if feat_name not in feats or feats[feat_name] is None:
                    continue
                val = float(feats[feat_name])
                if val <= 0:  # 特征缺失（=0 视为缺失）
                    continue
                w = FEATURE_WEIGHTS.get(feat_name, 1.0)
                if lo <= val <= hi:
                    score += w
                elif val < lo:
                    score += w * max(0.0, val / lo if lo > 0 else 0.0)
                else:
                    score += w * max(0.0, hi / val if val > 0 else 0.0)
                total_w += w
            scores[emotion] = (score / total_w) if total_w > 0 else 0.0

        best = max(scores, key=scores.get)
        return best, round(scores[best], 3), scores


# --------------------------------------------------------- Layer 2 模糊层
class FuzzyClassifier:
    """Layer 2: 模糊推理 —— 梯形隶属度 + 概率归一（处理边界模糊）"""

    def __init__(self):
        self.memberships: Dict[str, Dict[str, tuple]] = {}
        for emotion, rules in EMOTION_RULES.items():
            self.memberships[emotion] = {}
            for feat_name, (lo, hi) in rules.items():
                width = max((hi - lo) * 0.3, 2.0)  # 过渡区半宽
                self.memberships[emotion][feat_name] = (lo, hi, width)

    def classify(self, feats: Dict[str, float]) -> Dict:
        probs: Dict[str, float] = {}
        total = 0.0
        for emotion, mems in self.memberships.items():
            membership = 1.0
            for feat_name, (c_lo, c_hi, width) in mems.items():
                if feat_name not in feats or not feats[feat_name]:
                    continue
                val = float(feats[feat_name])
                if c_lo <= val <= c_hi:
                    mu = 1.0
                elif val < c_lo:
                    mu = max(0.0, 1 - (c_lo - val) / width)
                else:
                    mu = max(0.0, 1 - (val - c_hi) / width)
                membership *= mu
            probs[emotion] = membership
            total += membership

        if total > 1e-9:
            probs = {k: v / total for k, v in probs.items()}
        best = max(probs, key=probs.get)
        return {"emotion": best, "confidence": round(probs[best], 3),
                "probabilities": {k: round(v, 3)
                                  for k, v in sorted(probs.items(),
                                                     key=lambda x: -x[1])}}


# --------------------------------------------------------- Layer 3 ML 层
# MAD 7 类标签 → 本项目 5 类中文情绪
MAD_TO_ZH = {0: "平静", 1: "愉悦", 2: "高焦虑", 3: "高焦虑",
             4: "低焦虑", 5: "高焦虑", 6: "低焦虑"}
MAD_LABELS = {0: "Neutral", 1: "Happy", 2: "Anxiety", 3: "Fearful",
              4: "Sad", 5: "Angry", 6: "Disgust"}
ML_FEATURE_NAMES = ["hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct",
                    "lf_hf", "sd1_ms", "sd2_ms", "sd_ratio"]


class MLClassifier:
    """Layer 3: MAD 数据集随机森林（惰性加载，缺依赖/缺模型时自动禁用）"""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or config.ML_MODEL_PATH
        self._bundle = None
        self.available = False
        self._try_load()

    def _try_load(self):
        try:
            import joblib  # noqa: F401
            import os
            if not self.model_path or not os.path.exists(self.model_path):
                return
            self._bundle = joblib.load(self.model_path)
            if not isinstance(self._bundle, dict) or "model" not in self._bundle:
                return
            self.available = True
        except Exception:
            self.available = False

    def predict(self, feats: Dict[str, float]) -> Optional[Dict]:
        """
        返回 {emotion, confidence, mad_label} 或 None（不可用时）。

        标签映射策略（自描述优先）:
          * 新模型（真实数据训练）: bundle 携带 classes_zh 中文标签列表，
            按预测类别索引直接取中文情绪名；
          * 旧模型（MAD 7 类）: 无 classes_zh，回退到 MAD_TO_ZH 映射，保持兼容。
        """
        if not self.available:
            return None
        try:
            model = self._bundle["model"]
            scaler = self._bundle.get("scaler")
            names = self._bundle.get("feature_names", ML_FEATURE_NAMES)
            row = [[float(feats.get(n, 0) or 0) for n in names]]
            if scaler is not None:
                row = scaler.transform(row)
            eid = int(model.predict(row)[0])
            proba = model.predict_proba(row)[0] if hasattr(model, "predict_proba") else None
            conf = float(max(proba)) if proba is not None and len(proba) else 0.5

            classes_zh = self._bundle.get("classes_zh") or []
            if classes_zh and 0 <= eid < len(classes_zh):
                # 新模型: 自描述中文标签
                zh = classes_zh[eid]
                raw = self._bundle.get("label_map", {}).get(eid) or \
                    MAD_LABELS.get(eid, "Unknown")
            else:
                # 旧 MAD 模型兼容
                zh = MAD_TO_ZH.get(eid, "平静")
                raw = MAD_LABELS.get(eid, "Unknown")
            return {"emotion": zh,
                    "confidence": round(conf, 3),
                    "mad_label": raw}
        except Exception:
            return None


# ------------------------------------------------------------ 融合引擎
class EmotionRecognizer:
    """
    三层融合情绪识别引擎（完整版）
    流程: Layer1(规则,≥0.7 直接输出) → Layer2(模糊,≥0.6 输出)
          → Layer3(ML, 兜底) → 失败回退规则层
    """

    def __init__(self, model_path: Optional[str] = None):
        self.rule = RuleBasedClassifier()
        self.fuzzy = FuzzyClassifier()
        self.ml = MLClassifier(model_path)

    def recognize(self, feats: Dict[str, float]) -> EmotionResult:
        # Layer 1
        name_r, conf_r, rule_scores = self.rule.classify(feats)
        if conf_r >= config.RULE_CONFIDENCE_FLOOR:
            return EmotionResult(name_r, conf_r, self._risk(name_r, conf_r, feats),
                                 "rule", {"rule_score": conf_r})

        # Layer 2
        fuzzy_out = self.fuzzy.classify(feats)
        if fuzzy_out["confidence"] >= config.FUZZY_CONFIDENCE_FLOOR:
            return EmotionResult(fuzzy_out["emotion"], fuzzy_out["confidence"],
                                 self._risk(fuzzy_out["emotion"],
                                            fuzzy_out["confidence"], feats),
                                 "fuzzy",
                                 {"rule_score": conf_r,
                                  "fuzzy_score": fuzzy_out["confidence"],
                                  "probabilities": fuzzy_out["probabilities"]})

        # Layer 3
        ml_out = self.ml.predict(feats)
        if ml_out:
            return EmotionResult(ml_out["emotion"], ml_out["confidence"],
                                 self._risk(ml_out["emotion"],
                                            ml_out["confidence"], feats),
                                 "ml",
                                 {"rule_score": conf_r,
                                  "fuzzy_score": fuzzy_out["confidence"],
                                  "ml_score": ml_out["confidence"],
                                  "mad_label": ml_out["mad_label"]})

        # 全层失败 → 回退规则层
        return EmotionResult(name_r, conf_r, self._risk(name_r, conf_r, feats),
                             "rule(fallback)", {"rule_score": conf_r})

    @staticmethod
    def _risk(emotion: str, confidence: float, feats: Dict[str, float]) -> float:
        """综合风险评分 [0,100]（情绪基础分 + 生理偏离修正）"""
        base = EMOTION_BASE_RISK.get(emotion, 30)
        hr = float(feats.get("hr_bpm") or 70)
        hr_dev = abs(hr - 70) / 50.0
        sdnn = float(feats.get("sdnn_ms") or 50)
        sdnn_risk = max(0.0, 1 - sdnn / 60.0)
        lf_hf = float(feats.get("lf_hf") or 1.5)
        lf_risk = min(1.0, max(0.0, (lf_hf - 2.0) / 4.0))
        risk = base * (0.5 + 0.25 * hr_dev + 0.15 * sdnn_risk + 0.10 * lf_risk)
        risk *= (0.7 + 0.3 * confidence)
        return round(min(100.0, max(0.0, risk)), 1)


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    cases = [
        {"hr_bpm": 68, "sdnn_ms": 62, "rmssd_ms": 48, "lf_hf": 1.6, "rr_bpm": 14,
         "pnn50_pct": 18, "sd1_ms": 34, "sd2_ms": 80, "sd_ratio": 0.42},
        {"hr_bpm": 96, "sdnn_ms": 24, "rmssd_ms": 16, "lf_hf": 4.2, "rr_bpm": 22,
         "pnn50_pct": 3, "sd1_ms": 11, "sd2_ms": 40, "sd_ratio": 0.28},
        {"hr_bpm": 78, "sdnn_ms": 36, "rmssd_ms": 22, "lf_hf": 2.8, "rr_bpm": 17,
         "pnn50_pct": 7, "sd1_ms": 16, "sd2_ms": 52, "sd_ratio": 0.30},
        {"hr_bpm": 62, "sdnn_ms": 40, "rmssd_ms": 26, "lf_hf": 1.2, "rr_bpm": 12,
         "pnn50_pct": 6, "sd1_ms": 18, "sd2_ms": 55, "sd_ratio": 0.33},
    ]
    rec = EmotionRecognizer()
    for i, c in enumerate(cases, 1):
        r = rec.recognize(c)
        print(f"案例{i}: {r.emotion} conf={r.confidence} risk={r.risk_score} layer={r.layer}")
