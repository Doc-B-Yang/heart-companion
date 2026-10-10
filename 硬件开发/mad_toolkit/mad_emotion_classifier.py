"""
三层情绪识别引擎
===============
对应 PPT 中 STEP 03: 情绪状态识别引擎

三层架构:
  Layer 1 — 规则状态机: HRV 阈值 → 情绪边界判断 (快速, 边缘部署)
  Layer 2 — 模糊推理:     处理边界模糊区 (隶属度函数平滑过渡)
  Layer 3 — ML 分类器:    基于 MAD 数据集训练的 SVM/随机森林 (高精度)

情绪分类 (7类, 对应 MAD 数据集):
  0: Neutral   (中性)
  1: Happy     (开心)
  2: Anxiety   (焦虑) ← 核心目标
  3: Fearful   (恐惧)
  4: Sad       (悲伤)
  5: Angry     (愤怒)
  6: Disgust   (厌恶)

用法:
  from mad_emotion_classifier import EmotionClassifier
  clf = EmotionClassifier()
  result = clf.recognize(hrv_features)
  print(result["emotion"])  # "Anxiety"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 情绪标签
EMOTION_NAMES = {
    0: "Neutral", 1: "Happy", 2: "Anxiety",
    3: "Fearful", 4: "Sad", 5: "Angry", 6: "Disgust",
}


@dataclass
class EmotionResult:
    """情绪识别结果"""
    emotion: str           # 情绪名称
    emotion_id: int        # 情绪编号
    confidence: float      # 置信度 (0-1)
    risk_score: float      # 风险评分 (0-100, 焦虑/恐惧权重高)
    layer: str             # 来自哪一层: "rule"/"fuzzy"/"ml"
    detail: Dict = field(default_factory=dict)


class RuleBasedClassifier:
    """
    Layer 1: 规则状态机

    基于 HRV 阈值区间的快速情绪判断。
    优点: 计算开销小、可解释、适合 ESP32 边缘部署
    缺点: 硬边界、无法处理个体差异
    """

    # 情绪 → 特征范围 {feature: (min, max)}
    # 来源: 文献综述 + DEAP/WESAD 数据集的统计值
    # 单位: HR(bpm), SDNN(ms), RMSSD(ms), LF/HF(ratio)
    RULES = {
        "Neutral": {
            "hr_bpm":   (60, 78),
            "sdnn_ms":  (40, 80),
            "rmssd_ms": (30, 60),
            "lf_hf":    (0.8, 2.5),
        },
        "Happy": {
            "hr_bpm":   (65, 85),
            "sdnn_ms":  (35, 65),
            "rmssd_ms": (28, 55),
            "lf_hf":    (1.5, 3.0),
        },
        "Anxiety": {
            "hr_bpm":   (80, 110),
            "sdnn_ms":  (15, 40),
            "rmssd_ms": (10, 30),
            "lf_hf":    (2.5, 6.0),
        },
        "Fearful": {
            "hr_bpm":   (85, 120),
            "sdnn_ms":  (10, 35),
            "rmssd_ms": (8, 25),
            "lf_hf":    (3.0, 7.0),
        },
        "Sad": {
            "hr_bpm":   (55, 72),
            "sdnn_ms":  (30, 55),
            "rmssd_ms": (25, 45),
            "lf_hf":    (0.8, 2.0),
        },
        "Angry": {
            "hr_bpm":   (80, 105),
            "sdnn_ms":  (20, 40),
            "rmssd_ms": (15, 32),
            "lf_hf":    (2.0, 4.5),
        },
        "Disgust": {
            "hr_bpm":   (62, 80),
            "sdnn_ms":  (30, 50),
            "rmssd_ms": (25, 42),
            "lf_hf":    (1.0, 2.5),
        },
    }

    # 情绪权重 (用于匹配打分)
    FEATURE_WEIGHTS = {
        "hr_bpm": 1.0,
        "sdnn_ms": 1.2,    # SDNN 最重要
        "rmssd_ms": 1.0,
        "lf_hf": 0.8,      # LF/HF 容易受干扰
    }

    def classify(self, features: Dict) -> Tuple[int, float, str]:
        """
        规则匹配 → 最佳情绪

        Args:
            features: HRV 特征字典 (包含 hr_bpm, sdnn_ms, rmssd_ms, lf_hf)

        Returns:
            (emotion_id, confidence, emotion_name)
        """
        scores = {}

        for emotion_name, rules in self.RULES.items():
            score = 0.0
            total_weight = 0.0

            for feat, (lo, hi) in rules.items():
                if feat not in features:
                    continue
                val = features[feat]
                w = self.FEATURE_WEIGHTS.get(feat, 1.0)

                # 评分: 在范围内 = 满分, 越远分越低
                if lo <= val <= hi:
                    score += w
                elif val < lo:
                    # 越远离边界分越低
                    ratio = max(0, val / lo) if lo > 0 else 0
                    score += w * ratio
                else:  # val > hi
                    ratio = max(0, hi / val) if val > 0 else 0
                    score += w * ratio

                total_weight += w

            # 归一化
            if total_weight > 0:
                scores[emotion_name] = score / total_weight
            else:
                scores[emotion_name] = 0.0

        # 取最高分
        best_emotion = max(scores, key=scores.get)
        confidence = scores[best_emotion]

        # 映射到 ID
        emotion_id_map = {
            "Neutral": 0, "Happy": 1, "Anxiety": 2,
            "Fearful": 3, "Sad": 4, "Angry": 5, "Disgust": 6,
        }
        emotion_id = emotion_id_map.get(best_emotion, 0)

        return emotion_id, round(confidence, 3), best_emotion


class FuzzyClassifier:
    """
    Layer 2: 模糊推理分类器

    处理 Layer 1 的边界模糊问题:
    - 当多个情绪得分接近时(间距<0.15)，使用隶属度函数平滑过渡
    - 输出概率分布而非硬分类

    隶属度函数: 三角形/梯形，为每个情绪定义"典型区"和"过渡区"
    """

    def __init__(self):
        # 隶属度函数参数: (典型区_左, 典型区_右, 过渡区_半宽)
        # 格式: {emotion: {feature: (center_lo, center_hi, transition_width)}}
        self.memberships = self._build_memberships()

    def _build_memberships(self) -> Dict:
        """构建隶属度函数"""
        mem = {}
        for emotion, rules in RuleBasedClassifier.RULES.items():
            mem[emotion] = {}
            for feat, (lo, hi) in rules.items():
                # 过渡区宽度 = 范围的 30%
                width = max((hi - lo) * 0.3, 2.0)
                mem[emotion][feat] = (lo, hi, width)
        return mem

    def classify(self, features: Dict) -> Dict:
        """
        模糊推理 → 情绪概率分布

        Returns:
            {
                "emotion_id": int,
                "emotion": str,
                "confidence": float,
                "probabilities": {emotion_name: probability, ...}
            }
        """
        probs = {}
        total_membership = 0.0

        for emotion, mem_params in self.memberships.items():
            membership = 1.0
            for feat, (c_lo, c_hi, width) in mem_params.items():
                if feat not in features:
                    continue
                val = features[feat]

                # 梯形隶属度
                if c_lo <= val <= c_hi:
                    mu = 1.0  # 在典型区内
                elif val < c_lo:
                    mu = max(0, 1 - (c_lo - val) / width)
                else:
                    mu = max(0, 1 - (val - c_hi) / width)

                membership *= mu

            probs[emotion] = membership
            total_membership += membership

        # 归一化
        if total_membership > 0:
            for e in probs:
                probs[e] /= total_membership

        best = max(probs, key=probs.get)
        return {
            "emotion_id": {
                "Neutral": 0, "Happy": 1, "Anxiety": 2,
                "Fearful": 3, "Sad": 4, "Angry": 5, "Disgust": 6,
            }.get(best, 0),
            "emotion": best,
            "confidence": round(probs[best], 3),
            "probabilities": {k: round(v, 3) for k, v in sorted(probs.items(), key=lambda x: -x[1])},
        }


class MLClassifier:
    """
    Layer 3: 机器学习分类器

    在 MAD 数据集上训练 SVM 和随机森林模型。
    当 Layer 1+2 的置信度低于阈值时激活。

    特征向量 (8维):
      [hr_bpm, sdnn_ms, rmssd_ms, pnn50_pct, lf_hf, sd1_ms, sd2_ms, sd_ratio]
    """

    FEATURE_NAMES = [
        "hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct",
        "lf_hf", "sd1_ms", "sd2_ms", "sd_ratio",
    ]

    def __init__(self, model_type: str = "rf"):
        """
        Args:
            model_type: "rf" (随机森林, 推荐), 或 "svm"
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False

    def _create_model(self):
        """创建模型实例"""
        if self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        else:
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        训练 ML 分类器

        Args:
            X: 特征矩阵 (n_samples × 8)
            y: 标签数组 (n_samples,)

        Returns:
            训练报告 (准确率、分类报告等)
        """
        print(f"\n[ML 训练] 样本数: {len(X)}, 特征维度: {X.shape[1]}")
        print(f"  模型: {'RandomForest' if self.model_type == 'rf' else 'SVM'}")

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        # 交叉验证
        self.model = self._create_model()
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"  5-fold CV 准确率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # 全量训练
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # 评估
        y_pred = self.model.predict(X_scaled)
        report = classification_report(y, y_pred, target_names=[EMOTION_NAMES.get(i, f"C{i}") for i in sorted(set(y))],
                                       zero_division=0)

        return {
            "cv_accuracy_mean": round(cv_scores.mean(), 3),
            "cv_accuracy_std": round(cv_scores.std(), 3),
            "report": report,
        }

    def fit(self, X: np.ndarray, y: np.ndarray):
        """简易训练接口 (不打印报告)"""
        X_scaled = self.scaler.fit_transform(X)
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, features: Dict) -> Tuple[int, float, np.ndarray]:
        """
        预测单个样本的情绪

        Args:
            features: HRV 特征字典

        Returns:
            (emotion_id, confidence, probabilities_array)
        """
        if not self.is_trained or self.model is None:
            return 0, 0.0, np.zeros(7)

        # 提取特征向量
        x = np.array([features.get(f, 0) for f in self.FEATURE_NAMES]).reshape(1, -1)
        x_scaled = self.scaler.transform(x)

        emotion_id = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0]

        confidence = float(proba[emotion_id]) if emotion_id < len(proba) else 0.0
        return int(emotion_id), confidence, proba

    def features_to_array(self, features_list: List[Dict]) -> np.ndarray:
        """将特征字典列表转换为 numpy 数组"""
        X = []
        for f in features_list:
            row = [f.get(name, 0) for name in self.FEATURE_NAMES]
            X.append(row)
        return np.array(X)

    def save(self, path: str):
        """保存模型"""
        data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": self.FEATURE_NAMES,
        }
        joblib.dump(data, path)
        print(f"[ML] 模型已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "MLClassifier":
        """加载模型"""
        data = joblib.load(path)
        clf = cls(model_type=data["model_type"])
        clf.model = data["model"]
        clf.scaler = data["scaler"]
        clf.is_trained = True
        print(f"[ML] 模型已加载: {path}")
        return clf


class EmotionClassifier:
    """
    三层融合情绪识别引擎 (完整版)

    工作流程:
    1. Layer 1 (规则) → 如果 confidence > 0.7, 直接返回
    2. Layer 2 (模糊) → 如果 confidence > 0.6, 返回
    3. Layer 3 (ML)   → 最终分类

    风险评分计算:
    - Anxiety/Fearful 权重高
    - 结合 confidence 和 HRV 偏移量
    """

    # 情绪 → 风险权重
    RISK_WEIGHTS = {
        0: 5,   # Neutral: 极低风险
        1: 10,  # Happy: 低风险
        2: 75,  # Anxiety: 高风险 ← 核心
        3: 85,  # Fearful: 极高风险
        4: 50,  # Sad: 中高风险
        5: 60,  # Angry: 中高风险
        6: 30,  # Disgust: 中低风险
    }

    def __init__(self):
        self.rule_clf = RuleBasedClassifier()
        self.fuzzy_clf = FuzzyClassifier()
        self.ml_clf = None  # 训练后才赋值

    def set_ml_model(self, ml_classifier: MLClassifier):
        """注入训练好的 ML 模型"""
        self.ml_clf = ml_classifier

    def recognize(self, features: Dict) -> EmotionResult:
        """
        三层融合识别 (单样本)

        Args:
            features: HRV 特征字典

        Returns:
            EmotionResult 对象
        """
        # Layer 1: 规则
        eid_r, conf_r, name_r = self.rule_clf.classify(features)

        if conf_r >= 0.7:
            # 规则层高置信度 → 直接返回
            risk = self._calc_risk(eid_r, conf_r, features)
            return EmotionResult(
                emotion=name_r,
                emotion_id=eid_r,
                confidence=conf_r,
                risk_score=risk,
                layer="rule",
                detail={"rule_score": conf_r}
            )

        # Layer 2: 模糊推理
        fuzzy_result = self.fuzzy_clf.classify(features)
        eid_f = fuzzy_result["emotion_id"]
        conf_f = fuzzy_result["confidence"]

        if conf_f >= 0.6:
            risk = self._calc_risk(eid_f, conf_f, features)
            return EmotionResult(
                emotion=fuzzy_result["emotion"],
                emotion_id=eid_f,
                confidence=conf_f,
                risk_score=risk,
                layer="fuzzy",
                detail={
                    "rule_score": conf_r,
                    "fuzzy_score": conf_f,
                    "probabilities": fuzzy_result["probabilities"],
                }
            )

        # Layer 3: ML (如果已训练)
        if self.ml_clf and self.ml_clf.is_trained:
            eid_m, conf_m, proba = self.ml_clf.predict(features)
            risk = self._calc_risk(eid_m, conf_m, features)
            return EmotionResult(
                emotion=EMOTION_NAMES.get(eid_m, "Unknown"),
                emotion_id=eid_m,
                confidence=conf_m,
                risk_score=risk,
                layer="ml",
                detail={
                    "rule_score": conf_r,
                    "fuzzy_score": conf_f,
                    "ml_score": conf_m,
                    "ml_probabilities": proba.tolist() if len(proba) > 0 else [],
                }
            )

        # 所有层都失败 → 回退到规则层
        risk = self._calc_risk(eid_r, conf_r, features)
        return EmotionResult(
            emotion=name_r,
            emotion_id=eid_r,
            confidence=conf_r,
            risk_score=risk,
            layer="rule(fallback)",
            detail={}
        )

    def recognize_batch(self, features_list: List[Dict]) -> List[EmotionResult]:
        """批量识别"""
        return [self.recognize(f) for f in features_list]

    def train_ml_layer(self, X: np.ndarray, y: np.ndarray,
                       model_type: str = "rf") -> Dict:
        """训练 Layer 3 (ML 分类器)"""
        self.ml_clf = MLClassifier(model_type=model_type)
        report = self.ml_clf.train(X, y)
        return report

    def save_ml_model(self, path: str):
        """保存 ML 模型"""
        if self.ml_clf:
            self.ml_clf.save(path)

    def load_ml_model(self, path: str):
        """加载 ML 模型"""
        self.ml_clf = MLClassifier.load(path)

    def _calc_risk(self, emotion_id: int, confidence: float,
                   features: Dict) -> float:
        """计算综合风险评分 [0, 100]"""
        base_risk = self.RISK_WEIGHTS.get(emotion_id, 30)

        # HR 偏离正常值 (70 bpm) 越多, 风险越高
        hr = features.get("hr_bpm", 70)
        hr_deviation = abs(hr - 70) / 50.0  # 归一化到0-1

        # SDNN 越低风险越高
        sdnn = features.get("sdnn_ms", 50)
        sdnn_risk = max(0, 1 - sdnn / 60.0)

        # 综合
        risk = base_risk * (0.5 + 0.3 * hr_deviation + 0.2 * sdnn_risk)
        risk *= (0.7 + 0.3 * confidence)  # 置信度越高越确定

        return min(100, max(0, risk))


# ── 快速测试 ──
if __name__ == "__main__":
    print("=" * 60)
    print("三层情绪识别引擎 - 快速测试")
    print("=" * 60)

    # 模拟不同情绪的 HRV 特征
    test_cases = [
        {"hr_bpm": 72, "sdnn_ms": 55, "rmssd_ms": 42, "lf_hf": 1.8, "pnn50_pct": 18},
        {"hr_bpm": 92, "sdnn_ms": 28, "rmssd_ms": 20, "lf_hf": 3.8, "pnn50_pct": 5},
        {"hr_bpm": 75, "sdnn_ms": 50, "rmssd_ms": 38, "lf_hf": 2.2, "pnn50_pct": 14},
        {"hr_bpm": 65, "sdnn_ms": 42, "rmssd_ms": 34, "lf_hf": 1.5, "pnn50_pct": 10},
    ]

    expected = ["Neutral", "Anxiety", "Happy", "Sad"]

    clf = EmotionClassifier()

    for i, features in enumerate(test_cases):
        result = clf.recognize(features)
        match = "✅" if result.emotion == expected[i] else "❌"
        print(f"\n{match} 测试{i+1}: {features}")
        print(f"   预测: {result.emotion} (期望: {expected[i]})")
        print(f"   置信度: {result.confidence:.3f} | 层: {result.layer} | 风险: {result.risk_score:.0f}")
