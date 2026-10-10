# -*- coding: utf-8 -*-
"""三层情绪识别引擎单元测试（对应计划书 §4.2 算法子模块2 / PPT STEP 03）"""
import unittest

from hrv_companion_agent.algorithms.emotion_recognizer import (
    EmotionRecognizer, RuleBasedClassifier, FuzzyClassifier)

CALM = {"hr_bpm": 68, "sdnn_ms": 62, "rmssd_ms": 48, "lf_hf": 1.6,
        "rr_bpm": 14, "pnn50_pct": 18, "sd1_ms": 34, "sd2_ms": 80,
        "sd_ratio": 0.42}
HIGH_ANX = {"hr_bpm": 96, "sdnn_ms": 24, "rmssd_ms": 16, "lf_hf": 4.2,
            "rr_bpm": 22, "pnn50_pct": 3, "sd1_ms": 11, "sd2_ms": 40,
            "sd_ratio": 0.28}
LOW_ANX = {"hr_bpm": 78, "sdnn_ms": 36, "rmssd_ms": 22, "lf_hf": 2.8,
           "rr_bpm": 17, "pnn50_pct": 7, "sd1_ms": 16, "sd2_ms": 52,
           "sd_ratio": 0.30}
FATIGUE = {"hr_bpm": 62, "sdnn_ms": 40, "rmssd_ms": 26, "lf_hf": 1.2,
           "rr_bpm": 12, "pnn50_pct": 6, "sd1_ms": 18, "sd2_ms": 55,
           "sd_ratio": 0.33}


class TestRuleLayer(unittest.TestCase):
    def setUp(self):
        self.rule = RuleBasedClassifier()

    def test_calm(self):
        name, conf, _ = self.rule.classify(CALM)
        self.assertEqual(name, "平静")

    def test_high_anxiety(self):
        name, conf, _ = self.rule.classify(HIGH_ANX)
        self.assertEqual(name, "高焦虑")

    def test_missing_features_handled(self):
        name, conf, _ = self.rule.classify({"hr_bpm": 70})
        self.assertIn(name, ("平静", "低焦虑", "高焦虑", "愉悦", "疲劳"))


class TestFuzzyLayer(unittest.TestCase):
    def setUp(self):
        self.fuzzy = FuzzyClassifier()

    def test_probabilities_sum_to_one(self):
        out = self.fuzzy.classify(HIGH_ANX)
        total = sum(out["probabilities"].values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_high_anxiety_detected(self):
        out = self.fuzzy.classify(HIGH_ANX)
        self.assertEqual(out["emotion"], "高焦虑")


class TestEmotionRecognizer(unittest.TestCase):
    def setUp(self):
        self.rec = EmotionRecognizer()

    def test_calm_result(self):
        r = self.rec.recognize(CALM)
        self.assertEqual(r.emotion, "平静")
        self.assertGreaterEqual(r.confidence, 0.0)
        self.assertLessEqual(r.risk_score, 100)
        self.assertIn(r.layer, ("rule", "fuzzy", "ml", "rule(fallback)"))

    def test_high_anxiety_high_risk(self):
        r = self.rec.recognize(HIGH_ANX)
        self.assertEqual(r.emotion, "高焦虑")
        self.assertGreater(r.risk_score, 50)

    def test_calm_low_risk(self):
        r = self.rec.recognize(CALM)
        self.assertLess(r.risk_score, 40)

    def test_risk_bounds(self):
        for feats in (CALM, HIGH_ANX, LOW_ANX, FATIGUE):
            r = self.rec.recognize(feats)
            self.assertTrue(0 <= r.risk_score <= 100)

    def test_ml_layer_optional(self):
        """ML 模型加载失败/缺失时不能影响主流程"""
        rec = EmotionRecognizer(model_path="nonexistent_path.joblib")
        r = rec.recognize(CALM)
        self.assertIn(r.emotion, ("平静", "低焦虑", "高焦虑", "愉悦", "疲劳"))


if __name__ == "__main__":
    unittest.main()
