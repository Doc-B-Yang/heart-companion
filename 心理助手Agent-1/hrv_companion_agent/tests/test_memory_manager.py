# -*- coding: utf-8 -*-
"""长期记忆管理器单元测试（对应计划书 §4.3 memory_manager.py）"""
import unittest

from hrv_companion_agent.agent.memory_manager import MemoryManager


class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.mm = MemoryManager("test_user", db_path=":memory:")

    def test_profile_roundtrip(self):
        self.mm.update_user_profile({"preferences": {"称呼": "同学"}})
        self.assertEqual(self.mm.get_user_profile()["preferences"]["称呼"], "同学")

    def test_interaction_roundtrip(self):
        self.mm.save_interaction("我今天压力好大", "先做个深呼吸吧",
                                 {"emotion": "高焦虑", "risk_score": 85})
        recent = self.mm.get_recent_interactions(limit=10)
        roles = [r["role"] for r in recent]
        self.assertIn("user", roles)
        self.assertIn("assistant", roles)

    def test_emotional_timeline(self):
        self.mm.record_emotion_state({"emotion": "高焦虑", "confidence": 0.8,
                                      "risk_score": 85, "hr_bpm": 95})
        tl = self.mm.get_emotional_timeline(days=1)
        self.assertEqual(len(tl), 1)
        self.assertEqual(tl[0]["emotion"], "高焦虑")

    def test_semantic_search(self):
        self.mm.save_interaction("明天考试好焦虑", "试试呼吸练习",
                                 {"emotion": "低焦虑", "risk_score": 60})
        hits = self.mm.semantic_search("考试")
        self.assertTrue(any("焦虑" in h["content"] for h in hits))

    def test_baseline_ema(self):
        self.mm.update_baseline({"hr_bpm": 70, "sdnn_ms": 50,
                                 "rmssd_ms": 40, "lf_hf": 1.5})
        self.mm.update_baseline({"hr_bpm": 90, "sdnn_ms": 30,
                                 "rmssd_ms": 20, "lf_hf": 3.0})
        base = self.mm.get_baseline()
        # EMA 更新: 70 与 90 之间
        self.assertGreater(base["hr_bpm"], 70)
        self.assertLess(base["hr_bpm"], 90)

    def test_risk_history(self):
        for risk in (30, 40, 50):
            self.mm.record_emotion_state({"emotion": "低焦虑", "risk_score": risk})
        self.assertEqual(self.mm.get_risk_history(), [30.0, 40.0, 50.0])

    def test_stats(self):
        self.mm.save_interaction("hi", "hello")
        self.mm.record_emotion_state({"emotion": "平静", "risk_score": 10})
        stats = self.mm.stats()
        self.assertEqual(stats["interactions"], 2)
        self.assertEqual(stats["emotion_records"], 1)


if __name__ == "__main__":
    unittest.main()
