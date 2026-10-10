# -*- coding: utf-8 -*-
"""主动触发决策引擎单元测试（对应计划书 §4.2 算法子模块3 / PPT STEP 04）"""
import unittest

from hrv_companion_agent.algorithms.emotion_recognizer import EmotionResult
from hrv_companion_agent.algorithms.trigger_engine import TriggerEngine


def er(emotion, risk, hr=70):
    return EmotionResult(emotion, 0.8, risk, "rule", {"hr_bpm": hr})


class TestTriggerEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TriggerEngine(cooldown_sec=300, quiet_checkin_sec=7200)

    def test_no_trigger_low_risk(self):
        d = self.engine.decide(er("平静", 20), [15, 18, 20], now=1000.0)
        self.assertFalse(d.should_trigger)
        self.assertEqual(d.trigger_level, "none")

    def test_low_trigger(self):
        d = self.engine.decide(er("低焦虑", 68), [30, 35, 40], now=1000.0)
        self.assertTrue(d.should_trigger)
        self.assertIn(d.trigger_level, ("low", "medium"))

    def test_high_trigger(self):
        # 平缓的高风险（无恶化趋势）→ high 等级
        d = self.engine.decide(er("高焦虑", 93), [88, 90, 90], now=1000.0)
        self.assertTrue(d.should_trigger)
        self.assertEqual(d.trigger_level, "high")

    def test_emergency_overrides_cooldown(self):
        d = self.engine.decide(er("高焦虑", 99), [40, 50, 60],
                               last_trigger_time=999.0, now=1000.0)
        self.assertTrue(d.should_trigger)
        self.assertEqual(d.trigger_level, "emergency")

    def test_cooldown_suppresses(self):
        d = self.engine.decide(er("低焦虑", 70), [40, 50, 60],
                               last_trigger_time=990.0, now=1000.0)
        self.assertFalse(d.should_trigger)

    def test_quiet_checkin(self):
        """长时间无交互 + 低风险 → 陪伴式低频问候"""
        d = self.engine.decide(er("平静", 15), [10, 12, 15],
                               last_interaction_time=1000.0 - 8000,
                               now=1000.0)
        self.assertTrue(d.should_trigger)
        self.assertEqual(d.trigger_level, "quiet_checkin")

    def test_trend_amplifies(self):
        """风险上升趋势 → 综合风险高于当前值"""
        rising = [30, 40, 50, 60, 70, 80]
        d = self.engine.decide(er("低焦虑", 70), rising, now=1000.0)
        self.assertGreater(d.risk, 70)

    def test_feedback_adaptive_threshold(self):
        self.engine.record_feedback("active_dialogue", True)
        self.assertEqual(self.engine.personal_offsets["trigger"], 0.5)
        eff = self.engine.strategy_effectiveness("active_dialogue")
        self.assertGreater(eff, 0.5)
        self.engine.record_feedback("active_dialogue", False)
        eff = self.engine.strategy_effectiveness("active_dialogue")
        self.assertEqual(eff, 0.5)


if __name__ == "__main__":
    unittest.main()
