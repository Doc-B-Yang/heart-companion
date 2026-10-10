# -*- coding: utf-8 -*-
"""核心闭环系统端到端测试（离线模式，无需 API Key / 硬件）"""
import unittest

from hrv_companion_agent.core.simulator import SensorSimulator
from hrv_companion_agent.core.system import HeartCompanionSystem


class TestHeartCompanionSystem(unittest.TestCase):
    """离线模式下的完整闭环: 模拟BLE → 识别 → 触发 → 关怀 → 记忆 → 反馈"""

    def setUp(self):
        # 强制离线模式 + 关闭 TTS（保证测试不依赖网络）
        import hrv_companion_agent.config as config
        self._saved_key = config.LLM_API_KEY
        self._saved_tts = config.TTS_ENABLED
        config.LLM_API_KEY = ""
        config.TTS_ENABLED = False
        self.system = HeartCompanionSystem(user_id="test_user")

    def tearDown(self):
        import hrv_companion_agent.config as config
        config.LLM_API_KEY = self._saved_key
        config.TTS_ENABLED = self._saved_tts
        self.system.close()

    def test_full_closed_loop(self):
        system = self.system
        sim = SensorSimulator(scenario="stress_event", seed=42)
        states, triggers = [], []
        for i, pkt in enumerate(sim.run_all()):
            result = system.tick(pkt)
            if result is None:
                continue
            state = result["state"]
            if state["emotion"] != "数据积累中":
                states.append(state["emotion"])
            if result.get("should_trigger"):
                triggers.append(result)

        # 1. 情绪识别: 高焦虑阶段应被识别出来
        self.assertIn("高焦虑", states)
        self.assertGreater(len(states), 10)

        # 2. 主动触发: 整个压力场景至少触发一次主动关怀
        self.assertGreaterEqual(len(triggers), 1)
        self.assertTrue(triggers[0]["message"])
        self.assertIn(triggers[0]["trigger_level"],
                      ("low", "medium", "high", "emergency"))

        # 3. 记忆: 情绪时间线已沉淀
        timeline = system.get_timeline(days=1)
        self.assertGreater(len(timeline), 10)

        # 4. 反馈闭环: 干预策略评估可记录
        fb = system.feedback(after_risk=30)
        self.assertTrue(fb["recorded"])
        self.assertIn("trigger_threshold_offset", fb)

    def test_chat_offline(self):
        reply = self.system.chat("我今天考试压力好大")
        self.assertTrue(reply["response"])
        self.assertEqual(reply["emotion_adapted"], False)

    def test_crisis_chat_offline(self):
        """危机文本 → 自动危机干预（含热线资源）"""
        reply = self.system.chat("我最近总想伤害自己")
        self.assertIn("400-161-9995", reply["response"])

    def test_state_guards_small_window(self):
        """数据不足时输出'数据积累中'而非误判"""
        pkt = {"t": 1000, "hr": 72, "ibi": [800, 810], "sdnn": None,
               "rmssd": None, "lf_hf": None, "rr": None}
        state = self.system.ingest_sensor(pkt)
        self.assertEqual(state["emotion"], "不确定")
        self.assertFalse(state['validated'])
        self.assertFalse(state['quality']['quality_ok'])

    def test_profile_and_baseline(self):
        sim = SensorSimulator(scenario="happy_chat", seed=7)
        for pkt in sim.run_all():
            self.system.tick(pkt)
        profile = self.system.get_profile()
        self.assertGreater(profile["stats"]["emotion_records"], 5)
        self.assertTrue(profile["baseline"].get("hr_bpm"))


if __name__ == "__main__":
    unittest.main()
