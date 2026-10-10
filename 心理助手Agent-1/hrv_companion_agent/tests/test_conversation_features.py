# -*- coding: utf-8 -*-
"""离线话术脑 + 手动情绪注入 + 记忆工具 的单元测试"""
import unittest

from hrv_companion_agent.agent.memory_manager import MemoryManager
from hrv_companion_agent.agent.prompts import OfflineBrain
from hrv_companion_agent.agent.tools.memory_tool import MemoryTool
from hrv_companion_agent.core.system import HeartCompanionSystem


class TestOfflineBrain(unittest.TestCase):
    def setUp(self):
        self.brain = OfflineBrain()

    def test_identity(self):
        reply = self.brain.respond("你是谁")
        self.assertIn("小暖", reply)

    def test_greeting(self):
        reply = self.brain.respond("你好")
        self.assertIn("我在", reply)

    def test_farewell(self):
        reply = self.brain.respond("晚安")
        self.assertIn("休息", reply)

    def test_exam_failure(self):
        reply = self.brain.respond("我考砸了怎么办")
        self.assertIn("考试", reply)

    def test_context_followup(self):
        """两轮上下文: 先难过、再问考砸 → 回复应结合两轮"""
        self.brain.respond("我很难过")
        reply = self.brain.respond("我考砸了怎么办")
        self.assertIn("难过", reply)

    def test_no_repeat_fallback(self):
        """连续两次无关键词输入 → 回复不应机械重复"""
        r1 = self.brain.respond("emmm")
        r2 = self.brain.respond("然后呢")
        self.assertNotEqual(r1, r2)

    def test_crisis_priority(self):
        reply = self.brain.respond("我最近不想活了")
        self.assertIn("400-161-9995", reply)

    def test_ascii_keyword_word_boundary(self):
        """英文关键词词边界匹配: "high" 不应命中问候规则 "hi"（回归）"""
        reply = self.brain.respond("系统提示：触发等级为 high，请引导呼吸")
        self.assertNotIn("我在呢！今天过得怎么样", reply)

    def test_emo_word_boundary(self):
        """"emotion" 不应命中 "emo" 规则（词边界）"""
        reply = self.brain.respond("emotion 这个词怎么理解")
        # 不返回抑郁/emo 规则话术（含"谢谢你愿意把这些说出来"）
        self.assertNotIn("谢谢你愿意把这些说出来", reply)

    def test_proactive_offline_path(self):
        """[系统主动触发] 离线回复应走 proactive 话术而非关键词猜测（回归）"""
        from hrv_companion_agent.core.llm_client import CompanionLLM
        llm = CompanionLLM()
        llm._backend = "offline"
        msg = ("[系统主动触发 - 检测到情绪异常]\n"
               "当前心率: 96 bpm\n情绪状态: 高焦虑\n风险评分: 85/100\n"
               "触发等级: low\n开场白参考: 感觉你今天有点累呢，还好吗？\n"
               "请你主动发起一段温暖的关怀对话。")
        reply = llm.invoke([{"role": "user", "content": msg}])
        # 应包含安抚/呼吸引导（proactive 话术），而非"我在呢！今天过得怎么样"问候
        self.assertIn("深呼吸", reply)

    def test_proactive_low_risk_greeting(self):
        """低风险主动触发 → 使用策略开场白 + 陪伴收尾"""
        reply = self.brain.proactive({"risk_score": 30, "emotion": "平静"},
                                     "好久没和你聊天啦，今天过得怎么样？")
        self.assertIn("好久没和你聊天啦", reply)
        self.assertIn("🌷", reply)


class TestMemoryTool(unittest.TestCase):
    def test_record_fact(self):
        mm = MemoryManager("tool_user", db_path=":memory:")
        tool = MemoryTool(mm, "tool_user")
        out = tool.run({"fact": "用户是计算机专业大三学生"})
        self.assertIn("已记住", out)
        profile = mm.get_user_profile()
        self.assertTrue(any("计算机" in str(v) for v in profile["facts"].values()))

    def test_dedup(self):
        mm = MemoryManager("tool_user2", db_path=":memory:")
        tool = MemoryTool(mm, "tool_user2")
        tool.run({"fact": "用户喜欢在晚上复习"})
        out = tool.run({"fact": "用户喜欢在晚上复习"})
        self.assertIn("相似", out)


class TestManualEmotion(unittest.TestCase):
    def setUp(self):
        import hrv_companion_agent.config as config
        self._saved_key = config.LLM_API_KEY
        self._saved_tts = config.TTS_ENABLED
        config.LLM_API_KEY = ""
        config.TTS_ENABLED = False
        self.system = HeartCompanionSystem(user_id="manual_user")

    def tearDown(self):
        import hrv_companion_agent.config as config
        config.LLM_API_KEY = self._saved_key
        config.TTS_ENABLED = self._saved_tts
        self.system.close()

    def test_set_manual_emotion(self):
        result = self.system.set_manual_emotion("高焦虑", 85)
        self.assertTrue(result["ok"])
        state = self.system.get_state()
        self.assertEqual(state["emotion"], "高焦虑")
        self.assertEqual(state["risk_score"], 85)
        self.assertEqual(state["layer"], "manual")

    def test_invalid_emotion(self):
        result = self.system.set_manual_emotion("暴怒")
        self.assertFalse(result["ok"])

    def test_manual_emotion_affects_chat(self):
        """注入高焦虑后聊天 → emotion_adapted 应为 True"""
        self.system.set_manual_emotion("高焦虑", 88)
        reply = self.system.chat("我有点慌")
        self.assertTrue(reply["emotion_adapted"])
        self.assertTrue(reply["response"])


if __name__ == "__main__":
    unittest.main()
