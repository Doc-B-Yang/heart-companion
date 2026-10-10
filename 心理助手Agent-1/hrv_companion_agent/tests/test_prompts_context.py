# -*- coding: utf-8 -*-
"""Prompt 注入层契约测试（agent/prompts.py）

这一层是"生理数据 → 模型行为"的唯一通道，也是本项目出过最多**静默缺陷**
的地方（不报错的错最难查）。历史上真实出现过：

  · 「对话风格要求」「工具使用纪律」两段重复注入；
  · ``describe_affect`` 未导入模块 → 一调用即 NameError；
  · ``_QUALITY_LABELS`` 用 good/fair/poor 作键，而质量模块产出的是
    合格/勉强/不可用 → 查表恒为空串，"信号质量"这一行**从未被注入过**；
  · 质量标签里重复渲染"（生理判断可用）（生理判断可用）"。

共同点是：程序照跑、日志干净、单元测试全绿，只有把实际 Prompt 打出来
才看得见。所以本文件一律**对着渲染结果断言**，而不是对内部变量断言。
"""

import unittest

from hrv_companion_agent import config
from hrv_companion_agent.agent import prompts


def _state(**over) -> dict:
    st = {
        "emotion": "高焦虑", "confidence": 0.78, "risk_score": 82,
        "layer": "rule", "timestamp": 1700000000.0,
        "hr_bpm": 95, "sdnn_ms": 31.0, "rmssd_ms": 22.0, "lf_hf": 3.2,
        "pnn50": 4.0, "sd_hr_bpm": 5.0, "jitter": 0.05,
        "quality": {"quality_ok": True, "quality_level": "合格", "jitter": 0.05},
        "quality_level": "合格",
        "pad": {"P": -2.1, "A": 1.8, "D": -0.9, "confidence": 0.55,
                "d_confidence": 0.19, "method": "rule", "emotion_hint": "高焦虑"},
        "affect": {"level": "moderate", "mood": -0.32, "dominant": "fear",
                   "dominant_zh": "恐惧", "emotion_zh": "高焦虑",
                   "dominant_intensity": 0.51, "durations": {"fear": 26.0}},
        "rel_features": {"rmssd_ms": 0.51, "sdnn_ms": 0.38},
    }
    st.update(over)
    return st


class TestNoDuplicateSections(unittest.TestCase):
    """系统提示词里每个小节只能出现一次（回归：曾经整段重复）"""

    def test_section_headers_are_unique(self):
        text = prompts.COMPANION_SYSTEM_PROMPT
        for header in ("对话风格要求", "工具使用纪律", "安全与伦理",
                       "工具说明", "输出要求"):
            if header in text:
                self.assertEqual(text.count(header), 1,
                                 f"小节「{header}」出现了 {text.count(header)} 次")

    def test_prompt_is_not_empty(self):
        self.assertGreater(len(prompts.COMPANION_SYSTEM_PROMPT), 500)

    def test_prompt_names_the_five_emotions(self):
        """提示词里定义的标签集必须与识别器产出的 5 类一致

        否则模型会收到一个"提示词没定义过"的情绪名 —— 例如把情绪轮的
        「恐惧」当标签注入（`dominant_zh` 就是这个坑，故它禁止用于 Prompt）。
        """
        from hrv_companion_agent.algorithms.emotion_recognizer import EMOTION_BASE_RISK
        text = prompts.COMPANION_SYSTEM_PROMPT
        for emo in EMOTION_BASE_RISK:
            self.assertIn(emo, text, f"提示词未定义情绪「{emo}」")


class TestEmotionContextDefaults(unittest.TestCase):
    """默认**不注入原始生理数值**（v2 契约，与 v1 相反）"""

    def setUp(self):
        self._old = config.EXPOSE_RAW_METRICS
        config.EXPOSE_RAW_METRICS = False

    def tearDown(self):
        config.EXPOSE_RAW_METRICS = self._old

    def test_no_raw_numbers_by_default(self):
        """不得出现 bpm / 具体毫秒值 / 特征名

        依据两条实测经验：(a) 模型拿到数字后会在回复里复述，
        破坏"不直接说指标"的设定；(b) 数字对生成质量的边际贡献很低，
        而语义标签模型处理得更好。
        """
        text = prompts.build_emotion_context("今天真的好累", _state())
        self.assertNotIn("95", text)
        self.assertNotIn("bpm", text)
        self.assertNotIn("rmssd", text.lower())
        self.assertNotIn("sdnn", text.lower())
        self.assertNotIn("[调试用原始值]", text)

    def test_debug_mode_shows_raw(self):
        config.EXPOSE_RAW_METRICS = True
        text = prompts.build_emotion_context("今天真的好累", _state())
        self.assertIn("[调试用原始值]", text)
        self.assertIn("95", text)

    def test_semantic_band_present(self):
        """必须有语义化的紧张程度，且分档正确"""
        text = prompts.build_emotion_context("hi", _state(risk_score=82))
        self.assertIn("紧张程度", text)
        self.assertIn("很高", text)

    def test_low_risk_band(self):
        text = prompts.build_emotion_context("hi", _state(risk_score=10))
        self.assertIn("低", text)
        self.assertNotIn("很高", text)

    def test_message_is_preserved(self):
        msg = "我今天面试被拒了，觉得自己特别差劲"
        text = prompts.build_emotion_context(msg, _state())
        self.assertIn(msg, text)

    def test_affect_and_pad_included(self):
        text = prompts.build_emotion_context("hi", _state())
        self.assertIn("情感动力学", text)
        self.assertIn("维度倾向", text)


class TestQualityLineIsActuallyInjected(unittest.TestCase):
    """回归：质量提示曾因标签键不匹配而**从未被注入过**"""

    def setUp(self):
        self._old = config.EXPOSE_RAW_METRICS
        config.EXPOSE_RAW_METRICS = False

    def tearDown(self):
        config.EXPOSE_RAW_METRICS = self._old

    def test_good_quality_line_present(self):
        text = prompts.build_emotion_context(
            "hi", _state(quality={"quality_ok": True,
                                  "quality_level": "合格", "jitter": 0.05}))
        self.assertIn("信号质量", text)
        self.assertIn("良好", text)

    def test_fair_quality_line_present(self):
        """「勉强」也是可用档，必须告诉模型"可参考但以用户自述为准" """
        text = prompts.build_emotion_context(
            "hi", _state(quality={"quality_ok": True,
                                  "quality_level": "勉强", "jitter": 0.10}))
        self.assertIn("信号质量", text)
        self.assertIn("一般", text)

    def test_label_keys_match_producer_output(self):
        """**契约测试**：质量标签的键必须覆盖质量模块的**全部**产出值

        这是上面那条"从未被注入"缺陷的根因：键用了 good/fair/poor，
        而产出是合格/勉强/不可用/不足 → 查表恒空。此断言直接比对两侧取值。
        """
        from hrv_companion_agent.algorithms.hrv_extractor import HRVExtractor
        ex = HRVExtractor()

        def _rr(a, n=60, mean=850.0):
            return [mean + (a if i % 2 == 0 else -a) for i in range(n)]

        produced = set()
        for rr in (_rr(5.0), _rr(40.0), _rr(65.0), [850.0, 860.0]):
            produced.add(ex.quality_metrics(rr)["quality_level"])
        self.assertTrue(produced)
        missing = produced - set(prompts._QUALITY_LABELS)
        self.assertFalse(missing, f"质量标签缺少这些等级: {sorted(missing)}")

    def test_poor_quality_triggers_warning_block(self):
        """质量不可用 → 必须出现"以用户自述为准"的强制提醒"""
        text = prompts.build_emotion_context(
            "hi", _state(quality={"quality_ok": False,
                                  "quality_level": "不可用", "jitter": 0.45}))
        self.assertIn("信号质量质量不佳" if False else "信号质量不佳", text)
        self.assertIn("以用户自述为准", text)
        self.assertNotIn("信号质量: 良好", text)

    def test_no_double_parenthesis(self):
        """回归：标签里已含结论，不得再追加修饰语

        曾经渲染成「良好（生理判断可用）（生理判断可用）」——
        不影响运行，但一眼就能看出是拼接失手。
        """
        text = prompts.build_emotion_context(
            "hi", _state(quality={"quality_ok": True,
                                  "quality_level": "合格", "jitter": 0.05}))
        line = next(ln for ln in text.splitlines() if ln.startswith("信号质量"))
        self.assertEqual(line.count("生理判断可用"), 1)

    def test_no_quality_block_yields_no_line(self):
        text = prompts.build_emotion_context("hi", _state(quality={}))
        self.assertNotIn("信号质量", text)


class TestLowConfidenceHint(unittest.TestCase):
    """置信度偏低时必须告诉模型"这次判断没底" """

    def setUp(self):
        self._old = config.EXPOSE_RAW_METRICS
        config.EXPOSE_RAW_METRICS = False

    def tearDown(self):
        config.EXPOSE_RAW_METRICS = self._old

    def test_low_confidence_hint(self):
        text = prompts.build_emotion_context("hi", _state(confidence=0.31))
        self.assertIn("把握不大", text)

    def test_high_confidence_no_hint(self):
        text = prompts.build_emotion_context("hi", _state(confidence=0.88))
        self.assertNotIn("把握不大", text)


class TestEmptyState(unittest.TestCase):
    """无可用读数时应退化为纯消息（不得注入 N/A 块或假结论）"""

    def test_no_state_returns_plain_message(self):
        self.assertEqual(prompts.build_emotion_context("你好", None), "你好")
        self.assertEqual(prompts.build_emotion_context("你好", {}), "你好")

    def test_accumulating_state_is_plain(self):
        """「数据积累中」不得注入情绪块

        ⚠️ 这条测的不是"少输出点东西"，而是**防止把"不知道"说成"很放松"**：
        没有足够心跳样本时紧张程度会被算成 0 → 渲染成「低（很放松）」，
        模型可能因此回一句"你看起来挺放松的呀"。宁可什么都不注入。
        """
        st = {"emotion": "数据积累中", "confidence": 0.0, "risk_score": 0.0,
              "layer": "insufficient_data"}
        self.assertEqual(prompts.build_emotion_context("你好", st), "你好")

    def test_unrecognised_emotion_is_plain(self):
        for emo in ("未识别", "N/A", "unknown", ""):
            st = {"emotion": emo, "risk_score": 0.0, "layer": "rule"}
            self.assertEqual(prompts.build_emotion_context("你好", st), "你好",
                             f"情绪为 {emo!r} 时不应注入情绪块")

    def test_insufficient_layer_is_plain(self):
        """即使标签看着正常，识别层为 insufficient_data 也不得注入"""
        st = {"emotion": "平静", "risk_score": 0.0,
              "layer": "insufficient_data"}
        self.assertEqual(prompts.build_emotion_context("你好", st), "你好")

    def test_manual_layer_is_usable(self):
        """手动注入的情绪必须可用（无硬件联调的主要路径）"""
        st = {"emotion": "高焦虑", "confidence": 0.9, "risk_score": 85,
              "layer": "manual"}
        self.assertIn("[系统感知]", prompts.build_emotion_context("hi", st))

    def test_usable_state_judgement_is_shared(self):
        """判据只应有**一处定义**：Agent 与注入函数必须同源

        两份逻辑漂移时会出现"外面以为不注入、里面照样注入"的错位。
        """
        from hrv_companion_agent.agent.companion_agent import CompanionAgent
        cases = [
            None,
            {},
            {"emotion": "数据积累中", "layer": "insufficient_data"},
            {"emotion": "未识别", "layer": "rule"},
            {"emotion": "高焦虑", "layer": "rule"},
            {"emotion": "平静", "layer": "manual"},
        ]
        for st in cases:
            self.assertEqual(CompanionAgent._has_meaningful_state(st),
                             prompts.is_usable_state(st), f"判据不一致: {st}")


class TestTransitionHint(unittest.TestCase):
    """情绪转场提示（v2）"""

    def test_shift_produces_hint(self):
        self.assertTrue(prompts.build_transition_hint("平静", "高焦虑"))

    def test_same_emotion_no_hint(self):
        self.assertFalse(prompts.build_transition_hint("平静", "平静"))

    def test_empty_previous_no_hint(self):
        self.assertFalse(prompts.build_transition_hint("", "高焦虑"))


class _Decision:
    """最小 TriggerDecision 替身（只需 proactive 模板读的三个字段）"""

    def __init__(self, level="low", strategy="light_greeting",
                 message="开场白"):
        self.trigger_level = level
        self.strategy = strategy
        self.message = message


class TestProactiveMessage(unittest.TestCase):
    """主动关怀的消息构造

    ⚠️ 签名是 ``build_proactive_message(state, trigger_decision)`` ——
    第二个参数是 **TriggerDecision 对象**而不是字符串。
    开场白取自 ``decision.message``，策略与风格按 ``decision.strategy`` 查表。
    传成字符串不会报错，只会渲染出空的"开场白参考: "（以及空的"建议策略（）"），
    是最容易漏掉的一类拼接错误。
    """

    def setUp(self):
        self._old = config.EXPOSE_RAW_METRICS
        config.EXPOSE_RAW_METRICS = False

    def tearDown(self):
        config.EXPOSE_RAW_METRICS = self._old

    def test_contains_opener_from_decision(self):
        text = prompts.build_proactive_message(
            _state(risk_score=45),
            _Decision(message="感觉你今天有点累呢，还好吗？"))
        self.assertIn("感觉你今天有点累呢", text)

    def test_strategy_and_style_are_rendered(self):
        text = prompts.build_proactive_message(
            _state(risk_score=65),
            _Decision(level="high", strategy="breathing_guide"))
        self.assertIn("breathing_guide", text)
        self.assertNotIn("（）", text)      # 风格/工具查不到时会渲染成空括号

    def test_no_raw_numbers_by_default(self):
        text = prompts.build_proactive_message(_state(), _Decision())
        self.assertNotIn("95", text)
        self.assertIn("紧张程度", text)

    def test_emergency_includes_hotline(self):
        text = prompts.build_proactive_message(
            _state(risk_score=99),
            _Decision(level="emergency", strategy="crisis_intervention"))
        self.assertIn(config.CRISIS_HOTLINES[0][1], text)

    def test_bad_quality_blocks_body_assertions(self):
        st = _state(quality={"quality_ok": False, "quality_level": "不可用",
                             "jitter": 0.5})
        text = prompts.build_proactive_message(st, _Decision())
        self.assertIn("信号质量提示", text)

    def test_returns_str(self):
        self.assertIsInstance(
            prompts.build_proactive_message(_state(), _Decision()), str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
