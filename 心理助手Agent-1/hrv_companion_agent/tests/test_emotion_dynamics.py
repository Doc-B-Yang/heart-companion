# -*- coding: utf-8 -*-
"""情感动力学状态机单元测试（对应文献3 式(11)–(20) 与计划书 §4.2）

本文件的重心不是"公式算得对不对"，而是三件更容易出错、且一旦出错会
直接影响产品行为的事：

  ① **符号**。式(11) 的心境调制项必须用**带符号**的 mood。若误写成
     |mood|，会得出"消极心境抑制消极情绪"的反向结论 —— 数值上毫无异常，
     行为上完全颠倒，只靠读代码很难发现。
  ② **式(13) 的失稳**。原文每步都乘延长因子，刺激持续存在时 di 会指数
     增长 → 衰减停滞、情绪被"粘住"。本实现加了上限，这里守住它。
  ③ **人格参数的真实接线范围**。四档人格里哪些参数真的生效，
     必须与文档一致，否则实验设计会建在错误的假设上。
"""

import unittest

from hrv_companion_agent.algorithms.emotion_dynamics import (
    DURATION_MAX, DURATION_OTHER, DURATION_POSITIVE, EMOTION_POLARITY,
    EXTEND_FACTOR_OTHER, EXTEND_FACTOR_POSITIVE, LEVEL_THRESHOLDS, MOOD_DECAY,
    NEGATIVE_MOOD_GATE, PERSONALITY_PROFILES, RESERVED_PROFILE_PARAMS,
    WHEEL_EMOTIONS, WIRED_PROFILE_PARAMS, ZH_TO_WHEEL, EmotionDynamics,
    classify_level)


class TestWheelMapping(unittest.TestCase):
    """本项目 5 类中文情绪 ⇄ Plutchik 情绪轮的映射"""

    def test_every_zh_emotion_maps_into_wheel(self):
        """映射结果不得出现情绪轮以外的键（否则动力学里查不到、静默丢弃）"""
        for zh, weights in ZH_TO_WHEEL.items():
            for e in weights:
                self.assertIn(e, WHEEL_EMOTIONS, f"{zh} → {e} 不在情绪轮中")

    def test_nonempty_rows_sum_to_one(self):
        """非空行的权重和为 1（平静为空行，表示不激发任何情绪）"""
        for zh, weights in ZH_TO_WHEEL.items():
            if weights:
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=6,
                                       msg=f"{zh} 权重和不为 1")

    def test_calm_yields_no_stimulus(self):
        """「平静」映射为空 —— 平静不是一种被激起的情感，而是没有情感"""
        self.assertEqual(ZH_TO_WHEEL["平静"], {})

    def test_polarity_table_is_complete(self):
        for e in WHEEL_EMOTIONS:
            self.assertIn(e, EMOTION_POLARITY)

    def test_high_anxiety_is_negative_polarity(self):
        """高焦虑分解出的成分必须全是负效价（否则心境会朝错方向累积）"""
        for e in ZH_TO_WHEEL["高焦虑"]:
            self.assertEqual(EMOTION_POLARITY[e], -1)

    def test_pleasant_is_positive_polarity(self):
        for e in ZH_TO_WHEEL["愉悦"]:
            self.assertEqual(EMOTION_POLARITY[e], 1)


class TestLevelClassification(unittest.TestCase):
    """式(16) 强度分级"""

    def test_boundaries(self):
        h1, h2, h3 = LEVEL_THRESHOLDS
        self.assertEqual(classify_level(0.0), "unaware")
        self.assertEqual(classify_level(h1), "unaware")      # 含下界
        self.assertEqual(classify_level(h1 + 1e-6), "mild")
        self.assertEqual(classify_level(h2), "mild")
        self.assertEqual(classify_level(h2 + 1e-6), "moderate")
        self.assertEqual(classify_level(h3), "moderate")
        self.assertEqual(classify_level(h3 + 1e-6), "intense")

    def test_thresholds_are_ordered(self):
        h1, h2, h3 = LEVEL_THRESHOLDS
        self.assertLess(h1, h2)
        self.assertLess(h2, h3)


class TestStimulusResponse(unittest.TestCase):
    """持续刺激 → 强度上升；撤去刺激 → 衰减"""

    def test_emotion_forms_under_sustained_stimulus(self):
        dyn = EmotionDynamics()
        for _ in range(5):
            st = dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        # dominant_zh 是**情绪轮**的中文名（高焦虑在轮上分解为 fear 主导）
        self.assertEqual(st.dominant, "fear")
        # 折算回项目 5 类才是"高焦虑"—— 这两者不可混用
        self.assertEqual(dyn.effective_emotion_zh(), "高焦虑")
        self.assertIn(st.to_dict()["emotion_zh"], ("高焦虑",))
        self.assertGreater(st.dominant_intensity, LEVEL_THRESHOLDS[0])
        self.assertIn(st.level, ("mild", "moderate", "intense"))

    def test_intensity_decays_after_stimulus_removed(self):
        dyn = EmotionDynamics()
        for _ in range(5):
            dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        peak = dyn.intensities["fear"]
        for _ in range(10):
            dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertLess(dyn.intensities["fear"], peak)

    def test_emotion_extinguishes_below_hi1(self):
        """强度跌破 hi1 应被清零，而不是留一条永不消失的尾巴

        注意需把 mood 也拉回中性：mood 为负时式(11) 会持续给负性情绪
        供一个正调制项，情绪被"养着"不灭 —— 那是**有意**的行为
        （坏心情让烦躁更黏），见 test_negative_mood_prolongs_emotion。
        这里只验证"当心境也归中性后，确实存在熄灭通路"。
        """
        dyn = EmotionDynamics()
        dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        for _ in range(60):
            dyn.mood = 0.0                     # 只测衰减通路，排除心境再供能
            dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertEqual(dyn.intensities["fear"], 0.0)
        self.assertEqual(dyn.durations["fear"], 0.0)

    def test_extinct_tolerance_is_required(self):
        """hi1 是式(17) 的不动点 → 没有容差就永远熄不灭（EXTINCT_TOL）

        这条测试把 EXTINCT_TOL 的存在理由钉死：去掉它，
        强度会无限逼近 hi1 而永不 <= hi1，durations 永不归零，
        下游"这份难受已经跟着你一阵子了"会挂在早已消退的情绪上。
        """
        from hrv_companion_agent.algorithms import emotion_dynamics as ed
        self.assertGreater(ed.EXTINCT_TOL, 0.0)

        dyn = EmotionDynamics()
        dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        for _ in range(60):
            dyn.mood = 0.0
            dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        # 若熄灭逻辑失效，intensities 会稳定在 hi1 附近而不是 0
        self.assertEqual(dyn.intensities["fear"], 0.0)
        self.assertNotAlmostEqual(dyn.intensities["fear"],
                                  LEVEL_THRESHOLDS[0], places=3)

    def test_dominant_ignored_below_hi1(self):
        """**状态层不得上报"无情绪级别的情绪"**

        强度落在 unaware 区间（<= hi1）时，系统必须报"平静/无主导情绪"。
        用 >0 作判据的话，一个消退十几分钟的情绪仍会以
        「当前主导情绪 = 恐惧、强度 0.15、级别 unaware」的形式长期挂在
        Prompt 与前端上 —— 数值没错，但表述是错的。
        """
        dyn = EmotionDynamics()
        dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        # 衰减到刚好低于 hi1 但不为 0 的区间
        for _ in range(25):
            dyn.mood = 0.0
            dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        st = dyn.snapshot()
        if st.level == "unaware":
            self.assertEqual(st.dominant, "")
            self.assertEqual(st.dominant_zh, "平静")
            self.assertEqual(st.dominant_intensity, 0.0)

    def test_negative_mood_prolongs_emotion(self):
        """消极心境显著延长负性情绪的存活时间（式11 调制的直接后果）

        这既是"坏心情让烦躁更黏"的建模，也是产品上一条可用的线索：
        当系统发现情绪迟迟不消退而外部事件已过去，可以优先考虑
        "心境底色偏低"而非"又发生了什么"。
        """
        def steps_to_extinct(neutral_mood: bool) -> int:
            dyn = EmotionDynamics()
            dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
            for i in range(400):
                if neutral_mood:
                    dyn.mood = 0.0
                dyn.update_from_zh({"平静": 1.0}, dt=1.0)
                if dyn.intensities["fear"] == 0.0:
                    return i + 1
            return 10 ** 6

        with_mood = steps_to_extinct(neutral_mood=False)
        without = steps_to_extinct(neutral_mood=True)
        self.assertLess(without, 80)
        self.assertGreater(with_mood, without)

    def test_stronger_stimulus_gives_higher_intensity(self):
        """刺激强度单调性：这是"情绪强度可被调制"的前提"""
        weak, strong = EmotionDynamics(), EmotionDynamics()
        for _ in range(3):
            weak.update_from_zh({"低焦虑": 0.3}, dt=1.0)
            strong.update_from_zh({"低焦虑": 0.9}, dt=1.0)
        self.assertGreater(strong.intensities["fear"], weak.intensities["fear"])

    def test_dt_scaling(self):
        """同样两秒：一次 dt=2 的衰减应比两次 dt=1 更接近（同阶）"""
        a, b = EmotionDynamics(), EmotionDynamics()
        for d in (a, b):
            d.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        a.update_from_zh({"平静": 1.0}, dt=2.0)
        b.update_from_zh({"平静": 1.0}, dt=1.0)
        b.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertLess(a.intensities["fear"], b.intensities["fear"])

    def test_dt_guard_against_clock_jump(self):
        """时钟跳变（如离线补数据）不得让状态爆炸或产生 NaN"""
        dyn = EmotionDynamics()
        dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        st = dyn.update_from_zh({"高焦虑": 0.8}, dt=1e9)
        for v in st.intensities.values():
            self.assertFalse(v != v)          # NaN 检查
            self.assertLessEqual(v, 1.0 + 1e-9)

    def test_zero_dt_does_not_crash(self):
        dyn = EmotionDynamics()
        st = dyn.update_from_zh({"高焦虑": 0.8}, dt=0.0)
        self.assertIsNotNone(st)


class TestDurationFormula(unittest.TestCase):
    """式(12)(13)：积极情感持续更久、延长因子更大"""

    def test_paper_values(self):
        self.assertEqual(DURATION_POSITIVE, 8.0)
        self.assertEqual(DURATION_OTHER, 5.0)
        self.assertEqual(EXTEND_FACTOR_POSITIVE, 1.5)
        self.assertEqual(EXTEND_FACTOR_OTHER, 1.1)

    def test_new_emotion_gets_paper_duration(self):
        """式(12)：首次被激起 → 积极 8s、其他 5s"""
        pos = EmotionDynamics()
        pos.update_from_zh({"愉悦": 0.9}, dt=1.0)
        self.assertAlmostEqual(pos.durations["joy"], DURATION_POSITIVE, places=6)

        oth = EmotionDynamics()
        oth.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        self.assertAlmostEqual(oth.durations["fear"], DURATION_OTHER, places=6)

    def test_extend_factor_differs_by_polarity(self):
        """式(13)：弱刺激不升高强度，但按极性用不同因子延长持续时间"""
        pos = EmotionDynamics()
        pos.update_from_zh({"愉悦": 0.9}, dt=1.0)
        pos.update_from_zh({"愉悦": 0.3}, dt=1.0)
        self.assertAlmostEqual(pos.durations["joy"],
                               DURATION_POSITIVE * EXTEND_FACTOR_POSITIVE,
                               places=6)

        oth = EmotionDynamics()
        oth.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        oth.update_from_zh({"高焦虑": 0.3}, dt=1.0)
        self.assertAlmostEqual(oth.durations["fear"],
                               DURATION_OTHER * EXTEND_FACTOR_OTHER, places=6)

    def test_duration_is_capped(self):
        """**工程加固**：式(13) 每次乘因子，无上限时 di 会指数增长

        文献原式在刺激持续存在（例如长时间焦虑）时会让 di 无界增长，
        衰减随之趋近停滞 —— 情绪被"粘住"，再也回不到中性。
        这里直接验证 min(·, DURATION_MAX) 这层保护确实生效。
        """
        dyn = EmotionDynamics()
        dyn.intensities["joy"] = 0.5
        dyn.durations["joy"] = 55.0          # 55 × 1.5 = 82.5 > 60
        dyn.update({"joy": 0.2}, dt=1.0)
        self.assertEqual(dyn.durations["joy"], DURATION_MAX)

        # 再推进一步，不得继续增长
        dyn.update({"joy": 0.2}, dt=1.0)
        self.assertEqual(dyn.durations["joy"], DURATION_MAX)

    def test_persistent_stimulus_still_decays_eventually(self):
        """长刺激之下强度仍必须能消退（上限保护的最终目的）

        这条是上一条的"后果验证"：光有上限还不够，要确认在持续弱刺激下
        情绪最终仍能回到 unaware —— 否则上限只是拖慢了"粘住"。
        """
        dyn = EmotionDynamics()
        dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        for _ in range(120):
            dyn.mood = 0.0                     # 排除心境再供能，只看衰减
            dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertEqual(dyn.snapshot().level, "unaware")
        self.assertEqual(dyn.snapshot().dominant, "")


class TestMoodDynamics(unittest.TestCase):
    """式(18)(19)(20)：心境累积、符号、回中性"""

    def test_negative_emotion_lowers_mood(self):
        dyn = EmotionDynamics()
        for _ in range(10):
            dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        self.assertLess(dyn.mood, 0)
        self.assertEqual(dyn.snapshot().polarity, -1)

    def test_positive_emotion_raises_mood(self):
        dyn = EmotionDynamics()
        for _ in range(10):
            dyn.update_from_zh({"愉悦": 0.8}, dt=1.0)
        self.assertGreater(dyn.mood, 0)
        self.assertEqual(dyn.snapshot().polarity, 1)

    def test_mood_bounded(self):
        """心境必须始终落在 [−1, 1]（式20 的钳位）"""
        dyn = EmotionDynamics()
        for _ in range(400):
            dyn.update_from_zh({"高焦虑": 1.0}, dt=1.0)
            self.assertLessEqual(abs(dyn.mood), 1.0)
        for _ in range(400):
            dyn.update_from_zh({"愉悦": 1.0}, dt=1.0)
            self.assertLessEqual(abs(dyn.mood), 1.0)

    def test_negative_mood_decays_faster_toward_neutral(self):
        """式(20) 的**安全性设计**：消极心境额外 +0.05 加速回中性

        同幅度的正负心境，各推进 1 秒，负向应消退得更多。
        这条设计的目的很实际：不让用户被系统"记住"的坏心情拖着出不来。
        """
        neg, pos = EmotionDynamics(), EmotionDynamics()
        neg.mood = -0.5
        pos.mood = 0.5
        neg.update({}, dt=1.0)
        pos.update({}, dt=1.0)
        neg_moved = 0.5 - abs(neg.mood)
        pos_moved = 0.5 - abs(pos.mood)
        self.assertGreater(neg_moved, pos_moved)

    def test_bias_not_applied_above_gate(self):
        """偏置只在 mood ≤ NEGATIVE_MOOD_GATE 时施加（轻微微负不加速）"""
        dyn = EmotionDynamics()
        dyn.mood = NEGATIVE_MOOD_GATE / 2      # 高于门限
        dyn.update({}, dt=1.0)
        expected = (NEGATIVE_MOOD_GATE / 2) * 1.0  # 仅自然衰减，无偏置
        import math
        expected *= math.exp(-MOOD_DECAY * 1.0)
        self.assertAlmostEqual(dyn.mood, expected, places=6)

    def test_neutral_stimulus_leaves_mood_alone(self):
        """平静不贡献任何效价，心境只按自然衰减走"""
        dyn = EmotionDynamics()
        dyn.mood = 0.4
        dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        import math
        self.assertAlmostEqual(dyn.mood, 0.4 * math.exp(-MOOD_DECAY),
                               places=6)

    def test_mood_strength_is_absolute_value(self):
        dyn = EmotionDynamics()
        for _ in range(8):
            dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        self.assertAlmostEqual(dyn.snapshot().mood_strength,
                               abs(dyn.mood), places=6)


class TestMoodModulationSign(unittest.TestCase):
    """式(11) 调制项的**符号** —— 本文件最该守住的一组断言"""

    @staticmethod
    def _run(mood: float, rounds: int = 4) -> float:
        dyn = EmotionDynamics()
        dyn.mood = mood
        for _ in range(rounds):
            dyn.update_from_zh({"高焦虑": 0.5}, dt=1.0)
        return dyn.intensities["fear"]

    def test_negative_mood_amplifies_negative_emotion(self):
        """消极心境应放大负性情绪（fear），而不是抑制它

        若式(11) 误用 |mood|，这条会失败：fear 的 EMOTION_POLARITY 是 −1，
        取模后 mood_term 变号，结论会整个倒过来。
        """
        self.assertGreater(self._run(-0.5), self._run(0.0))

    def test_positive_mood_suppresses_negative_emotion(self):
        self.assertLess(self._run(0.5), self._run(0.0))

    def test_positive_emotion_modulated_oppositely(self):
        """正性情绪的方向必须相反：积极心境放大、消极心境抑制"""
        def run(mood):
            dyn = EmotionDynamics()
            dyn.mood = mood
            for _ in range(4):
                dyn.update_from_zh({"愉悦": 0.5}, dt=1.0)
            return dyn.intensities["joy"]

        self.assertGreater(run(0.5), run(0.0))
        self.assertLess(run(-0.5), run(0.0))

    def test_mood_term_can_activate_latent_emotion(self):
        """零刺激下，消极心境仍能唤起潜伏的负性情绪

        这不是 bug 而是式(11) 的直接推论：心境是情绪的"底色"。
        实测意义在于 —— 一个心境已经很低的用户，即使当下没发生什么，
        系统也可能检测到情绪活动；此时回应应当偏向"陪伴"而非"追问事件"。
        """
        dyn = EmotionDynamics()
        dyn.mood = -0.6
        dyn.update({}, dt=1.0)
        self.assertGreater(dyn.intensities["fear"], 0.0)


class TestPersonalityProfiles(unittest.TestCase):
    """文献3 表2 四档人格 —— 接线范围必须与文档一致"""

    def test_four_profiles_exist(self):
        self.assertEqual(set(PERSONALITY_PROFILES),
                         {"ideal", "indifferent", "sensitive", "persistent"})

    def test_all_profiles_have_chinese_names(self):
        for key, p in PERSONALITY_PROFILES.items():
            self.assertTrue(p.get("zh"), f"{key} 缺少中文名")

    def test_unknown_profile_falls_back_to_ideal(self):
        dyn = EmotionDynamics(personality="不存在的档位")
        self.assertEqual(dyn.personality_key, "ideal")

    def test_sensitive_amplifies_response(self):
        """敏感型 interaction=2.0 → 同一刺激下强度更高"""
        sens, ideal = EmotionDynamics("sensitive"), EmotionDynamics("ideal")
        for _ in range(4):
            sens.update_from_zh({"高焦虑": 0.3}, dt=1.0)
            ideal.update_from_zh({"高焦虑": 0.3}, dt=1.0)
        self.assertGreater(sens.intensities["fear"], ideal.intensities["fear"])

    def test_persistent_decays_slower_than_indifferent(self):
        """执着型 α=0.02 < 冷漠型 α=0.15 → 同样时间后保留更多强度"""
        per, ind = EmotionDynamics("persistent"), EmotionDynamics("indifferent")
        for _ in range(6):
            per.update_from_zh({"高焦虑": 0.8}, dt=1.0)
            ind.update_from_zh({"高焦虑": 0.8}, dt=1.0)
        per.update_from_zh({"平静": 1.0}, dt=1.0)
        ind.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertGreater(per.intensities["fear"], ind.intensities["fear"])

    def test_wired_and_reserved_partition_the_fields(self):
        """已接线的 + 未接线的，必须覆盖文献3 表2 的全部字段

        这条测试的作用是防文档腐烂：日后若把 threshold 接上，
        必须同时把它从 RESERVED 移到 WIRED，否则这里会失败。
        """
        declared = set(WIRED_PROFILE_PARAMS) | set(RESERVED_PROFILE_PARAMS)
        for key, p in PERSONALITY_PROFILES.items():
            fields = {k for k in p if k != "zh"}
            self.assertEqual(fields, declared, f"{key} 的字段与声明不一致")
        self.assertEqual(set(WIRED_PROFILE_PARAMS) & set(RESERVED_PROFILE_PARAMS),
                         set())

    def test_reserved_params_really_have_no_effect(self):
        """**契约测试**：未接线参数改动后输出必须一模一样

        如果哪天有人把 threshold 接进公式而忘了更新文档，
        这条会立刻失败 —— 这正是我们要的提醒。

        注意：参照值必须在**改动档案之前**算好，否则等于拿改动后的
        系统去比自己，测试永远通过、也永远没有意义。
        """
        keys = ("ideal", "indifferent", "sensitive", "persistent")
        reference = {}
        for key in keys:
            dyn = EmotionDynamics(key)
            for _ in range(5):
                dyn.update_from_zh({"高焦虑": 0.6, "愉悦": 0.2}, dt=1.0)
            reference[key] = dyn.intensities["fear"]

        base = {k: dict(v) for k, v in PERSONALITY_PROFILES.items()}
        try:
            for key in keys:
                for param in RESERVED_PROFILE_PARAMS:
                    if param in PERSONALITY_PROFILES[key]:
                        cur = PERSONALITY_PROFILES[key][param]
                        # 数值改三倍，字符串换一个值
                        PERSONALITY_PROFILES[key][param] = (
                            cur * 3 if isinstance(cur, (int, float))
                            else "changed")
            for key in keys:
                dyn = EmotionDynamics(key)
                for _ in range(5):
                    dyn.update_from_zh({"高焦虑": 0.6, "愉悦": 0.2}, dt=1.0)
                self.assertAlmostEqual(
                    dyn.intensities["fear"], reference[key], places=9,
                    msg=f"{key}: 某个 RESERVED 参数意外生效了，请更新文档与常量")
        finally:
            PERSONALITY_PROFILES.clear()
            PERSONALITY_PROFILES.update(base)


class TestSnapshot(unittest.TestCase):
    """快照契约（会直接进 Prompt 与前端，字段必须稳定、可序列化）"""

    def test_snapshot_fields(self):
        dyn = EmotionDynamics()
        st = dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0).to_dict()
        for key in ("intensities", "durations", "mood", "mood_strength",
                    "polarity", "dominant", "dominant_zh",
                    "dominant_intensity", "level", "probabilities"):
            self.assertIn(key, st)

    def test_snapshot_is_json_friendly(self):
        import json
        dyn = EmotionDynamics()
        st = dyn.update_from_zh({"高焦虑": 0.8}, dt=1.0).to_dict()
        json.dumps(st)          # 不可序列化会直接抛错

    def test_calm_state_reports_calm(self):
        """没有任何情绪被激起时，主导情绪必须是「平静」而非空串"""
        dyn = EmotionDynamics()
        st = dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        self.assertEqual(st.dominant_zh, "平静")
        self.assertEqual(st.level, "unaware")

    def test_effective_emotion_prefers_dominant(self):
        dyn = EmotionDynamics()
        for _ in range(5):
            dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        self.assertEqual(dyn.effective_emotion_zh(), "高焦虑")

    def test_effective_emotion_falls_back(self):
        dyn = EmotionDynamics()
        self.assertEqual(dyn.effective_emotion_zh(fallback="平静"), "平静")

    def test_intensity_of_zh(self):
        dyn = EmotionDynamics()
        for _ in range(5):
            dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        self.assertGreater(dyn.intensity_of_zh("高焦虑"), 0.0)
        self.assertEqual(dyn.intensity_of_zh("愉悦"), 0.0)

    def test_reset_clears_state(self):
        dyn = EmotionDynamics()
        for _ in range(5):
            dyn.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        dyn.reset()
        self.assertEqual(dyn.mood, 0.0)
        self.assertEqual(sum(dyn.intensities.values()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
