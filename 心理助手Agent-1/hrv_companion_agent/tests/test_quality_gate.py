# -*- coding: utf-8 -*-
"""信号质量门禁单元测试（本项目实测结论的代码化）

这个文件守的是一条从实验里换来的教训：

    **PPG 会出现「HR 完全正常，但 HRV 完全是假的」。**

指腹接触不良时实测 HR 显示 73 bpm（看着很对），而 RMSSD 达 130ms、
jitter 0.455（静息应 ≤0.08），脉搏带功率占比仅 0.01%。若把这种数据
放进情绪识别并据此触发关怀，等于用一个噪声数字去打扰用户。

因此本文件断言三件事：
  ① **jitter 能识破假象**，而单纯看 HR 不能；
  ② 质量不合格时**必须**降置信度并抑制触发（宁可少关怀一次）；
  ③ 分级阈值必须由 config 驱动 —— 现场把配置收紧却行为照旧，
     是最难排查的一类问题。
"""

import unittest

from hrv_companion_agent import config
from hrv_companion_agent.algorithms.emotion_recognizer import EmotionResult
from hrv_companion_agent.algorithms.hrv_extractor import HRVExtractor
from hrv_companion_agent.algorithms.trigger_engine import TriggerEngine


def _regular_rr(n: int = 60, mean: float = 850.0, jitter_ms: float = 0.0):
    """构造一段 RR 间期（毫秒）

    ``jitter_ms`` 是相邻拍之间的**交替偏移幅度**：
    奇数拍 +a、偶数拍 −a，于是相邻差恒为 2a，RMSSD = 2a。
    这样 jitter = 2a / mean 可控且可解析验算。

    幅度必须控制在 ~±65ms 以内（相对 850ms 约 7.6%）：再大就会被
    ``clean_ibi`` 当作离群 RR 整段剔除，反而变成"有效拍数不足"，
    测不到抖动本身。
    """
    out = []
    for i in range(n):
        out.append(mean + (jitter_ms if i % 2 == 0 else -jitter_ms))
    return out


class TestJitterMetric(unittest.TestCase):
    """jitter = RMSSD / mean(IBI) —— 识破"HR 正常但 HRV 是假的" """

    def setUp(self):
        self.ex = HRVExtractor()

    def test_clean_signal_passes(self):
        rep = self.ex.quality_metrics(_regular_rr(jitter_ms=8.0))
        self.assertTrue(rep["quality_ok"])
        self.assertEqual(rep["quality_level"], "合格")
        self.assertLessEqual(rep["jitter"], config.QUALITY_JITTER_GOOD)

    def test_hrv_is_fake_while_hr_looks_fine(self):
        """**核心断言**：抖动超标时，HR 依然"正常"，jitter 必须报警

        实测指腹贴放：HR 73 bpm、RMSSD 130ms、jitter 0.455、接触三项指标
        全不合格。这里复现其**判据特征**（HR 落在正常静息区间 + jitter 超标）：

          · HR 仍落在 60~100 —— 说明只看 HR 会完全漏判；
          · quality_ok 为 False —— 说明 jitter 能抓住它。

        关于幅度：实测的 0.455 在指腹场景由"双峰形态"造成（隔 176ms 的
        早到假峰把真峰顶进不应期），这种形态在本合成序列里做不到 ——
        等幅交替只要超过约 ±70ms 就会被 clean_ibi 整段判为离群。
        所以本测试验的是**机制**（哪个指标能识破假象），
        不是复刻实测数值；数值本身记录在说明书与 `_similar` 的实测注里。
        """
        rr = _regular_rr(n=60, mean=850.0, jitter_ms=65.0)
        rep = self.ex.quality_metrics(rr)
        hr = 60000.0 / (sum(rr) / len(rr))
        self.assertGreater(hr, 60.0)
        self.assertLess(hr, 100.0)                       # HR 看起来没问题
        self.assertGreater(rep["jitter"], config.QUALITY_JITTER_POOR)
        self.assertFalse(rep["quality_ok"])
        self.assertEqual(rep["quality_level"], "不可用")

    def test_levels_are_monotone_in_jitter(self):
        levels = []
        for j in (2.0, 20.0, 40.0, 65.0):
            levels.append(self.ex.quality_metrics(
                _regular_rr(jitter_ms=j))["quality_level"])
        order = {"合格": 0, "勉强": 1, "不可用": 2, "不足": 3}
        self.assertEqual([order[x] for x in levels],
                         sorted(order[x] for x in levels))

    def test_insufficient_beats(self):
        rep = self.ex.quality_metrics([850.0, 860.0])
        self.assertFalse(rep["quality_ok"])
        self.assertEqual(rep["quality_level"], "不足")

    def test_reason_is_actionable(self):
        """不可用时必须给出**可下手**的原因，而不是只说"不可用"

        jitter 超标（接触/灌注问题）与伪迹比例超标（异位搏动、漏检双计）
        的改善措施完全不同，混在一起等于没说。
        """
        rep = self.ex.quality_metrics(_regular_rr(jitter_ms=65.0))
        self.assertFalse(rep["quality_ok"])
        self.assertTrue(rep["reason"])
        self.assertIn("接触", rep["reason"])

    def test_ectopic_reason_is_distinct_from_jitter(self):
        """伪迹比例超标的提示必须与抖动超标**不同**

        这条守护的正是"原因可下手"：两种失效的处置方式不一样，
        如果 reason 只有一个含糊的"不可用"，现场只能靠猜。
        """
        rr = _regular_rr(n=60, mean=850.0, jitter_ms=8.0)
        for k in range(15, 60, 3):          # 每隔三拍插一个粗大伪迹
            rr[k] = 1500.0
        rep = self.ex.quality_metrics(rr)
        self.assertFalse(rep["quality_ok"])
        self.assertGreater(rep["ectopic_ratio"], config.QUALITY_ECTOPIC_MAX)
        self.assertLessEqual(rep["jitter"], config.QUALITY_JITTER_POOR)
        self.assertIn("伪迹", rep["reason"])

    def test_ok_has_empty_reason(self):
        rep = self.ex.quality_metrics(_regular_rr(jitter_ms=5.0))
        self.assertTrue(rep["quality_ok"])
        self.assertEqual(rep["reason"], "")


class TestThresholdsComeFromConfig(unittest.TestCase):
    """阈值必须真的由 config 驱动（改配置 → 行为必须变）"""

    def test_config_values_are_ordered(self):
        self.assertLess(config.QUALITY_JITTER_GOOD, config.QUALITY_JITTER_POOR)
        self.assertLess(config.QUALITY_ECTOPIC_GOOD, config.QUALITY_ECTOPIC_MAX)

    def test_relaxing_threshold_changes_verdict(self):
        """收紧/放宽 jitter 门限，同一段信号的结论必须随之改变

        这条测试的价值：它证明配置不是装饰。
        若哪天有人把阈值重新写死在算法里，这条会失败。
        """
        ex = HRVExtractor()
        rr = _regular_rr(jitter_ms=40.0)     # 中等抖动（验算值 jitter≈0.094）
        old_good = config.QUALITY_JITTER_GOOD
        old_poor = config.QUALITY_JITTER_POOR
        try:
            # 收紧：把"勉强"档都关掉
            config.QUALITY_JITTER_POOR = 0.05
            strict = ex.quality_metrics(rr)
            # 放宽：连"合格"档都能通过
            config.QUALITY_JITTER_GOOD = 5.0
            config.QUALITY_JITTER_POOR = 5.0
            loose = ex.quality_metrics(rr)
        finally:
            config.QUALITY_JITTER_GOOD = old_good
            config.QUALITY_JITTER_POOR = old_poor
        self.assertFalse(strict["quality_ok"])
        self.assertEqual(strict["quality_level"], "不可用")
        self.assertTrue(loose["quality_ok"])
        self.assertEqual(loose["quality_level"], "合格")


class TestRelativeFeatures(unittest.TestCase):
    """文献1「相对特征值法」：符号方向必须能自解释"""

    BASELINE = {"hr_bpm": 70.0, "rmssd_ms": 45.0, "sdnn_ms": 50.0,
                "pnn50_pct": 20.0, "lf_hf": 1.5}

    def test_sign_convention(self):
        """正 = 低于基线，负 = 高于基线（这是本项目全链路统一的约定）"""
        cur = dict(self.BASELINE, rmssd_ms=27.0, hr_bpm=91.0)
        rel = HRVExtractor.relative_features(cur, self.BASELINE)
        self.assertGreater(rel["rmssd_ms"], 0)      # RMSSD 掉了 → 正
        self.assertLess(rel["hr_bpm"], 0)           # HR 升了 → 负

    def test_zero_when_equal(self):
        rel = HRVExtractor.relative_features(dict(self.BASELINE), self.BASELINE)
        for v in rel.values():
            self.assertAlmostEqual(v, 0.0, places=6)

    def test_clipping_prevents_explosion(self):
        """基线极小时相对值会爆炸，必须钳位（否则一条特征就能主导判分）"""
        cur = dict(self.BASELINE, rmssd_ms=1000.0)
        rel = HRVExtractor.relative_features(cur, self.BASELINE, clip=3.0)
        self.assertGreaterEqual(rel["rmssd_ms"], -3.0)
        self.assertLessEqual(rel["rmssd_ms"], 3.0)

    def test_missing_baseline_is_omitted(self):
        """缺基线的特征**不得**出现在结果里（v2 修正的量纲陷阱）

        原实现以 ``out = dict(feats)`` 起手，"没基线就保留原值"，结果
        原始量纲的值混进了比值空间：基线只有 7 个键而 RELATIVE_KEYS 有 14 个，
        于是 sampen=1.5 会被消费方读成"相对偏移 +1.5"，
        还会因为数值最大而占据 pad_tool 归因列表首位 —— 静默错误，最难查。
        """
        rel = HRVExtractor.relative_features({"hr_bpm": 90.0}, self.BASELINE)
        self.assertIn("hr_bpm", rel)
        self.assertNotIn("rmssd_ms", rel)
        self.assertNotIn("sampen", rel)

    def test_unknown_keys_are_never_returned(self):
        """不在 RELATIVE_KEYS 里的键一律不返回（无论基线有没有）"""
        cur = dict(self.BASELINE, hr_bpm=91.0, sampen=1.5, dfa_alpha1=1.02)
        base = dict(self.BASELINE, sampen=1.6, dfa_alpha1=1.0)
        rel = HRVExtractor.relative_features(cur, base)
        self.assertNotIn("不存在的特征", rel)
        for k in rel:
            self.assertIn(k, HRVExtractor.RELATIVE_KEYS)

    def test_all_returned_values_are_ratios_not_raw(self):
        """**契约测试**：结果里每个值都必须是比值（量级 ≪ 1），不含原始值

        判据取 |v| <= clip 且不出现"明显是原始特征量纲"的大数。
        这条测试是对上面那个陷阱的直接围堵。
        """
        cur = dict(self.BASELINE, hr_bpm=91.0, rmssd_ms=27.0,
                   sampen=1.5, dfa_alpha1=1.02)
        base = dict(self.BASELINE, sampen=1.5, dfa_alpha1=1.0)
        rel = HRVExtractor.relative_features(cur, base)
        for k, v in rel.items():
            self.assertLessEqual(abs(v), 3.0, f"{k} 越出钳位范围")
        # sampen 这次有基线且几乎无变化 → 应接近 0，而不是 1.5
        self.assertAlmostEqual(rel["sampen"], 0.0, places=3)

    def test_empty_when_no_baseline(self):
        """完全无基线时返回空字典（而不是把 feats 原样回传）"""
        self.assertEqual(HRVExtractor.relative_features({"hr_bpm": 90.0}, None),
                         {})

    def test_none_value_is_omitted(self):
        self.assertEqual(
            HRVExtractor.relative_features({"hr_bpm": None}, self.BASELINE), {})


class TestTriggerQualityGate(unittest.TestCase):
    """质量门禁在触发链路上的行为"""

    @staticmethod
    def _result(risk: float = 70.0, emotion: str = "高焦虑"):
        return EmotionResult(emotion=emotion, confidence=0.8,
                             risk_score=risk, layer="rule",
                             detail={"hr_bpm": 95.0})

    def test_unreliable_signal_suppresses_trigger(self):
        """质量不可用 → 不得触发（用噪声打扰用户比不打扰更糟）"""
        eng = TriggerEngine()
        d = eng.decide(self._result(70.0), risk_history=[60, 65, 68],
                       quality={"quality_ok": False, "quality_level": "不可用",
                                "jitter": 0.45})
        self.assertFalse(d.should_trigger)
        self.assertIn("质量", eng.last_gate_reason)

    def test_ok_signal_allows_trigger(self):
        eng = TriggerEngine()
        d = eng.decide(self._result(70.0), risk_history=[60, 65, 68],
                       quality={"quality_ok": True, "quality_level": "合格",
                                "jitter": 0.04})
        self.assertTrue(d.should_trigger)

    def test_no_quality_report_keeps_old_behaviour(self):
        """不传质量报告时行为与 v1 一致（向后兼容）"""
        eng = TriggerEngine()
        d = eng.decide(self._result(70.0), risk_history=[60, 65, 68])
        self.assertTrue(d.should_trigger)
        self.assertEqual(eng.last_gate_reason, "")

    def test_emergency_still_passes_unreliable_signal(self):
        """**例外条款**：极高分仍放行 —— 宁可多关怀一次

        质量差时若连危机风险也一起挡掉，后果不可接受。
        这条测试守住这个"宁可错杀"的不对称设计。
        """
        eng = TriggerEngine()
        d = eng.decide(self._result(99.0, "高焦虑"), risk_history=[95, 97],
                       quality={"quality_ok": False, "quality_level": "不可用",
                                "jitter": 0.5})
        self.assertTrue(d.should_trigger)
        self.assertEqual(d.trigger_level, "emergency")

    def test_gate_can_be_disabled(self):
        """QUALITY_GATE_ENABLED=0 时行为回到 v1（便于对照实验）"""
        eng = TriggerEngine()
        old = config.QUALITY_GATE_ENABLED
        try:
            config.QUALITY_GATE_ENABLED = False
            d = eng.decide(self._result(70.0), risk_history=[60, 65],
                           quality={"quality_ok": False,
                                    "quality_level": "不可用", "jitter": 0.5})
        finally:
            config.QUALITY_GATE_ENABLED = old
        self.assertTrue(d.should_trigger)

    def test_consecutive_guard_can_delay_trigger(self):
        """连续确认：单点误报不足以发起关怀（PPG 信号不稳时尤其重要）"""
        old = config.TRIGGER_MIN_CONSECUTIVE
        try:
            config.TRIGGER_MIN_CONSECUTIVE = 3
            eng = TriggerEngine()
            q = {"quality_ok": True, "quality_level": "合格", "jitter": 0.03}
            first = eng.decide(self._result(70.0), risk_history=[60],
                               quality=q)
            self.assertFalse(first.should_trigger)
            self.assertEqual(first.detail.get("gate"), "consecutive")
            eng.decide(self._result(70.0), risk_history=[60], quality=q)
            third = eng.decide(self._result(70.0), risk_history=[60], quality=q)
            self.assertTrue(third.should_trigger)
        finally:
            config.TRIGGER_MIN_CONSECUTIVE = old

    def test_consecutive_guard_resets_on_drop(self):
        old = config.TRIGGER_MIN_CONSECUTIVE
        try:
            config.TRIGGER_MIN_CONSECUTIVE = 3
            eng = TriggerEngine()
            q = {"quality_ok": True, "quality_level": "合格", "jitter": 0.03}
            eng.decide(self._result(70.0), risk_history=[60], quality=q)
            eng.decide(self._result(10.0, "平静"), risk_history=[60], quality=q)
            d = eng.decide(self._result(70.0), risk_history=[60], quality=q)
            self.assertFalse(d.should_trigger, "风险回落后计数必须清零")
        finally:
            config.TRIGGER_MIN_CONSECUTIVE = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
