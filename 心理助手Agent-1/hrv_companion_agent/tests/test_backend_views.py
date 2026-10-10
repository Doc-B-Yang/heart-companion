# -*- coding: utf-8 -*-
"""对外状态视图层单元测试（backend/views.py）

这一层做两件会被追问的事，所以单独测：

  ① **PAD 连续量 → 焦虑/压力/疲劳** 的映射。它是"情绪内区分度"的来源：
     按标签查表时，两位都被判「高焦虑」的用户在 APP 上看到的紧张值
     完全相同；换成 PAD 之后才有区分。这里断言映射的单调方向，
     以及值域在满量程组合下不越界。
  ② **哪些数值可以对外**。默认只给语义区间 + 质量等级，
     原始生理值收在 EXPOSE_RAW_METRICS 后面。这与 v1 行为相反，是有意的：
     实测已确认 PPG 会出现「HR 可信但 HRV 是假的」，把数值直接交给上层
     会诱导出伪精确表达。
"""

import unittest

from hrv_companion_agent import config
from hrv_companion_agent.backend import views


def _pad(p: float, a: float, d: float, conf: float = 0.7,
         dconf: float = 0.25) -> dict:
    return {"P": p, "A": a, "D": d, "method": "rule",
            "confidence": conf, "d_confidence": dconf}


class TestPadToDetail(unittest.TestCase):
    """PAD → anxiety / stress / fatigue / valence"""

    def test_neutral_maps_to_centre(self):
        out = views.detail_from_pad(_pad(0.0, 0.0, 0.0))
        self.assertAlmostEqual(out["anxiety_level"], 50.0, places=1)
        self.assertAlmostEqual(out["stress_level"], 50.0, places=1)
        self.assertAlmostEqual(out["fatigue_level"], 50.0, places=1)
        self.assertAlmostEqual(out["valence"], 0.5, places=2)

    def test_source_is_declared(self):
        self.assertEqual(views.detail_from_pad(_pad(0, 0, 0))["source"], "pad")
        self.assertEqual(
            views.detail_from_table({"emotion": "平静"})["source"], "table")
        self.assertEqual(views.detail_view(None)["source"], "none")

    def test_anxiety_rises_with_arousal(self):
        """唤醒度升高 → 紧张上升（焦虑的必要条件之一）"""
        low = views.detail_from_pad(_pad(-2.0, -2.0, 0.0))["anxiety_level"]
        high = views.detail_from_pad(_pad(-2.0, 2.0, 0.0))["anxiety_level"]
        self.assertGreater(high, low)

    def test_anxiety_rises_as_pleasure_falls(self):
        good = views.detail_from_pad(_pad(2.0, 2.0, 0.0))["anxiety_level"]
        bad = views.detail_from_pad(_pad(-2.0, 2.0, 0.0))["anxiety_level"]
        self.assertGreater(bad, good)

    def test_fatigue_falls_with_arousal(self):
        """疲劳随唤醒度下降而上升 —— 与焦虑的方向相反

        这条正是"两个构念不可混为一谈"的体现：低唤醒+负效价是疲劳，
        高唤醒+负效价才是焦虑。若两者同向单调，说明映射写错了。
        """
        tired = views.detail_from_pad(_pad(-2.0, -2.0, -1.0))["fatigue_level"]
        wired = views.detail_from_pad(_pad(-2.0, 2.0, -1.0))["fatigue_level"]
        self.assertGreater(tired, wired)
        # 同时确认：高唤醒时焦虑更高而疲劳更低（交叉）
        self.assertGreater(
            views.detail_from_pad(_pad(-2.0, 2.0, 0.0))["anxiety_level"],
            views.detail_from_pad(_pad(-2.0, 2.0, 0.0))["fatigue_level"])

    def test_low_dominance_adds_fatigue(self):
        """低支配感加重疲劳（无力感），对焦虑也应略有加成"""
        with_d = views.detail_from_pad(_pad(-1.0, -1.0, 3.0))
        without_d = views.detail_from_pad(_pad(-1.0, -1.0, -3.0))
        self.assertGreater(without_d["fatigue_level"], with_d["fatigue_level"])

    def test_valence_maps_monotonically(self):
        vals = [views.detail_from_pad(_pad(p, 0.0, 0.0))["valence"]
                for p in (-4.0, -2.0, 0.0, 2.0, 4.0)]
        self.assertEqual(vals, sorted(vals))
        self.assertAlmostEqual(vals[0], 0.0, places=2)
        self.assertAlmostEqual(vals[-1], 1.0, places=2)

    def test_all_levels_bounded_at_extremes(self):
        """满量程的 8 个角点组合都不得越出 [0, 100] / [0, 1]"""
        for p in (-4.0, 4.0):
            for a in (-4.0, 4.0):
                for d in (-4.0, 4.0):
                    out = views.detail_from_pad(_pad(p, a, d))
                    for key in ("anxiety_level", "stress_level",
                                "fatigue_level"):
                        self.assertGreaterEqual(out[key], 0.0)
                        self.assertLessEqual(out[key], 100.0)
                    self.assertGreaterEqual(out["valence"], 0.0)
                    self.assertLessEqual(out["valence"], 1.0)

    def test_out_of_range_pad_does_not_overflow(self):
        """PAD 越界（模型异常/脏数据）时仍必须钳在界内，不得把脏值直传 APP"""
        out = views.detail_from_pad(_pad(99.0, -99.0, 99.0))
        for key in ("anxiety_level", "stress_level", "fatigue_level"):
            self.assertGreaterEqual(out[key], 0.0)
            self.assertLessEqual(out[key], 100.0)
        self.assertGreaterEqual(out["valence"], 0.0)
        self.assertLessEqual(out["valence"], 1.0)

    def test_confidence_is_carried_through(self):
        """D 轴置信度必须透传 —— APP 需要据此决定要不要展示该维度"""
        out = views.detail_from_pad(_pad(0, 0, 0, conf=0.62, dconf=0.21))
        self.assertAlmostEqual(out["pad_confidence"], 0.62, places=6)
        self.assertAlmostEqual(out["d_confidence"], 0.21, places=6)

    def test_continuous_not_stepwise(self):
        """**关键在于连续**：P/A 微调应带来数值变化，而不是同一档位跳变

        按标签查表会给出阶梯状常量；这条测试钉住"改成连续量"这件事本身。
        """
        a = views.detail_from_pad(_pad(-1.0, 1.0, 0.0))["anxiety_level"]
        b = views.detail_from_pad(_pad(-1.0, 1.5, 0.0))["anxiety_level"]
        self.assertNotAlmostEqual(a, b, places=1)


class TestDetailView(unittest.TestCase):
    """detail_view 的分派与兜底"""

    def test_prefers_pad(self):
        st = {"emotion": "高焦虑", "risk_score": 80,
              "pad": _pad(-2.0, 2.0, -1.0)}
        out = views.detail_view(st)
        self.assertEqual(out["source"], "pad")
        self.assertEqual(out["raw_risk"], 80)

    def test_falls_back_to_table(self):
        st = {"emotion": "高焦虑", "confidence": 0.8, "risk_score": 80}
        out = views.detail_view(st)
        self.assertEqual(out["source"], "table")
        self.assertGreater(out["anxiety_level"], 50)

    def test_no_state_is_safe(self):
        out = views.detail_view(None)
        self.assertEqual(out["source"], "none")
        self.assertEqual(out["anxiety_level"], 0)

    def test_unknown_emotion_uses_default_row(self):
        out = views.detail_view({"emotion": "未知情绪", "confidence": 0.5})
        self.assertEqual(out["source"], "table")
        self.assertAlmostEqual(out["valence"], 0.5, places=2)

    def test_manual_injection_without_pad_uses_table(self):
        """手动注入情绪时 pad=None，必须走兜底而不能报错"""
        out = views.detail_view({"emotion": "疲劳", "confidence": 0.9,
                                 "pad": None})
        self.assertEqual(out["source"], "table")
        self.assertGreater(out["fatigue_level"], out["anxiety_level"])


class TestRiskBand(unittest.TestCase):
    """语义区间 —— 与 prompts.py 的 _RISK_BANDS 必须是同一套口径"""

    def test_bands(self):
        self.assertEqual(views.band(95), "很高")
        self.assertEqual(views.band(80), "很高")
        self.assertEqual(views.band(79.9), "偏高")
        self.assertEqual(views.band(60), "偏高")
        self.assertEqual(views.band(40), "中等")
        self.assertEqual(views.band(20), "较低")
        self.assertEqual(views.band(0), "低")

    def test_bad_input_is_unknown(self):
        self.assertEqual(views.band(None), "低")
        self.assertEqual(views.band("abc"), "未知")

    def test_matches_prompt_layer(self):
        """跨层一致：视图层与 Prompt 层的**阈值必须完全相同**

        不断言标签字符串一致 —— 两层的展示需求不同（界面要短、模型要提示），
        标签本来就该不同。真正必须一致的是**分档的位置**，
        否则就会出现"前端显示偏高、模型被告知中等"这种自相矛盾：
        两边单看都没错，只有放在一起才看得出来。
        """
        from hrv_companion_agent.agent.prompts import _RISK_BANDS
        self.assertEqual([lo for lo, _ in views.RISK_BANDS],
                         [lo for lo, _ in _RISK_BANDS])

    def test_both_layers_derive_from_config(self):
        """两层的区间都必须**派生自** config，而不是各写一份

        这条是防回归的关键：谁要是图省事在某一层里重新硬编码一份区间，
        上面那条测试可能仍通过（只要数值抄对了），但下次改配置就会分叉。
        """
        from hrv_companion_agent.agent.prompts import _RISK_BANDS
        self.assertEqual([lo for lo, _ in views.RISK_BANDS],
                         [lo for lo, _l, _h in config.RISK_BANDS])
        self.assertEqual([lo for lo, _ in _RISK_BANDS],
                         [lo for lo, _l, _h in config.RISK_BANDS])
        for lo, label, hint in config.RISK_BANDS:
            self.assertTrue(label)
            self.assertTrue(hint)
        # 必须覆盖到 0，否则最低一档会出现"无档可归"
        self.assertEqual(config.RISK_BANDS[-1][0], 0)

    def test_every_score_maps_to_a_label(self):
        """0~100 全量程都必须能拿到非空标签（含边界值）"""
        from hrv_companion_agent.agent.prompts import _band, _RISK_BANDS
        for v in list(range(0, 101)) + [-5, 150]:
            self.assertTrue(views.band(v))
            self.assertTrue(_band(v, _RISK_BANDS))


class TestPublicState(unittest.TestCase):
    """对外状态的裁剪规则"""

    STATE = {
        "timestamp": 1700000000.0, "emotion": "高焦虑", "confidence": 0.78,
        "risk_score": 82, "layer": "rule",
        "detail": {"hr_bpm": 95},
        "quality": {"quality_ok": True, "quality_level": "合格",
                    "jitter": 0.03},
        "quality_level": "合格", "jitter": 0.03,
        "pad": _pad(-2.0, 2.0, -1.0),
        "affect": {"level": "moderate", "mood": -0.3},
        "hr_bpm": 95, "sdnn_ms": 31.0, "rmssd_ms": 22.0, "pnn50": 4.0,
        "lf_hf": 3.2, "sd_hr_bpm": 5.0, "sig_quality": 0.9,
        "feats": {"hr_bpm": 95.0, "rmssd_ms": 22.0},
        "rel_features": {"rmssd_ms": 0.51},
    }

    def test_none_returns_none(self):
        self.assertIsNone(views.public_state(None))

    def test_semantic_fields_present(self):
        out = views.public_state(self.STATE)
        for key in ("emotion", "confidence", "risk_score", "risk_band",
                    "detail", "quality", "quality_level", "jitter",
                    "pad", "affect", "privacy_notice"):
            self.assertIn(key, out)
        self.assertEqual(out["risk_band"], "很高")

    def test_raw_metrics_hidden_by_default(self):
        """**v2 契约**：默认不下发原始生理值（与 v1 相反，有意为之）"""
        old = config.EXPOSE_RAW_METRICS
        try:
            config.EXPOSE_RAW_METRICS = False
            out = views.public_state(self.STATE)
        finally:
            config.EXPOSE_RAW_METRICS = old
        self.assertNotIn("metrics", out)
        # 顶层不应出现裸数值（jitter 是质量判据，属于必要信息，保留）
        self.assertNotIn("hr_bpm", out)
        self.assertNotIn("rmssd_ms", out)
        self.assertNotIn("feats", out)

    def test_raw_metrics_available_for_debugging(self):
        old = config.EXPOSE_RAW_METRICS
        try:
            config.EXPOSE_RAW_METRICS = True
            out = views.public_state(self.STATE)
        finally:
            config.EXPOSE_RAW_METRICS = old
        self.assertIn("metrics", out)
        self.assertAlmostEqual(out["metrics"]["hr_bpm"], 95.0, places=6)
        self.assertIn("rel_features", out["metrics"])

    def test_quality_is_always_visible(self):
        """质量**必须**始终可见 —— 上层要靠它决定这次读数能不能用

        它不属于"原始生理值"，而是"这个读数可不可信"的元信息，
        收起来会让 APP 无法履行"先看质量再下结论"的约定。
        """
        old = config.EXPOSE_RAW_METRICS
        try:
            config.EXPOSE_RAW_METRICS = False
            out = views.public_state(self.STATE)
        finally:
            config.EXPOSE_RAW_METRICS = old
        self.assertEqual(out["quality_level"], "合格")
        self.assertAlmostEqual(out["jitter"], 0.03, places=6)
        self.assertTrue(out["quality"]["quality_ok"])

    def test_json_serialisable(self):
        import json
        json.dumps(views.public_state(self.STATE))   # 不可序列化会直接抛错

    def test_missing_optional_fields_are_tolerated(self):
        """只有最小状态时也不得抛错（"数据积累中"阶段就是这种状态）

        注意 detail 的来源是 table 而不是 none —— 此时状态本身是存在的，
        只是还没有 PAD；"none" 只在**完全没有状态**（首次启动）时出现。
        这个区分有意义：APP 可以据此判断"该显示暂无数据"还是"该显示兜底值"。
        """
        out = views.public_state({"emotion": "数据积累中", "confidence": 0.0,
                                  "risk_score": 0.0, "layer": "insufficient_data",
                                  "detail": {}, "quality": {}, "timestamp": 1.0})
        self.assertEqual(out["emotion"], "数据积累中")
        self.assertEqual(out["detail"]["source"], "table")
        self.assertIsNone(out["pad"])
        self.assertIsNone(views.public_state(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
