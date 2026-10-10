# -*- coding: utf-8 -*-
"""PAD 三维情感模型单元测试（对应文献2 与计划书 §4.2 pad_model.py）

覆盖四层：
  ① 量表公式 —— 12 项、反向题符号、式(1) 系数方向
  ② 分类映射 —— pad_to_emotion 的分界行为
  ③ 规则路径 —— RulePAD 的生理方向性与置信度保护
  ④ 回归路径 —— CV 口径、CCC 的惩罚性质、**阴性对照**

第 ④ 组里最重要的不是"能不能学到"，而是"学不到时会不会假装学到"：
最后一项测试用与特征无关的随机标签训练，断言 CV ρc 落在噪声水平 ——
如果它跑出显著正值，说明管线存在数据泄漏，比模型不准严重得多。
"""

import math
import unittest

from hrv_companion_agent.algorithms.pad_model import (
    PAD_FEATURE_NAMES, PAD_SCALE_ITEMS, PADRegressor, RulePAD,
    _regression_metrics, estimate_pad, pad_from_questionnaire, pad_to_emotion)


class TestScaleFormula(unittest.TestCase):
    """文献2 式(1) 的 12 项量表换算"""

    def test_item_count(self):
        self.assertEqual(len(PAD_SCALE_ITEMS), 12)

    def test_item_layout(self):
        """12 项是**交错**排列：P 在 V1/V4/V7/V10，A 在 V2/V5/V8/V11，D 在 V3/V6/V9/V12

        交错而非分组是量表设计导致的（同一维度的正反向题交替出现，
        防止被试察觉规律后形成固定反应）。这里把顺序钉死，
        因为后续任何按位置取值的代码都必须与式(1) 的下标完全一致。
        """
        dims = [dim for dim, _left, _right in PAD_SCALE_ITEMS]
        self.assertEqual(dims, ["P", "A", "D"] * 4)

    def test_each_item_has_two_poles(self):
        """每项都必须是「左端词 / 右端词」的对偶描述，且两词不同"""
        for dim, left, right in PAD_SCALE_ITEMS:
            self.assertIn(dim, ("P", "A", "D"))
            self.assertTrue(left and right)
            self.assertNotEqual(left, right)

    def test_anxious_vector_hits_paper_criteria(self):
        """焦虑向量应命中文献2 的「P<0, A>0, D<0」区域判据

        取值按交错顺序：下标 0/3/6/9 给 P，1/4/7/10 给 A，2/5/8/11 给 D。
        反向题（V2/V4/V6/V8/V9/V12）的符号已由式(1) 统一，
        故这里只需按"越靠右端词、分值绝对值越大"填写。
        """
        # 焦虑：愤怒/轻蔑/不满/不快(P−)、清醒/激动/专注/兴奋(A+)、受控/被支配/被影响/被忽视(D−)
        anxious = [-3, -3, -3, 3, 3, 3, -3, -3, -3, 3, 3, 3]
        pad = pad_from_questionnaire(anxious)
        self.assertLess(pad["P"], 0)
        self.assertGreater(pad["A"], 0)
        self.assertLess(pad["D"], 0)

    def test_relaxed_vector_hits_paper_criteria(self):
        # 放松：感兴趣/友好/满足/愉快(P+)、困倦/平静/分心/沉静(A−)、主控/支配/有影响力/重要(D+)
        relaxed = [3, 3, 3, -3, -3, -3, 3, 3, 3, -3, -3, -3]
        pad = pad_from_questionnaire(relaxed)
        self.assertGreater(pad["P"], 0)
        self.assertLess(pad["A"], 0)
        self.assertGreater(pad["D"], 0)

    def test_value_range(self):
        """极值输入不应越界（量表 7 级，居中后 ±3）"""
        extreme = [3] * 12
        pad = pad_from_questionnaire(extreme)
        for axis in ("P", "A", "D"):
            self.assertLessEqual(abs(pad[axis]), 3.0001)

    def test_axis_is_mean_not_sum(self):
        """式(1) 是组内带符号求和后除以 4，故单项贡献为 ±0.25 而非 ±1

        构造 P 的四项同向：v1=+1、v4=−1（反向题，取负号后仍为 +1）、
        v7=+1、v10=−1 → P=(1−(−1)+1−(−1))/4=+1。
        """
        v = [0.0] * 12
        v[0], v[3], v[6], v[9] = 1.0, -1.0, 1.0, -1.0   # P 的四项
        pad = pad_from_questionnaire(v)
        self.assertAlmostEqual(pad["P"], 1.0, places=6)
        # 只有一项时贡献为 0.25，证明是"除以 4"而不是"求和"
        v2 = [0.0] * 12
        v2[0] = 1.0
        self.assertAlmostEqual(pad_from_questionnaire(v2)["P"], 0.25, places=6)

    def test_reverse_items_actually_reverse(self):
        """反向题的符号必须真的翻转

        对 P 而言 V4 是反向题（右端词是「轻蔑的」）。要让 P 上升，
        V1 应取正、V4 应取**负**。因此 (v1=+1, v4=−1) 得到的 P
        必须**大于** (v1=+1, v4=+1)。这一断言方向容易写反，
        正是它保证了"按右端词填写"这一约定在全表一致。
        """
        same_dir = [0.0] * 12
        same_dir[0], same_dir[3] = 1.0, 1.0      # V4 与 V1 同向 → 贡献抵消
        reverse_dir = [0.0] * 12
        reverse_dir[0], reverse_dir[3] = 1.0, -1.0   # V4 反向 → 贡献叠加
        self.assertGreater(pad_from_questionnaire(reverse_dir)["P"],
                           pad_from_questionnaire(same_dir)["P"])

    def test_short_input_rejected(self):
        with self.assertRaises(ValueError):
            pad_from_questionnaire([1.0] * 11)


class TestEmotionMapping(unittest.TestCase):
    """pad_to_emotion：P/A 两轴划 5 类（D 不参与，因文献2 中 D 最弱）"""

    def test_high_anxiety(self):
        self.assertEqual(pad_to_emotion(-2.0, 2.0, -1.0), "高焦虑")

    def test_low_anxiety(self):
        self.assertEqual(pad_to_emotion(-2.0, 0.0, -1.0), "低焦虑")

    def test_fatigue(self):
        self.assertEqual(pad_to_emotion(-2.0, -2.0, -1.0), "疲劳")

    def test_pleasant(self):
        self.assertEqual(pad_to_emotion(2.0, 2.0, 1.0), "愉悦")

    def test_calm(self):
        self.assertEqual(pad_to_emotion(2.0, -2.0, 1.0), "平静")

    def test_d_axis_does_not_change_result(self):
        """同一 (P, A) 下 D 任意取值，分类必须一致

        这是有意设计：文献2 实测 D 轴 ρc 仅 0.30，让它参与离散化
        等于把最大的噪声源引进来。
        """
        for d in (-4.0, -1.0, 0.0, 1.0, 4.0):
            self.assertEqual(pad_to_emotion(-2.0, 2.0, d), "高焦虑")


class TestRulePAD(unittest.TestCase):
    """规则路径的生理方向性与置信度保护"""

    #: 用人群中值兜底（无个体基线）时的典型静息特征
    CALM = {"hr_bpm": 70.0, "rmssd_ms": 42.0, "sdnn_ms": 50.0,
            "pnn50_pct": 15.0, "lf_hf": 1.5, "sd_hr_bpm": 3.0}

    def test_arousal_rises_with_hr(self):
        """心率升高 + LF/HF 升高 → 唤醒度 A 上升（焦虑的必要条件之一）"""
        stressed = dict(self.CALM, hr_bpm=98.0, lf_hf=4.0)
        a_calm = RulePAD().predict(self.CALM, self.CALM).A
        a_stress = RulePAD().predict(stressed, self.CALM).A
        self.assertGreater(a_stress, a_calm)

    def test_pleasure_falls_with_rmssd(self):
        """RMSSD 相对基线下降 → 愉悦度 P 下降（副交感撤出的直接后果）"""
        suppressed = dict(self.CALM, rmssd_ms=18.0, sdnn_ms=22.0, pnn50_pct=2.0)
        p_calm = RulePAD().predict(self.CALM, self.CALM).P
        p_low = RulePAD().predict(suppressed, self.CALM).P
        self.assertLess(p_low, p_calm)

    def test_values_within_scale(self):
        """输出必须落在 ±4 量纲内，否则视图层换算会越界"""
        wild = {"hr_bpm": 200.0, "rmssd_ms": 1.0, "sdnn_ms": 1.0,
                "pnn50_pct": 0.0, "lf_hf": 50.0, "sd_hr_bpm": 40.0}
        r = RulePAD().predict(wild, self.CALM)
        for axis in (r.P, r.A, r.D):
            self.assertLessEqual(abs(axis), 4.0001)

    def test_confidence_needs_individual_baseline(self):
        """无个体基线时置信度必须显著更低（用的是人群中值兜底）"""
        no_base = RulePAD().predict(self.CALM, None)
        has_base = RulePAD().predict(self.CALM, self.CALM)
        self.assertLess(no_base.confidence, has_base.confidence)
        self.assertTrue(no_base.detail["based_on_baseline"] is False)
        self.assertTrue(has_base.detail["based_on_baseline"] is True)

    def test_high_jitter_discounts_confidence(self):
        """抖动指数高 → 置信度打折（实测已确认这是 HRV 失效的直接指征）"""
        clean = RulePAD().predict(dict(self.CALM, jitter=0.03), self.CALM)
        dirty = RulePAD().predict(dict(self.CALM, jitter=0.20), self.CALM)
        self.assertLess(dirty.confidence, clean.confidence)

    def test_d_confidence_always_below_overall(self):
        """D 轴置信度必须低于整体置信度（文献2：D 是最难预测的一轴）"""
        r = RulePAD().predict(self.CALM, self.CALM)
        self.assertLess(r.d_confidence, r.confidence)

    def test_estimate_pad_rule_path(self):
        """强制规则路径时，返回的 method 必须是 rule"""
        r = estimate_pad(self.CALM, self.CALM, None, prefer="rule")
        self.assertEqual(r.method, "rule")
        self.assertEqual(r.emotion_hint,
                         pad_to_emotion(r.P, r.A, r.D))


class TestUntrainedRegressor(unittest.TestCase):
    """未训练时不得伪装可用"""

    def test_not_available_without_model(self):
        reg = PADRegressor(model_path=None)
        self.assertFalse(reg.available)
        self.assertIsNone(reg.predict({}))

    def test_estimate_falls_back_to_rule(self):
        reg = PADRegressor(model_path=None)
        r = estimate_pad(TestRulePAD.CALM, None, reg)
        self.assertEqual(r.method, "rule")


class TestRegressionMetrics(unittest.TestCase):
    """文献2 式(11)(12)(13)：MSE / ρ / CCC(ρc)"""

    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0]
        m = _regression_metrics(y, y)
        self.assertAlmostEqual(m["mse"], 0.0, places=6)
        self.assertAlmostEqual(m["rho"], 1.0, places=6)
        self.assertAlmostEqual(m["ccc"], 1.0, places=6)

    def test_systematic_bias_is_penalised(self):
        """ρc 相对 ρ 的增量价值就在于此：等比例放大时 ρ=1 但 ρc 明显掉"""
        y = [1.0, 2.0, 3.0, 4.0]
        biased = [v * 1.5 for v in y]
        m = _regression_metrics(y, biased)
        self.assertAlmostEqual(m["rho"], 1.0, places=6)
        self.assertLess(m["ccc"], 0.99)

    def test_constant_prediction_penalised(self):
        """只会输出均值的模型：ρ 无定义（0）、ρc 应接近 0 而非虚高"""
        y = [1.0, 2.0, 3.0, 4.0]
        m = _regression_metrics(y, [2.5] * 4)
        self.assertAlmostEqual(m["rho"], 0.0, places=6)
        self.assertLessEqual(m["ccc"], 0.05)

    def test_anticorrelated(self):
        m = _regression_metrics([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
        self.assertLess(m["rho"], 0)
        self.assertLess(m["ccc"], 0)

    def test_short_series_does_not_crash(self):
        self.assertEqual(_regression_metrics([], [])["ccc"], 0.0)
        self.assertEqual(_regression_metrics([1.0], [1.0])["ccc"], 0.0)


class TestCrossValHonesty(unittest.TestCase):
    """CV 口径与阴性对照 —— 本文件分量最重的一组测试"""

    @staticmethod
    def _make(n=90, seed=7, learnable=True):
        import random
        rng = random.Random(seed)
        X, y = [], []
        for _ in range(n):
            p, a = rng.uniform(-3, 3), rng.uniform(-3, 3)
            feats = {nm: rng.gauss(0, 1) for nm in PAD_FEATURE_NAMES}
            feats["hr_bpm"] = 70 + 12 * a
            feats["rmssd_ms"] = 45 - 8 * p
            X.append([feats[nm] for nm in PAD_FEATURE_NAMES])
            # D 轴与任何特征都无关（learnable=False 时连 P/A 也随机）
            y.append([p if learnable else rng.uniform(-3, 3),
                      a if learnable else rng.uniform(-3, 3),
                      rng.uniform(-3, 3)])
        return X, y

    def test_cv_reported_as_source(self):
        """有 CV 结果时，metadata.ccc 的来源必须标成 cv 而非 in_sample"""
        X, y = self._make()
        res = PADRegressor(model_path=None).fit(X, y, use_pca=False,
                                                cv_folds=5, verbose=False)
        self.assertEqual(res["ccc_source"], "cv")
        self.assertTrue(res["cv_metrics"])

    def test_in_sample_optimistic_vs_cv(self):
        """样本内 ρc 必须**不低于** CV ρc（SVR 会记住训练点）

        不断言"高多少"—— 那取决于数据。只断言这个方向性关系，
        因为它是"用样本内指标汇报成绩会偏乐观"这一结论的依据。
        """
        X, y = self._make()
        res = PADRegressor(model_path=None).fit(X, y, use_pca=False,
                                                cv_folds=5, verbose=False)
        for axis in ("P", "A", "D"):
            self.assertGreaterEqual(res["metrics"][axis]["ccc"] + 1e-6,
                                    res["cv_metrics"][axis]["ccc"])

    def test_learnable_axis_generalises(self):
        """真有关联时 CV 应当学到（P 轴由 RMSSD 直接决定）"""
        X, y = self._make()
        res = PADRegressor(model_path=None).fit(X, y, use_pca=False,
                                                cv_folds=5, verbose=False)
        self.assertGreater(res["cv_metrics"]["P"]["ccc"], 0.5)

    def test_negative_control_no_leakage(self):
        """**阴性对照**：标签与特征完全无关时，CV ρc 必须落在噪声水平

        这条测试守的不是模型精度，而是管线的可信度：如果 K 折切分、
        标准化或 PCA 把训练信息漏进了验证集，ρc 会跑出显著正值，
        届时所有"效果好"的结论都不成立。阈值放宽到 0.35，
        只拦截"明显学出来了"这种程度的泄漏，避免随机波动导致误报。
        """
        X, y = self._make(learnable=False)
        res = PADRegressor(model_path=None).fit(X, y, use_pca=False,
                                                cv_folds=5, verbose=False)
        for axis in ("P", "A", "D"):
            ccc = res["cv_metrics"][axis]["ccc"]
            self.assertLess(ccc, 0.35,
                            f"{axis} 轴在随机标签下 ρc={ccc:.3f}，疑似数据泄漏")
        # 反向确认：随机标签下 ρ 也应接近 0（给 ρc 的结论加一道旁证）
        for axis in ("P", "A", "D"):
            rho = abs(res["cv_metrics"][axis]["rho"])
            self.assertLess(rho, 0.35)

    def test_confidence_tracks_cv_not_in_sample(self):
        """predict 的 confidence 必须来自 CV，不得用样本内值虚高"""
        X, y = self._make()
        reg = PADRegressor(model_path=None)
        res = reg.fit(X, y, use_pca=False, cv_folds=5, verbose=False)
        pred = reg.predict({nm: 1.0 for nm in PAD_FEATURE_NAMES})
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred.confidence,
                               round(res["mean_ccc"], 3), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
