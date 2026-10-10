# -*- coding: utf-8 -*-
"""HRV 特征提取单元测试（对应计划书 §4.2 算法子模块1）"""
import random
import unittest

from hrv_companion_agent.algorithms.hrv_extractor import HRVExtractor


class TestHRVExtractor(unittest.TestCase):

    def setUp(self):
        self.ex = HRVExtractor()

    def test_clean_ibi_filters_artifacts(self):
        ibi = [800, 850, 100, 5000, 820, 830, 840]
        cleaned = self.ex.clean_ibi(ibi)
        self.assertTrue(all(300 <= v <= 2000 for v in cleaned))
        self.assertNotIn(100, cleaned)
        self.assertNotIn(5000, cleaned)

    def test_time_domain_constant_ibi(self):
        ibi = [850.0] * 30
        feats = self.ex.time_domain(ibi)
        self.assertAlmostEqual(feats["sdnn_ms"], 0.0, places=2)
        self.assertAlmostEqual(feats["rmssd_ms"], 0.0, places=2)
        self.assertAlmostEqual(feats["pnn50_pct"], 0.0, places=2)
        self.assertAlmostEqual(feats["mean_hr_bpm"], 60000 / 850, places=1)

    def test_time_domain_alternating_ibi(self):
        ibi = [700.0, 800.0] * 15  # 差值恒为100ms
        feats = self.ex.time_domain(ibi)
        self.assertAlmostEqual(feats["rmssd_ms"], 100.0, places=2)
        self.assertAlmostEqual(feats["pnn50_pct"], 100.0, places=2)

    def test_insufficient_data(self):
        feats = self.ex.time_domain([800.0])
        self.assertEqual(feats["sdnn_ms"], 0.0)
        self.assertEqual(feats["n_beats"], 1)

    def test_frequency_domain_positive(self):
        random.seed(1)
        ibi = [850 + random.gauss(0, 25) for _ in range(60)]
        fd = self.ex.frequency_domain(ibi)
        self.assertGreaterEqual(fd["lf_ms2"], 0)
        self.assertGreaterEqual(fd["hf_ms2"], 0)

    def test_extract_from_ibi_keys(self):
        random.seed(2)
        ibi = [800 + random.gauss(0, 20) for _ in range(40)]
        feats = self.ex.extract_from_ibi(ibi)
        for key in ("hr_bpm", "sdnn", "rmssd", "pnn50", "lf_hf", "sd1_ms",
                    "sd2_ms", "sd_ratio", "ibi_list"):
            self.assertIn(key, feats)

    def test_extract_from_ble_prefers_firmware_values(self):
        packet = {"hr": 92, "ibi": [650, 660, 655], "sdnn": 28.0,
                  "rmssd": 18.0, "pnn50": 4.0, "lf_hf": 3.9,
                  "rr": 21.0, "sig_quality": 0.9, "t": 1000.0}
        feats = self.ex.extract_from_ble(packet)
        self.assertEqual(feats["sdnn"], 28.0)
        self.assertEqual(feats["lf_hf"], 3.9)
        self.assertEqual(feats["rr_bpm"], 21.0)

    def test_extract_from_ble_fallback_from_hr(self):
        packet = {"hr": 75, "ibi": [], "sdnn": None, "rmssd": None}
        feats = self.ex.extract_from_ble(packet)
        # A lone HR is not evidence of heart-rate variability.
        self.assertEqual(feats["sdnn"], 0)
        self.assertEqual(feats["rmssd"], 0)
        self.assertEqual(feats.get('sd1_ms', 0), 0)


if __name__ == "__main__":
    unittest.main()
