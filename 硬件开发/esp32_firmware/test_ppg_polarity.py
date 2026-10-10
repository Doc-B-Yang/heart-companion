"""Synthetic detector invariants; these tests do not validate human HRV."""
import math
import unittest
from hrv import HRVAnalyzer


class PolarityTests(unittest.TestCase):
    def test_explicit_default_preserves_every_metric(self):
        a, b = HRVAnalyzer(), HRVAnalyzer(polarity=1)
        for i in range(3000):
            value = 30000 + 1000 * math.sin(2 * math.pi * i / 100)
            self.assertEqual(a.add_sample(value, i * 10), b.add_sample(value, i * 10))
            if i % 100 == 99:
                self.assertEqual(a.metrics(), b.metrics())

    def test_reversed_landmark_preserves_raw_dc_and_saturation(self):
        positive, negative = HRVAnalyzer(), HRVAnalyzer(polarity=-1)
        for i in range(7000):
            # Mirrored optical signals have the same known 1-second period.
            wave = 1000 * math.sin(2 * math.pi * i / 100)
            positive.add_sample(30000 + wave, i * 10)
            negative.add_sample(30000 - wave, i * 10)
            if i % 100 == 99:
                positive.metrics(); negative.metrics()
        self.assertGreater(len(positive.rr), 50)
        self.assertEqual(len(positive.rr), len(negative.rr))
        for a, b in zip(positive.rr, negative.rr):
            self.assertAlmostEqual(a, b, places=7)
        self.assertGreater(negative.dc, 28000)
        self.assertEqual(negative.sat_n, 0)

    def test_both_polarities_break_on_missing_samples_and_motion(self):
        for polarity in (-1, 1):
            analyzer = HRVAnalyzer(polarity=polarity)
            for i in range(3000):
                analyzer.add_sample(30000 + 1000 * math.sin(2 * math.pi * i / 100), i * 10)
                if i % 100 == 99:
                    analyzer.metrics()
            self.assertGreater(len(analyzer.rr), 20)
            old_segment = analyzer.segment_id
            analyzer.add_sample(30000, 31000)
            self.assertEqual(analyzer.segment_id, old_segment + 1)
            self.assertEqual(analyzer.rr, [])
            analyzer.add_sample(30000, 31010, motion=True)
            self.assertEqual(analyzer.rr, [])
            self.assertEqual(analyzer.pending_events, [])

    def test_constant_input_never_becomes_beats(self):
        for polarity in (-1, 1):
            analyzer = HRVAnalyzer(polarity=polarity)
            for i in range(6000):
                self.assertFalse(analyzer.add_sample(30000, i * 10))
            self.assertEqual(analyzer.metrics()['nn'], 0)

    def test_invalid_polarity_fails_explicitly(self):
        with self.assertRaises(ValueError):
            HRVAnalyzer(polarity=0)


if __name__ == '__main__':
    unittest.main()
