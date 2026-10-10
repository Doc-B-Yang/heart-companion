"""Diagnostic safety checks with synthetic fixtures; no human validation."""
import json
import tempfile
import unittest
from pathlib import Path

from analyse_raw_diagnostic import analyse, split_ppg, eda_summary


class DiagnosticAnalysisTests(unittest.TestCase):
    def test_missing_sample_motion_and_fifo_breaks_are_never_joined(self):
        rows = [[0, 10, 50, 60, 0], [0, 20, 50, 60, 0],
                [0, 40, 50, 60, 0], [0, 50, 50, 60, 1],
                [0, 60, 50, 60, 0], [1, 70, 50, 60, 0]]
        self.assertEqual([list(p[:, 1]) for p in split_ppg(rows)],
                         [[10, 20], [40], [60], [70]])

    def test_partial_capture_and_packet_loss_remain_visible(self):
        rows = [{'type': 'raw', 'stage': 'fixture', 'seq': seq,
                 'ppg': [], 'eda': [], 'imu': []} for seq in (1, 3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            result = analyse(path)
        self.assertFalse(result['capture_complete'])
        self.assertFalse(result['classification_allowed'])
        self.assertFalse(result['accuracy_validated'])
        self.assertFalse(result['stages']['fixture']['packet_sequence_contiguous'])

    def test_eda_gap_disables_spectral_interpolation(self):
        rows = [[i * 50 + (500 if i >= 10 else 0), 100, .0125] for i in range(30)]
        result = eda_summary(rows)
        self.assertIn('gaps', result['analysis'])
        self.assertNotIn('dominant_raw_hz', result)

    def test_negative_time_never_forms_continuous_ppg(self):
        self.assertEqual(len(split_ppg([[0, 20, 5, 6, 0], [0, 10, 5, 6, 0]])), 2)


if __name__ == '__main__':
    unittest.main()
