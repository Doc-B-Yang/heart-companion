"""Synthetic audit refusal checks; these are not hardware accuracy evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from replay_raw_pipeline import evaluate


class ReplayRefusalTests(unittest.TestCase):
    def fixture(self, folder, change=None):
        source = Path(folder) / 'fixture.jsonl'
        rows = [dict(type='raw', stage='s', seq=1, fifo_events=[],
                     ppg=[[0,t,50000,50000,0] for t in range(10, 1010, 10)],
                     eda=[[t,10000,1.25] for t in range(50,1050,50)],
                     imu=[[1000,0,0,0,0]]),
                dict(type='result', stage='s', errors=dict(ppg=0,fifo=0,imu=0,ads=0,clock=0),
                     fifo_recovery_errors=0,usb_errors=0), dict(type='complete')]
        if change:
            change(rows)
        source.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
        metric = dict(time_ms=1000, segment_id=0, sig_quality=.9, pi=1, ibi_events=[])
        source.with_suffix('.polarity.json').write_text(json.dumps(
            dict(stages=dict(s=[dict(negative=dict(metrics=[metric]))]))), encoding='utf-8')
        return source

    def test_incomplete_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                evaluate(self.fixture(folder, lambda r: r.pop()))

    def test_unlocalized_adc_error_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Unaccounted'):
                evaluate(self.fixture(folder, lambda r: r[1]['errors'].update(ads=1)))

    def test_raw_sequence_gap_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'capture gaps'):
                evaluate(self.fixture(folder, lambda r: r[0].update(seq=2)))

    def test_motion_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Motion'):
                evaluate(self.fixture(folder, lambda r: r[0]['ppg'][0].__setitem__(4,1)))

    def test_short_high_quality_is_not_a_window(self):
        with tempfile.TemporaryDirectory() as folder:
            audit = evaluate(self.fixture(folder))
            self.assertEqual(audit['stages']['s']['negative']['reconstructed_backend_passing_reports'], 0)
            self.assertFalse(audit['classification_allowed'])
            self.assertFalse(audit['accuracy_validated'])


if __name__ == '__main__':
    unittest.main()
