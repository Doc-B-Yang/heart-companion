"""Artificial fixtures verify isolation/contracts; never real emotion accuracy."""
import copy
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from hrv_companion_agent.models.experimental_arousal import prepare, split_days, train, predict, PPG, GSR
from hrv_companion_agent.tests.test_training_revision import fixture
from hrv_companion_agent.core.label_store import LabelStore


def rated(day=1, index=0, score=2):
    # Invalid ratings must reach prepare(), rather than fail in this fixture.
    label = '平静' if type(score) is int and score < 5 else '高焦虑'
    row = fixture(index=index, label=label)
    row['session_id'] = 'fixture-day-%d' % day
    row['timestamp'] = datetime(2026, 10, day, 12, tzinfo=timezone(timedelta(hours=8))).timestamp() + index * 60
    row['collection_day'] = '2026-10-%02d' % day
    row['self_report'] = {'arousal_1_9': score, 'valence_1_9': None,
                          'scope': 'preceding_60_second_window'}
    return row


class TestExperimentalArousal(unittest.TestCase):
    def test_independent_rating_and_middle_state(self):
        missing = rated(index=0); missing.pop('self_report')
        middle = rated(index=1, score=5)
        high_but_calm = rated(index=2, score=8); high_but_calm['label'] = '平静'
        rows, excluded = prepare([missing, middle, high_but_calm])
        self.assertEqual([r['arousal_target'] for r in rows], [1])
        self.assertEqual(excluded, {'neutral_excluded': 1, 'unrated_excluded': 1})
        for invalid in (True, '8', 10):
            with self.assertRaises(ValueError): prepare([rated(score=invalid)])

    def test_false_day_subject_and_overlap_are_rejected(self):
        row = rated(); row['collection_day'] = '2026-10-02'
        with self.assertRaises(ValueError): prepare([row])
        row = rated(day=2); row['subject_id'] = 'someone-else'
        with self.assertRaises(ValueError): prepare([rated(), row])
        with self.assertRaises(ValueError): prepare([rated(), rated()])

    def test_day_holdout_scaler_and_prediction(self):
        import joblib
        rows = [rated(day, index=n, score=2 if n < 6 else 8) for day in (1, 2) for n in range(12)]
        selected, _ = prepare(rows)
        tr, te = split_days(selected, '2026-10-02')
        self.assertFalse({r['collection_day'] for r in tr} & {r['collection_day'] for r in te})
        with self.assertRaises(ValueError): split_days(selected, '2026-10-01')
        # Shift only held-out observations: fitted scaler must remain train mean.
        for row in rows[12:]: row['features']['gsr_slope_v_s'] = .1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.joblib'
            report = train(rows, path, '2026-10-02')
            bundle = joblib.load(path)
            names = PPG + GSR
            expected = sum(r['features']['gsr_slope_v_s'] for r in rows[:12]) / 12
            self.assertAlmostEqual(bundle['model'][0].mean_[names.index('gsr_slope_v_s')], expected)
            self.assertFalse(report['production_ready'])
            self.assertEqual(set(report['metrics']), {'ppg', 'gsr', 'fusion'})
            result = predict(bundle, rated(index=3))
            self.assertIsNone(result['emotion'])
            self.assertFalse(result['probability_calibrated'])
            wrong_person = rated(); wrong_person['subject_id'] = 'other'
            with self.assertRaises(ValueError): predict(bundle, wrong_person)
            with self.assertRaises(ValueError): train(rows, path, '2026-10-02')

    def test_small_dataset_and_rating_input_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError): train([rated()], Path(directory)/'small.joblib', '2026-10-01')
        for invalid in (True, 0, 10, '4'):
            with self.assertRaises(ValueError): LabelStore().append(None, '平静', arousal_score=invalid)


if __name__ == '__main__':
    unittest.main()
