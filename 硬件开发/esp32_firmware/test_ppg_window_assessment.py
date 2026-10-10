import unittest
from assess_ppg_windows import assess


def metric(t, q=.6, pi=.4, nn=40, segment=0):
    return {'time_ms': t, 'sig_quality': q, 'pi': pi, 'nn': nn, 'segment_id': segment}


class WindowAssessmentTests(unittest.TestCase):
    def test_independent_maxima_do_not_produce_a_passing_report(self):
        result = assess({'metrics': [metric(1000, q=.9, pi=.1), metric(2000, q=.3, pi=1)],
                         'break_times_ms': []})
        self.assertEqual(result['simultaneously_passing_reports'], 0)

    def test_timestamp_gap_and_internal_break_split_runs(self):
        result = assess({'metrics': [metric(t) for t in (1000, 2000, 4000, 5000, 6000, 7000)],
                         'break_times_ms': [5500]})
        self.assertEqual([run['reports'] for run in result['runs']], [2, 2, 2])
        self.assertEqual(result['longest_passing_report_span_s'], 1)

    def test_nn_and_segment_changes_prevent_stitching(self):
        result = assess({'metrics': [metric(1000), metric(2000, nn=29), metric(3000),
                                    metric(4000, segment=1)], 'break_times_ms': []})
        self.assertEqual([run['reports'] for run in result['runs']], [1, 1, 1])
        self.assertFalse(result['qualified_physiological_window'])


if __name__ == '__main__':
    unittest.main()
