import copy
import unittest
from analyse_ble_pacing import analyse


class PacingAnalysisTests(unittest.TestCase):
    def evidence(self):
        packets = [{'schema': 2, 'device_id': 'board', 'boot_id': 'session', 'seq': i,
                    'device_ms': i * 1000, 'sample_rate': 100,
                    'transport': {'usb_dropped': 0},
                    'errors': {key: 0 for key in ('ppg', 'fifo', 'imu', 'ads', 'clock')}}
                   for i in range(1, 30)]
        board = packets + [{'type': 'ble_stage_result', 'stage': 0, 'start_ms': 0, 'end_ms': 30000,
            'connected_service_calls': 100, 'service_calls': 100, 'errors': 0, 'dropped': 0},
            {'type': 'complete'}, {'type': 'host_result', 'complete': True, 'reset_sent': True,
                                  'startup_seen': True, 'production_packet_seen': True}]
        receiver = [{'type': 'received', 'packet': copy.deepcopy(p)} for p in packets]
        receiver += [{'type': 'receiver_result'}]
        return board, receiver

    def test_complete_matching_connected_stage_passes(self):
        result = analyse(*self.evidence())
        self.assertTrue(result['stages'][0]['short_transport_check_passed'])
        self.assertFalse(result['accuracy_validated'])
        self.assertFalse(result['classification_allowed'])

    def test_disconnected_zero_errors_cannot_pass(self):
        board, receiver = self.evidence()
        board[-3]['connected_service_calls'] = 0
        self.assertFalse(analyse(board, receiver)['stages'][0]['short_transport_check_passed'])

    def test_missing_or_changed_actual_field_rejects_stage(self):
        board, receiver = self.evidence()
        receiver[0]['packet']['sample_rate'] = 90
        del receiver[1]
        result = analyse(board, receiver)['stages'][0]
        self.assertFalse(result['short_transport_check_passed'])
        self.assertEqual(result['missing_ble_sequences'], [2])
        self.assertEqual(result['payload_mismatches'], [{'seq': 1, 'fields': ['sample_rate']}])

    def test_unrestored_or_incomplete_capture_rejects_stage(self):
        for field in ('complete', 'production_packet_seen'):
            board, receiver = self.evidence()
            board[-1][field] = False
            self.assertFalse(analyse(board, receiver)['stages'][0]['short_transport_check_passed'])


if __name__ == '__main__':
    unittest.main()
