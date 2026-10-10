"""New-channel regressions. Fixtures exercise behavior, never emotion accuracy."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrv_companion_agent import config
from hrv_companion_agent.algorithms.multimodal_emotion import CLASSES, MultimodalEmotionRecognizer, deployment_failures
from hrv_companion_agent.core.sensor_pipeline import FEATURE_NAMES, RAW_FEATURE_NAMES, FEATURE_SCHEMA, SensorPipeline
from hrv_companion_agent.core.system import HeartCompanionSystem
from hrv_companion_agent.core.label_store import LabelStore
from hrv_companion_agent.agent.tools.pad_tool import PADTool
from test_multimodal_pipeline import packet


class ProbabilityModel:
    classes_ = CLASSES
    probabilities = [.02, .04, .90, .02, .02]
    def predict_proba(self, values):
        return [self.probabilities]


def test_bundle():
    # Engineered metadata for contract tests, never a saved/deployed model.
    evaluation = {'group_holdout': True, 'calibration_grouped': True,
                  'train_subjects': ['train-%d'%i for i in range(16)],
                  'test_subjects': ['test-%d'%i for i in range(4)],
                  'test_class_counts': dict.fromkeys(CLASSES, 10),
                  'balanced_accuracy': .90, 'macro_f1': .88, 'min_class_recall': .80,
                  'coverage': .80, 'accepted_accuracy': .95}
    return {'schema': FEATURE_SCHEMA, 'feature_names': FEATURE_NAMES,
            'classes': CLASSES, 'label_source': 'independent_self_report',
            'evaluation': evaluation, 'model': ProbabilityModel(),
            'confidence_threshold': .70,
            'feature_bounds': {k: [-1000, 1000] for k in FEATURE_NAMES}}


class TestInferencePolicy(unittest.TestCase):
    def recognizer(self, bundle=None):
        with patch('joblib.load', return_value=bundle or test_bundle()):
            return MultimodalEmotionRecognizer('temporary-test-model')

    def test_unqualified_evaluation_is_not_activated(self):
        for field in ('balanced_accuracy', 'macro_f1', 'min_class_recall', 'coverage', 'accepted_accuracy'):
            bundle = test_bundle(); bundle['evaluation'][field] = .1
            self.assertFalse(self.recognizer(bundle).available, field)
        bundle = test_bundle(); bundle['evaluation']['test_subjects'][0] = 'train-0'
        self.assertFalse(self.recognizer(bundle).available)

    def test_missing_actual_class_or_bounds_is_rejected(self):
        bundle = test_bundle(); bundle['model'].classes_ = CLASSES[:-1]
        self.assertFalse(self.recognizer(bundle).available)
        bundle = test_bundle(); bundle.pop('feature_bounds')
        self.assertFalse(self.recognizer(bundle).available)

    def test_confirmation_uses_new_sequence_and_monotonic_device_time(self):
        rec = self.recognizer(); feats = dict.fromkeys(FEATURE_NAMES, 0.)
        self.assertEqual(rec.recognize(feats, 1, 1, 'boot').layer, 'confirming')
        # Repeating a packet cannot satisfy the confirmation interval.
        self.assertFalse(rec.recognize(feats, 100, 1, 'boot').detail['validated'])
        rec.reset()
        for n in range(1, 12): result = rec.recognize(feats, n, n, 'boot')
        self.assertTrue(result.detail['validated'])
        self.assertEqual(result.emotion, '高焦虑')
        self.assertFalse(rec.recognize(feats, 12, 12, 'new-boot').detail['validated'])

    def test_probability_overlap_invalid_and_ood_abstain(self):
        feats = dict.fromkeys(FEATURE_NAMES, 0.)
        for probs in ([.21, .20, .20, .20, .19], [float('nan'), .1, .8, .1, 0], [1, 1, 1, 1, 1]):
            bundle = test_bundle(); bundle['model'].probabilities = probs
            rec = self.recognizer(bundle)
            self.assertFalse(rec.recognize(feats, 1, 1).detail['validated'])
        rec = self.recognizer(); feats['gsr_relative'] = 1001
        self.assertIn('训练数据范围', rec.recognize(feats, 1, 1).detail['reason'])


class TestNewChannelQuality(unittest.TestCase):
    def warm(self):
        pipeline = SensorPipeline()
        for n in range(1, 66): flow = pipeline.process(packet(n))
        return pipeline, flow

    def test_missing_pi_bad_motion_or_segment_cannot_pass(self):
        for field, value in [('pi', None), ('pi', .1), ('motion', float('nan')),
                             ('motion', -.1), ('segment_id', True)]:
            pipeline, _ = self.warm(); raw = packet(66); raw[field] = value
            self.assertFalse(pipeline.process(raw)['quality']['quality_ok'], field)

    def test_delayed_gsr_tail_and_long_sampling_gap_block(self):
        for case in ('tail', 'gap'):
            pipeline, _ = self.warm(); raw = packet(66)
            if case == 'tail': raw['eda_samples'] = raw['eda_samples'][:5]
            else: raw['eda_samples'] = raw['eda_samples'][5:]
            self.assertFalse(pipeline.process(raw)['quality']['eda_ok'], case)

    def test_short_window_does_not_equal_model_window(self):
        pipeline = SensorPipeline()
        for n in range(1, 41): flow = pipeline.process(packet(n))
        self.assertFalse(flow['quality']['quality_ok'])

    def test_malformed_crossing_rr_is_checked_before_discard(self):
        raw = packet(1); raw['ibi_events'][0][2] = 50000
        self.assertFalse(SensorPipeline().process(raw)['quality']['quality_ok'])


class TestSystemRevision(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.old = config.MEMORY_DB_PATH
        config.MEMORY_DB_PATH = Path(self.temp.name)/'memory.db'
        self.system = HeartCompanionSystem(user_id='revision-test')
    def tearDown(self):
        self.system.close(); config.MEMORY_DB_PATH = self.old; self.temp.cleanup()
    def feed(self, start, end):
        for n in range(start, end+1): state = self.system.ingest_sensor(packet(n))
        return state
    def prepare(self):
        self.feed(1, 1); self.system.calibrate_sensor('confirm_contact')
        self.feed(2, 185); self.system.calibrate_sensor('capture_baseline')
        return self.feed(186, 250)

    def test_contact_settle_and_rest_window_are_required(self):
        state = self.feed(1, 65)
        self.assertEqual(state['layer'], 'calibration_required')
        with self.assertRaises(ValueError): self.system.calibrate_sensor('capture_baseline')
        self.system.calibrate_sensor('confirm_contact'); self.feed(66, 200)
        with self.assertRaises(ValueError): self.system.calibrate_sensor('capture_baseline')
        self.feed(201, 250); self.system.calibrate_sensor('capture_baseline')
        state = self.feed(251, 315)
        self.assertTrue(all(k in state['features'] for k in FEATURE_NAMES))
        self.assertEqual(state['layer'], 'model_unavailable')

    def test_invalid_packet_revokes_existing_result_but_duplicate_does_not(self):
        self.prepare()
        self.system.multimodal = TestInferencePolicy().recognizer()
        state = self.feed(251, 262)
        self.assertTrue(state['validated'])
        self.assertTrue(self.system.ingest_sensor(packet(262))['validated'])
        bad = packet(263); bad['segment_id'] = None
        state = self.system.ingest_sensor(bad)
        self.assertFalse(state['validated']); self.assertIsNone(state['risk_score'])

    def test_stale_quality_affect_and_pad_do_not_bypass_gate(self):
        self.prepare(); self.system._last_sensor_received -= 10
        self.assertFalse(self.system.quality_report()['quality_ok'])
        self.assertIsNone(self.system.get_affect())
        self.assertFalse(self.system.get_state()['calibration']['baseline_ready'])
        tool = PADTool(state_provider=self.system.get_state)
        self.assertEqual(tool._resolve_pad({'simulation':False, 'feats':dict.fromkeys(RAW_FEATURE_NAMES,1)}), {})

    def test_label_data_roundtrip_and_purge(self):
        self.prepare(); store = LabelStore(Path(self.temp.name)/'labels.jsonl')
        store.append(self.system, '平静')
        from hrv_companion_agent.models.train_multimodal import load_rows
        rows = load_rows(store.path)
        self.assertEqual(len(rows), 1); self.assertTrue(rows[0]['calibration']['contact_confirmed'])
        self.assertEqual(store.purge('other'), 0)
        self.assertEqual(len(store.export(self.system.user_id)), 1)
        self.assertEqual(store.purge(self.system.user_id), 1)

    def test_boot_change_manual_and_replay_cannot_produce_labels(self):
        self.prepare(); self.system.ingest_sensor(packet(1, boot='new-boot'))
        self.assertFalse(self.system.get_state()['calibration']['baseline_ready'])
        self.system.set_manual_emotion('平静')
        with self.assertRaises(ValueError): self.system.calibrate_sensor('confirm_contact')
        store = LabelStore(Path(self.temp.name)/'labels.jsonl')
        with self.assertRaises(ValueError): store.append(self.system, '平静')

    def test_unknown_risk_cannot_be_implicit_success_feedback(self):
        self.prepare(); self.system._pending_strategy = 'active_dialogue'
        self.system._pending_before_risk = 80
        self.assertFalse(self.system.feedback()['recorded'])

    def test_unknown_and_simulated_predictions_are_not_agreement_samples(self):
        self.prepare()
        self.system.memory.save_self_report(5, emotion='平静', objective=self.system.get_state())
        self.system.memory.save_self_report(5, emotion='平静', objective={'simulation':True,'emotion':'高焦虑','risk_score':90})
        stats=self.system.memory.self_report_consistency()
        self.assertEqual(stats['n'],2)
        self.assertEqual(stats['n_comparable'],0)
        self.assertEqual(stats['n_conflict'],0)

    def test_trigger_poll_does_not_increment_same_packet(self):
        self.prepare(); self.system.multimodal = TestInferencePolicy().recognizer()
        self.feed(251,262)
        self.system._last_interaction_time = __import__('time').time()
        self.system.check_trigger()
        self.assertIn('本包已检查', self.system.check_trigger()['reason'])


class TestDeviceDrivers(unittest.TestCase):
    def test_ads_single_shot_signed_once_timeout_and_short_read(self):
        from ads1115 import ADS1115
        class Bus:
            config = None
            short = False
            def writeto_mem(self, addr, reg, data): self.config = data
            def readfrom_mem(self, addr, reg, count):
                if self.short:return b'\x00'
                return self.config if reg == 1 else b'\xff\xfe'
        bus = Bus(); adc = ADS1115(bus)
        with patch('time.ticks_diff', side_effect=lambda a,b:a-b, create=True):
            self.assertTrue(adc.start(0)); self.assertFalse(adc.start(1))
            code, value = adc.poll(10); self.assertEqual(code,-2)
            self.assertIsNone(adc.poll(11))
            adc.start(20)
            with self.assertRaises(OSError): adc.poll(200)
            self.assertFalse(adc.pending)
            adc.start(201); bus.short=True
            with self.assertRaises(OSError): adc.poll(210)

    def test_imu_spike_is_not_smoothed_into_stationary(self):
        from mpu6050 import MPU6050
        imu = MPU6050(None)
        imu.motion_deviations = lambda *_: (0, 25, 0)
        self.assertTrue(imu.is_motion()[0])

    def test_uart_buffer_limit_partial_write_and_error(self):
        from serial_transport import USBTransport
        class UART:
            mode = 'ok'
            def write(self, data):
                if self.mode == 'error':raise OSError('disconnected')
                return len(data) if self.mode == 'ok' else len(data)-1
        uart = UART()
        transport = USBTransport(64, stream=uart)
        self.assertTrue(transport.send('{"ok":true}'))
        while transport.pending: transport.service()
        self.assertFalse(transport.send('x'*64))
        uart.mode='partial'; self.assertTrue(transport.send('{}'))
        transport.service()
        self.assertTrue(transport.pending)
        uart.mode='ok'; transport.service()
        uart.mode='error'; self.assertTrue(transport.send('{}'))
        transport.service()
        self.assertEqual(transport.pending,'\n')
        uart.mode='ok'; transport.service()
        self.assertTrue(transport.send('{}'))
        self.assertEqual(transport.dropped,2)
