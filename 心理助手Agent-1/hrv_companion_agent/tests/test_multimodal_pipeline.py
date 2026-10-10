"""Failure regressions using deterministic sensor fixtures, no hardware/network."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from hrv_companion_agent import config
from hrv_companion_agent.core.sensor_pipeline import SensorPipeline, RAW_FEATURE_NAMES
from hrv_companion_agent.algorithms.multimodal_emotion import MultimodalEmotionRecognizer
from hrv_companion_agent.core.system import HeartCompanionSystem
from hrv_companion_agent.core.label_store import LabelStore
FIRMWARE=Path(__file__).resolve().parents[3]/'硬件开发'/'esp32_firmware'
sys.path.insert(0,str(FIRMWARE))
from ble_receiver import FrameAssembler
from ble_transport import frames
from hrv import HRVAnalyzer


def packet(seq, boot='boot-a', segment=0):
    now=seq*1000
    ends=list(range(((now-1000)//800+1)*800,(now//800)*800+1,800))
    return {'schema':2,'device_id':'esp-test','boot_id':boot,'seq':seq,
            'device_ms':now,'segment_id':segment,'continuity':True,
            'ppg':True,'imu':True,'ads':True,'motion_known':True,'moving':False,
            'motion':0.0,'sig_quality':0.9,'pi':1.0,'sample_rate':100,'expected_rate':100,
            'errors':dict.fromkeys(('ppg','imu','ads','fifo','clock'),0),
            'ibi_new':[800 for _ in ends],'ibi':[800,800,800],
            'ibi_events':[[end//800,end,800] for end in ends],
            'eda_samples':[[t,1.5+0.002*math.sin(t/2000)] for t in range(now-950,now+1,50)]}


class TestStreamQuality(unittest.TestCase):
    def warm(self,n=65):
        pipeline=SensorPipeline()
        for i in range(1,n+1):flow=pipeline.process(packet(i))
        return pipeline,flow

    def test_clean_multichannel_window_passes(self):
        pipeline,flow=self.warm()
        self.assertTrue(flow['quality']['quality_ok'],flow)
        self.assertEqual(len(pipeline.beats),75)
        self.assertTrue(all(k in flow['features'] for k in RAW_FEATURE_NAMES))

    def test_equal_intervals_are_distinct_beats(self):
        pipeline,_=self.warm()
        self.assertEqual(len(pipeline.beats),75)
        self.assertEqual(pipeline.last_beat_id,81)

    def test_duplicate_and_out_of_order_do_not_accumulate(self):
        pipeline,_=self.warm();before=len(pipeline.beats)
        self.assertFalse(pipeline.process(packet(65))['accepted'])
        self.assertFalse(pipeline.process(packet(64))['accepted'])
        self.assertEqual(len(pipeline.beats),before)

    def test_missing_packet_clears_window(self):
        pipeline,_=self.warm();f=pipeline.process(packet(67))
        self.assertFalse(f['quality']['quality_ok'])
        self.assertEqual(len(pipeline.beats),0)

    def test_error_unknown_motion_and_bad_rate_block(self):
        for kind in ('ppg','imu','fifo','clock','unknown','moving','rate','lowq'):
            pipeline,_=self.warm();bad=packet(66)
            if kind in bad['errors']:bad['errors'][kind]=1
            elif kind=='unknown':bad['motion_known']=False
            elif kind=='moving':bad['moving']=True
            elif kind=='rate':bad['sample_rate']=70
            else:bad['sig_quality']=None
            flow=pipeline.process(bad)
            self.assertFalse(flow['quality']['quality_ok'],kind)
            self.assertEqual(len(pipeline.beats),0)

    def test_recovery_does_not_get_stuck_at_boundary(self):
        pipeline,_=self.warm();bad=packet(66);bad['errors']['ppg']=1;pipeline.process(bad)
        for seq in range(67,132):flow=pipeline.process(packet(seq))
        self.assertTrue(flow['quality']['quality_ok'],flow)

    def test_boot_resets_and_old_session_is_rejected(self):
        pipeline,_=self.warm();pipeline.process(packet(1,boot='boot-b'))
        self.assertLess(len(pipeline.beats),5)
        self.assertFalse(pipeline.process(packet(66))['accepted'])

    def test_time_or_event_identity_gap_is_not_spliced(self):
        for change in ('id','time'):
            pipeline,_=self.warm();bad=packet(66)
            bad['ibi_events'][0][0 if change=='id' else 1]+=10 if change=='id' else 200
            flow=pipeline.process(bad)
            self.assertFalse(flow['quality']['quality_ok'])
            self.assertEqual(len(pipeline.beats),0)

    def test_gsr_noise_blocks_eda(self):
        pipeline,_=self.warm()
        for seq in range(66,91):
            bad=packet(seq);bad['eda_samples']=[[t,1.0 if i%2 else 2.0] for i,(t,v) in enumerate(bad['eda_samples'])]
            flow=pipeline.process(bad)
        self.assertTrue(flow['quality']['ppg_ok'])
        self.assertFalse(flow['quality']['eda_ok'])

    def test_gsr_fault_is_visible_even_when_ppg_rejects(self):
        pipeline,_=self.warm();bad=packet(66)
        bad['sig_quality']=.2
        bad['eda_samples'][5][1]=.0005
        flow=pipeline.process(bad)
        channels=flow['quality']['channels']
        self.assertFalse(channels['ppg']['packet_ok'])
        self.assertFalse(channels['gsr']['packet_ok'])
        self.assertIn('GSR异常电压或削顶',channels['gsr']['reasons'])
        self.assertEqual(flow['features'],{})
        self.assertFalse(flow['quality']['quality_ok'])
        self.assertEqual(len(pipeline.eda),0)

    def test_packet_diagnostics_do_not_qualify_a_window(self):
        flow=SensorPipeline().process(packet(1))
        self.assertTrue(flow['quality']['channels']['gsr']['packet_ok'])
        self.assertFalse(flow['quality']['channels']['qualified_window'])
        self.assertFalse(flow['quality']['quality_ok'])

    def test_invalid_gsr_is_reported_without_crashing_ppg_rejection(self):
        for sample in ([1000,float('nan')],['bad',1.5],None):
            bad=packet(1);bad['sig_quality']=.1;bad['eda_samples']=[sample]
            flow=SensorPipeline().process(bad)
            self.assertFalse(flow['quality']['channels']['gsr']['packet_ok'])

    def test_nan_and_old_schema_cannot_pass(self):
        for bad in ({'hr':72,'sig_quality':1},packet(1)):
            if bad.get('schema'):bad['ibi_events'][0][2]=float('nan')
            self.assertFalse(SensorPipeline().process(bad)['quality']['quality_ok'])


class TestBLEFrames(unittest.TestCase):
    def test_utf8_mtu23_roundtrip_out_of_order(self):
        raw=packet(40);raw['note']='中文标签';pieces=frames(json.dumps(raw,ensure_ascii=False),65535)
        self.assertTrue(all(len(p)<=20 for p in pieces));receiver=FrameAssembler();result=None
        for piece in reversed(pieces):result=receiver.add(piece,now=0) or result
        self.assertEqual(result,raw)

    def test_missing_frame_timeout_then_next_message(self):
        receiver=FrameAssembler();pieces=frames(json.dumps(packet(1)),2)
        for p in pieces[:-1]:self.assertIsNone(receiver.add(p,now=1))
        self.assertIsNone(receiver.add(pieces[-1],now=5));result=None
        for p in frames('{"ok":true}',3):result=receiver.add(p,now=6) or result
        self.assertEqual(result,{'ok':True})

    def test_corruption_is_rejected(self):
        receiver=FrameAssembler();self.assertIsNone(receiver.add(frames('not-json',1)[0]))
        pieces=frames(json.dumps(packet(1)),2);receiver.add(pieces[0])
        self.assertIsNone(receiver.add(pieces[0][:-1]+b'x'));self.assertGreater(receiver.dropped,0)


class TestSafeInference(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.saved=config.MEMORY_DB_PATH
        config.MEMORY_DB_PATH=Path(self.temp.name)/'memory.db'
        self.system=HeartCompanionSystem(user_id='test-sensor')
    def tearDown(self):
        self.system.close();config.MEMORY_DB_PATH=self.saved;self.temp.cleanup()
    def warm(self,n=65):
        for i in range(1,n+1):state=self.system.ingest_sensor(packet(i))
        return state

    def test_no_model_abstains_without_polluting_memory(self):
        state=self.warm();self.assertTrue(state['quality']['quality_ok'])
        self.assertEqual(state['emotion'],'不确定');self.assertIsNone(state['risk_score'])
        self.assertFalse(self.system.check_trigger()['should_trigger']);self.assertEqual(self.system.get_timeline(1),[])

    def test_channel_diagnostics_reach_public_state(self):
        bad=packet(1);bad['sig_quality']=.1;bad['eda_samples'][0][1]=.0005
        state=self.system.ingest_sensor(bad)
        self.assertFalse(state['quality']['channels']['gsr']['packet_ok'])
        self.assertEqual(state['emotion'],'不确定')
        self.assertFalse(state['validated'])

    def test_stale_state_blocks_trigger(self):
        self.warm();self.system._last_sensor_received-=10
        self.assertEqual(self.system.get_state()['layer'],'stale')
        self.assertFalse(self.system.check_trigger()['should_trigger'])

    def test_empty_increment_never_uses_recent_ibi(self):
        self.warm();raw=packet(66);raw['ibi_new']=[];raw['ibi_events']=[]
        before=len(self.system.sensor_pipeline.beats);self.system.ingest_sensor(raw)
        self.assertLessEqual(len(self.system.sensor_pipeline.beats),before)

    def test_label_requires_quality_and_nonoverlap(self):
        store=LabelStore(Path(self.temp.name)/'labels.jsonl')
        with self.assertRaises(ValueError):store.append(self.system,'平静')
        self.warm()
        with self.assertRaises(ValueError):store.append(self.system,'平静')
        self.system.calibrate_sensor('confirm_contact')
        for i in range(66,251):self.system.ingest_sensor(packet(i))
        self.system.calibrate_sensor('capture_baseline')
        for i in range(251,316):self.system.ingest_sensor(packet(i))
        self.assertTrue(store.append(self.system,'平静')['saved'])
        with self.assertRaises(ValueError):store.append(self.system,'愉悦')
        for i in range(316,381):self.system.ingest_sensor(packet(i))
        self.assertTrue(store.append(self.system,'愉悦')['saved'])

    def test_legacy_bundle_rejected(self):
        with patch('joblib.load',return_value={'model':object(),'schema':'old'}):rec=MultimodalEmotionRecognizer('test-path')
        self.assertFalse(rec.available)


class TestHRVContinuity(unittest.TestCase):
    def test_motion_clears_old_intervals(self):
        hrv=HRVAnalyzer();hrv.rr=[800,810,790];hrv.rr_t=[1,2,3];hrv.pending_rr=[790]
        old=hrv.segment_id;hrv.add_sample(60000,t_ms=1000,motion=True)
        self.assertEqual(hrv.rr,[]);self.assertGreater(hrv.segment_id,old)

    def test_timestamp_gap_clears_window(self):
        hrv=HRVAnalyzer();hrv.add_sample(60000,t_ms=0);hrv.rr=[800,810];hrv.rr_t=[1,2]
        hrv.add_sample(60000,t_ms=1000);self.assertEqual(hrv.rr,[])
