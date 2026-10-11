"""Real API contract via ASGI transport; storage stays in temporary directories."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from hrv_companion_agent import config
from hrv_companion_agent.backend.main import app
from hrv_companion_agent.backend import routes
from hrv_companion_agent.core.label_store import LabelStore, SelfReportStore
from test_multimodal_pipeline import packet


class TestWebContract(unittest.TestCase):
    def test_private_network_preflight_only_for_allowed_origins(self):
        headers={'Origin':'https://doc-b-yang.github.io',
                 'Access-Control-Request-Method':'POST',
                 'Access-Control-Request-Headers':'content-type',
                 'Access-Control-Request-Private-Network':'true'}
        response=self.client.options('/api/dataset/self-report',headers=headers)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.headers.get('access-control-allow-private-network'),'true')
        self.assertEqual(response.headers.get('access-control-allow-origin'),headers['Origin'])
        headers['Origin']='https://untrusted.example'
        response=self.client.options('/api/dataset/self-report',headers=headers)
        self.assertEqual(response.status_code,400)
        self.assertNotIn('access-control-allow-origin',response.headers)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.old=config.MEMORY_DB_PATH
        config.MEMORY_DB_PATH=Path(self.temp.name)/'memory.db'
        routes._labels=LabelStore(Path(self.temp.name)/'labels.jsonl')
        self.old_reports=routes._experiment_reports
        routes._experiment_reports=SelfReportStore(Path(self.temp.name)/'reports.jsonl')
        self.client=TestClient(app)
    def tearDown(self):
        for system in routes._systems.values():system.close()
        routes._systems.clear();self.client.close();config.MEMORY_DB_PATH=self.old
        routes._experiment_reports=self.old_reports;self.temp.cleanup()
    def post(self,raw):
        return self.client.post('/api/sensor/raw',json={'user_id':'api-test','packet':raw})

    def test_webpage_and_assets_exist(self):
        self.assertEqual(self.client.get('/').status_code,200)
        self.assertIn('连接蓝牙设备',self.client.get('/').text)
        for p in ('app.js','style.css','ble_frames.js'):
            self.assertEqual(self.client.get('/static/'+p).status_code,200)

    def test_subjective_report_without_qualified_sensor_window(self):
        payload={'user_id':'report-test','label':'疲劳','note':'测试文字',
                 'arousal_score':2,'valence_score':4}
        response=self.client.post('/api/dataset/self-report',json=payload)
        self.assertEqual(response.status_code,200,response.text)
        self.assertTrue(response.json()['saved'])
        self.assertFalse(response.json()['training_sample_saved'])
        self.assertEqual(routes._labels.export('report-test'),[])
        rows=routes._experiment_reports.export('report-test')
        self.assertEqual(rows[0]['note'],'测试文字')
        self.assertEqual(rows[0]['arousal_1_9'],2)
        self.assertEqual(rows[0]['record_kind'],'subjective_only')
        self.assertEqual(routes._experiment_reports.export('another-user'),[])
        self.assertEqual(routes._experiment_reports.purge('report-test'),1)
        self.assertEqual(routes._experiment_reports.export('report-test'),[])
        for invalid in ('未知类别',):
            self.assertEqual(self.client.post('/api/dataset/self-report',json={**payload,'label':invalid}).status_code,422)
        self.assertEqual(routes._experiment_reports.export('report-test'),[])

    def test_quality_model_abstention_and_duplicate_contract(self):
        for i in range(1,66):r=self.post(packet(i))
        body=r.json();self.assertTrue(body['state']['quality']['quality_ok'])
        self.assertIsNone(body['state']['risk_score']);self.assertEqual(body['state']['risk_band'],'未知')
        self.assertFalse(body['state']['validated'])
        self.assertFalse(self.post(packet(65)).json()['accepted'])
        bad=packet(66);bad['motion_known']=False
        self.assertFalse(self.post(bad).json()['state']['quality']['quality_ok'])

    def test_label_and_legacy_endpoint_fail_closed(self):
        r=self.client.post('/api/dataset/label',json={'user_id':'api-test','label':'平静'})
        self.assertEqual(r.status_code,409)
        self.post(packet(1))
        r=self.client.post('/api/sensor/calibration',json={'user_id':'api-test','action':'confirm_contact'})
        self.assertEqual(r.status_code,200)
        for i in range(2,186):self.post(packet(i))
        r=self.client.post('/api/sensor/calibration',json={'user_id':'api-test','action':'capture_baseline'})
        self.assertEqual(r.status_code,200,r.text)
        for i in range(186,251):self.post(packet(i))
        for invalid in (True, '8', 0, 10, 2.5):
            r=self.client.post('/api/dataset/label',json={'user_id':'api-test','label':'平静','arousal_score':invalid})
            self.assertEqual(r.status_code,422)
        r=self.client.post('/api/dataset/label',json={'user_id':'api-test','label':'平静',
                                                    'arousal_score':8,'valence_score':3})
        self.assertEqual(r.status_code,200)
        saved=routes._labels.export('api-test')
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]['self_report'], {'arousal_1_9':8,'valence_1_9':3,
                                                 'scope':'preceding_60_second_window'})
        self.assertEqual(saved[0]['label'],'平静')  # Emotion name cannot overwrite independent ratings.
        self.assertRegex(saved[0]['collection_day'], r'^\d{4}-\d{2}-\d{2}$')
        r=self.client.post('/api/dataset/label',json={'user_id':'api-test','label':'愉悦'})
        self.assertEqual(r.status_code,409)
        r=self.client.post('/api/sensor/data',json={'user_id':'old-test','hr':72})
        self.assertFalse(r.json()['accepted'])

    def test_simulated_frames_rejected_at_hardware_entry(self):
        raw=packet(1);raw['_simulation']=True
        self.assertEqual(self.post(raw).status_code,422)

    def test_chat_remains_available_without_sensor_classification(self):
        r=self.client.post('/api/chat/message',json={'user_id':'chat-test','message':'今天有些累'})
        self.assertEqual(r.status_code,200);self.assertTrue(r.json()['response'])
