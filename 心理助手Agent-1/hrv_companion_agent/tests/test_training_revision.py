"""Training contract tests use artificial fixtures solely in temporary files."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from hrv_companion_agent.core.sensor_pipeline import RAW_FEATURE_NAMES, FEATURE_SCHEMA
from hrv_companion_agent.core.sensor_calibration import SensorCalibration
from hrv_companion_agent.models.train_multimodal import load_rows, validate_rows, train
from hrv_companion_agent.algorithms.multimodal_emotion import CLASSES, MultimodalEmotionRecognizer


def fixture(subject='test-subject', index=0, label='平静'):
    base = dict(zip(RAW_FEATURE_NAMES, [70.,40.,25.,10.,1.5,.005,0.,.002,.02,0.]))
    values = dict(base); group = CLASSES.index(label)
    values['hr_bpm'] += group * 10 + (index % 3) * .1
    values['gsr_mean_v'] += group * .15
    values['rmssd_ms'] += group * 3
    check = SensorCalibration(); check.baseline = base
    start = 200000 + index * 60000
    return {'subject_id':subject, 'device_id':'test-device', 'session_id':'test-session',
            'segment_id':0, 'window_start_ms':start, 'window_end_ms':start+58000,
            'features':check.transform(values), 'feature_schema':FEATURE_SCHEMA,
            'quality':{'quality_ok':True}, 'simulation':False,
            'label':label, 'label_source':'independent_self_report',
            'calibration':{'contact_confirmed':True,'rest_confirmed':True,
                           'baseline_end_ms':180000, 'baseline':base}}


class TestTrainingRows(unittest.TestCase):
    def test_auto_label_simulation_old_schema_and_bool_feature_are_rejected(self):
        rows=[]
        for key,value in [('label_source','algorithm'),('simulation',True),('feature_schema','old')]:
            row=fixture(); row[key]=value; rows.append(row)
        row=fixture(); row['features']['hr_bpm']=True; rows.append(row)
        row=fixture(); row['features']['gsr_std_v']=float('nan'); rows.append(row)
        for row in rows:
            with self.assertRaises(ValueError):validate_rows([row])

    def test_overlap_is_rejected_independent_of_row_order(self):
        row=fixture(); duplicate=copy.deepcopy(row)
        for rows in ([row,duplicate],[fixture(index=1),row,duplicate]):
            with self.assertRaises(ValueError):validate_rows(rows)
        self.assertEqual(len(validate_rows([fixture(index=1),row])),2)

    def test_baseline_overlap_missing_contact_or_inconsistent_relative_rejected(self):
        rows=[]
        row=fixture(); row['calibration']['baseline_end_ms']=220000; rows.append(row)
        row=fixture(); row['calibration']['contact_confirmed']=False; rows.append(row)
        row=fixture(); row['features']['hr_relative']=.3; rows.append(row)
        row=fixture(); row['window_end_ms']=float('inf'); rows.append(row)
        for row in rows:
            with self.assertRaises(ValueError):validate_rows([row])

    def test_small_dataset_is_not_a_five_class_model(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):train([fixture()],Path(directory)/'model.joblib')
            self.assertFalse((Path(directory)/'model.joblib').exists())

    def test_grouped_train_calibrate_and_save_exploratory_model(self):
        # No reported metric from this fixture is real emotion performance.
        rows=[fixture('fixture-subject-%d'%subject,index=group*12+n,label=label)
              for subject in range(8) for group,label in enumerate(CLASSES) for n in range(12)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.jsonl'
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            output=Path(directory)/'model.joblib'
            evaluation=train(load_rows(path),output)
            self.assertTrue(output.exists())
            self.assertFalse(set(evaluation['train_subjects'])&set(evaluation['test_subjects']))
            self.assertTrue(evaluation['calibration_grouped'])
            self.assertFalse(evaluation['deployment_ready'])
            self.assertFalse(MultimodalEmotionRecognizer(str(output)).available)
            with self.assertRaises(ValueError):train(rows,output)
