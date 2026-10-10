"""Fresh, quality-passed sensor windows paired with independent user labels.
No algorithm predictions as ground truth; no overlapping labelled windows.
"""
import json
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from .. import config
from .sensor_pipeline import FEATURE_NAMES, FEATURE_SCHEMA
from .sensor_pipeline import finite
from ..algorithms.multimodal_emotion import CLASSES


class LabelStore:
    def __init__(self, path=None):
        self.path = Path(path or config.DATA_DIR / 'multimodal_labels.jsonl')
        self.lock = threading.Lock()

    def append(self, system, label, note='', arousal_score=None, valence_score=None):
        if label not in CLASSES:
            raise ValueError('请选择已定义的五类自评标签')
        for score in (arousal_score, valence_score):
            if score is not None and (type(score) is not int or not 1 <= score <= 9):
                raise ValueError('唤醒度和愉悦度必须为1~9的整数自评，或留空')
        with system._sensor_lock, self.lock:
            state = system.get_state() or {}
            flow = system._sensor_flow or {}
            if (state.get('simulation') is not False or state.get('historical') or not state.get('quality',{}).get('quality_ok')
                    or not flow.get('accepted')):
                raise ValueError('请先获得新鲜且通过质量检查的硬件窗口')
            feats = state.get('features',{})
            if not all(finite(feats.get(k)) for k in FEATURE_NAMES):
                raise ValueError('多通道特征不完整')
            calibration = system.sensor_calibration
            if not state.get('calibration', {}).get('contact_ready') or calibration.baseline is None:
                raise ValueError('请先完成本次佩戴的电极确认和静息基线')
            session = state['sensor']
            start, end = flow['window']['start_ms'], flow['window']['end_ms']
            if start < calibration.baseline_end_ms:
                raise ValueError('分类窗口仍与静息基线重叠，请继续稳定采集约60秒')
            key = (system.user_id,session['device_id'],session['boot_id'])
            existing = []
            if self.path.exists():
                for line in self.path.read_text(encoding='utf-8').splitlines():
                    if line.strip():
                        existing.append(json.loads(line))
            for row in reversed(existing):
                if (row['subject_id'],row['device_id'],row['session_id']) == key:
                    if start < row['window_end_ms']:
                        raise ValueError('与上次标注窗口重叠，请继续稳定采集至少60秒')
                    break
            captured_at = time.time()
            row = {'subject_id':system.user_id,'device_id':session['device_id'],
                   'session_id':session['boot_id'],'segment_id':session['segment_id'],
                   'window_start_ms':start,'window_end_ms':end,'timestamp':captured_at,
                   'collection_day': datetime.fromtimestamp(captured_at, timezone(timedelta(hours=8))).date().isoformat(),
                   'feature_schema':FEATURE_SCHEMA,'features':{k:feats[k] for k in FEATURE_NAMES},
                   'quality':state['quality'],'label':label,
                   'simulation': False,
                   'calibration': {'contact_confirmed': True, 'rest_confirmed': True,
                       'baseline_end_ms': calibration.baseline_end_ms,
                       'baseline': dict(calibration.baseline)},
                   'label_source':'independent_self_report','note':str(note)[:1000]}
            # Separate human ratings: never infer arousal/valence from emotion names.
            row['self_report'] = {'arousal_1_9': arousal_score, 'valence_1_9': valence_score,
                                  'scope': 'preceding_60_second_window'}
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open('a',encoding='utf-8') as out:
                out.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            return {'saved':True,'label':label,'samples':len(existing)+1,
                    'next_label_after_sec':60,'feature_schema':FEATURE_SCHEMA}

    def export(self, user_id):
        with self.lock:
            rows = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()] if self.path.exists() else []
            return [row for row in rows if row.get('subject_id') == user_id]

    def purge(self, user_id):
        with self.lock:
            if not self.path.exists():
                return 0
            rows = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
            keep = [row for row in rows if row.get('subject_id') != user_id]
            staged = self.path.with_suffix('.tmp')
            staged.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in keep), encoding='utf-8')
            staged.replace(self.path)
            return len(rows) - len(keep)


class SelfReportStore(LabelStore):
    """Human observations remain valid even without a qualified sensor window."""
    def __init__(self, path=None):
        super().__init__(path or config.DATA_DIR / 'experiment_self_reports.jsonl')

    def record(self, user_id, label, note='', arousal_score=None, valence_score=None):
        if label not in CLASSES:
            raise ValueError('请选择已定义的五类自评标签')
        for score in (arousal_score, valence_score):
            if score is not None and (type(score) is not int or not 1 <= score <= 9):
                raise ValueError('评分必须为1~9的整数，或留空')
        row = {'subject_id': user_id, 'timestamp': time.time(), 'label': label,
               'note': str(note)[:1000], 'label_source': 'independent_self_report',
               'arousal_1_9': arousal_score, 'valence_1_9': valence_score,
               'scope': 'preceding_60_second_window', 'record_kind': 'subjective_only'}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as out:
                out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        return row
