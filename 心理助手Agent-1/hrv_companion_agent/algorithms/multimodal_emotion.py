"""Validated multichannel classifier. No model -> explicit abstention.

Legacy MAD/auto-labelled bundles are not silently promoted to five-class truth.
Only an explicit opt-in activates a bundle produced by train_multimodal.py,
whose group-held-out metrics and confidence policy are recorded in metadata.
"""
import math
from pathlib import Path
from .. import config
from ..core.sensor_pipeline import FEATURE_NAMES, FEATURE_SCHEMA, finite
from .emotion_recognizer import EmotionResult, EMOTION_BASE_RISK

CLASSES = ['平静', '低焦虑', '高焦虑', '愉悦', '疲劳']


def deployment_failures(evaluation):
    """Engineering release criteria, not medical validation or a promised accuracy."""
    reasons = []
    if evaluation.get('group_holdout') is not True or evaluation.get('calibration_grouped') is not True:
        reasons.append('缺少按受试者分组的留出和概率校准')
    train, test = evaluation.get('train_subjects', []), evaluation.get('test_subjects', [])
    if not all(isinstance(x, str) for x in train + test) or set(train) & set(test):
        reasons.append('训练与测试受试者不独立')
    if len(set(train + test)) < 20 or len(set(test)) < 4:
        reasons.append('需要至少20名受试者，其中至少4名独立测试受试者')
    if any(evaluation.get('test_class_counts', {}).get(c, 0) < 10 for c in CLASSES):
        reasons.append('测试集中每类需要至少10个不重叠窗口')
    for key, threshold in [('balanced_accuracy', .85), ('macro_f1', .80),
                           ('min_class_recall', .70), ('coverage', .50),
                           ('accepted_accuracy', .85)]:
        value = evaluation.get(key)
        if not finite(value) or not threshold <= value <= 1:
            reasons.append(key + '未达到发布筛查条件')
    return reasons


class MultimodalEmotionRecognizer:
    def __init__(self, path=None):
        self.bundle = None
        self.error = '尚未启用经过真实标签验证的多通道模型'
        self.pending = None
        self.pending_since = None
        self.last_seq = None
        self.last_now = None
        self.available = False
        path = path or config.MULTIMODAL_MODEL_PATH
        if not config.MULTIMODAL_MODEL_ENABLED and path == config.MULTIMODAL_MODEL_PATH:
            return
        try:
            import joblib
            bundle = joblib.load(Path(path))
            if (bundle.get('schema') != FEATURE_SCHEMA or bundle.get('feature_names') != FEATURE_NAMES
                    or bundle.get('label_source') != 'independent_self_report'
                    or not bundle.get('evaluation', {}).get('group_holdout')
                    or set(bundle.get('classes', [])) != set(CLASSES)):
                raise ValueError('模型来源、特征版本、标签或分组验证不兼容')
            failures = deployment_failures(bundle['evaluation'])
            if failures:
                raise ValueError('；'.join(failures))
            if set(bundle['model'].classes_) != set(CLASSES) or len(bundle['model'].classes_) != len(CLASSES):
                raise ValueError('模型实际类别不完整')
            bounds = bundle.get('feature_bounds', {})
            if any(k not in bounds or len(bounds[k]) != 2 or not all(finite(v) for v in bounds[k])
                   or bounds[k][0] > bounds[k][1] for k in FEATURE_NAMES):
                raise ValueError('缺少训练域范围，无法拒绝域外输入')
            if not finite(bundle.get('confidence_threshold')) or not .7 <= bundle['confidence_threshold'] <= 1:
                raise ValueError('无效置信度门槛')
            self.bundle, self.available, self.error = bundle, True, ''
        except Exception as exc:
            self.error = '模型不可用: ' + str(exc)

    def reset(self):
        self.pending = self.pending_since = self.last_seq = self.last_now = None

    def abstain(self, reason, layer='uncertain', detail=None):
        return EmotionResult('不确定', 0.0, 0.0, layer,
                             {'reason': reason, 'validated': False, **(detail or {})})

    def recognize(self, feats, now, seq, session=None):
        if not self.available:
            return self.abstain(self.error, 'model_unavailable')
        if any(not finite(feats.get(k)) for k in FEATURE_NAMES):
            self.reset()
            return self.abstain('多通道特征缺失')
        try:
            if not finite(now) or not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
                raise ValueError('推理时间或序号无效')
            if any(not self.bundle['feature_bounds'][k][0] <= feats[k] <= self.bundle['feature_bounds'][k][1] for k in FEATURE_NAMES):
                self.reset()
                return self.abstain('输入超出训练数据范围，需要补充当前场景的标注')
            probabilities = self.bundle['model'].predict_proba([[feats[k] for k in FEATURE_NAMES]])[0]
            labels = list(self.bundle['model'].classes_)
            if (len(probabilities) != len(CLASSES) or set(labels) != set(CLASSES)
                    or not all(finite(float(p)) and 0 <= p <= 1 for p in probabilities)
                    or abs(sum(probabilities) - 1) > .001):
                raise ValueError('模型返回无效概率')
            pairs = sorted(zip(labels, map(float, probabilities)), key=lambda p:p[1], reverse=True)
            label, confidence = pairs[0]
            margin = confidence-pairs[1][1]
            minimum = max(0.65, float(self.bundle.get('confidence_threshold', 0.7)))
            if confidence < minimum or margin < 0.15:
                self.reset()
                return self.abstain('类别概率重叠或置信不足', detail={'probabilities': dict(pairs)})
            # Persistence is measured in device time, not repeated HTTP polls.
            key = (session, label)
            if self.pending != key or (self.last_seq is not None and
                    (seq != self.last_seq+1 or not 0 < now-self.last_now <= 2.5)):
                self.pending, self.pending_since = key, now
            self.last_seq = seq
            self.last_now = now
            if now-self.pending_since < config.EMOTION_CONFIRM_SEC:
                return self.abstain('分类持续确认中', 'confirming', {'probabilities':dict(pairs)})
            return EmotionResult(label, round(confidence,3), EMOTION_BASE_RISK[label],
                                 'multimodal_ml', {'probabilities':dict(pairs), 'validated':True,
                                  'model_evaluation':self.bundle['evaluation'],
                                  'feature_schema':FEATURE_SCHEMA})
        except Exception as exc:
            self.reset()
            return self.abstain('模型推理失败: '+str(exc))
