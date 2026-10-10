"""Personal exploratory arousal model; never auto-loaded into production.

Run from the package parent with python -m hrv_companion_agent.models.experimental_arousal.
Dependencies: numpy, scikit-learn, joblib (requirements-optional.txt).
Only independent 1..9 human ratings are targets. Hold out the latest collection
day; no random window split, test-driven tuning, or production accuracy claim.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

from ..core.sensor_pipeline import FEATURE_NAMES, FEATURE_SCHEMA, finite
from .train_multimodal import validate_rows

PPG = ('hr_bpm', 'sdnn_ms', 'rmssd_ms', 'pnn50_pct', 'hr_relative', 'rmssd_relative')
GSR = ('gsr_mean_v', 'gsr_std_v', 'gsr_slope_v_s', 'gsr_step_rms_v', 'gsr_range_v', 'gsr_relative')
FEATURE_SETS = {'ppg': PPG, 'gsr': GSR, 'fusion': PPG + GSR}
CLASSES = ('低唤醒', '高唤醒')
MODEL_SCHEMA = 'personal-arousal-experiment-v1'


def prepare(rows):
    rows = validate_rows(rows)
    if len({r['subject_id'] for r in rows}) != 1:
        raise ValueError('个人实验模型必须且只能包含同一参与者')
    selected, neutral, unrated = [], 0, 0
    for row in rows:
        score = row.get('self_report', {}).get('arousal_1_9')
        if score is None:
            unrated += 1
            continue
        if type(score) is not int or not 1 <= score <= 9:
            raise ValueError('必须使用独立的1~9唤醒度自评，不能从情绪名称推导')
        if row.get('self_report', {}).get('scope') != 'preceding_60_second_window':
            raise ValueError('自评必须对应刚过去的60秒窗口')
        stamp = row.get('timestamp')
        if not finite(stamp) or stamp <= 0:
            raise ValueError('缺少真实采集时间，无法按日期隔离')
        day = datetime.fromtimestamp(stamp, timezone(timedelta(hours=8))).date().isoformat()
        if row.get('collection_day') != day:
            raise ValueError('采集日期与时间戳不一致')
        if score == 5:
            neutral += 1
            continue
        selected.append({**row, 'arousal_target': int(score > 5)})
    return selected, {'neutral_excluded': neutral, 'unrated_excluded': unrated}


def split_days(rows, test_day):
    days = {r['collection_day'] for r in rows}
    if len(days) < 2 or test_day != max(days):
        raise ValueError('至少两个采集日期；测试日期必须是最后一个日期')
    train = [r for r in rows if r['collection_day'] < test_day]
    test = [r for r in rows if r['collection_day'] == test_day]
    for part in (train, test):
        counts = Counter(r['arousal_target'] for r in part)
        if any(counts[c] < 5 for c in (0, 1)):
            raise ValueError('训练及测试两侧各需要每类至少5个不重叠有效窗口；这只是试验最低量')
    return train, test


def train(rows, output, test_day):
    import numpy as np
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, accuracy_score

    output = Path(output)
    report_path = output.with_suffix('.evaluation.json')
    if output.exists() or report_path.exists():
        raise ValueError('输出已存在，请选择新名称，保留每次试验记录')
    selected, excluded = prepare(rows)
    tr, te = split_days(selected, test_day)
    ytr = np.array([r['arousal_target'] for r in tr])
    yte = np.array([r['arousal_target'] for r in te])
    metrics, models = {}, {}
    for name, names in FEATURE_SETS.items():
        xtr = np.array([[r['features'][k] for k in names] for r in tr])
        xte = np.array([[r['features'][k] for k in names] for r in te])
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., class_weight='balanced',
                                                                    max_iter=2000, random_state=42))
        model.fit(xtr, ytr)  # Scaler is fitted ONLY to train days.
        probabilities = model.predict_proba(xte)
        prediction = model.predict(xte)
        accepted = probabilities.max(axis=1) >= .65
        metrics[name] = {'balanced_accuracy': float(balanced_accuracy_score(yte, prediction)),
                         'macro_f1': float(f1_score(yte, prediction, average='macro', zero_division=0)),
                         'confusion_matrix': confusion_matrix(yte, prediction, labels=[0, 1]).tolist(),
                         'coverage_at_0_65': float(accepted.mean()),
                         'accepted_accuracy': float(accuracy_score(yte[accepted], prediction[accepted])) if accepted.any() else None}
        models[name] = model
    dummy = DummyClassifier(strategy='most_frequent').fit(np.zeros((len(tr), 1)), ytr)
    dummy_score = balanced_accuracy_score(yte, dummy.predict(np.zeros((len(te), 1))))
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()
    report = {'schema': MODEL_SCHEMA, 'scope': 'personal_exploratory_only', 'production_ready': False,
              'subject_id': tr[0]['subject_id'], 'train_days': sorted({r['collection_day'] for r in tr}),
              'test_day': test_day, 'train_windows': len(tr), 'test_windows': len(te),
              'train_class_counts': dict(Counter(int(x) for x in ytr)),
              'test_class_counts': dict(Counter(int(x) for x in yte)), 'excluded': excluded,
              'classes': list(CLASSES), 'metrics': metrics, 'dummy_balanced_accuracy': float(dummy_score),
              'selection': 'fusion fixed before evaluation; no selecting best test score',
              'probability_note': 'Uncalibrated model score; 0.65 is an exploratory abstention threshold, not emotional certainty.',
              'data_sha256': digest,
              'limitations': ['Same-person held-out day only; no unseen-person claim.',
                              'Small correlated task blocks; no clinical or named-emotion accuracy claim.',
                              'PPG PRV and uncalibrated GSR voltage; not ECG HRV or conductance.']}
    names = FEATURE_SETS['fusion']
    bundle = {'schema': MODEL_SCHEMA, 'feature_schema': FEATURE_SCHEMA, 'feature_names': names,
              'subject_id': tr[0]['subject_id'], 'device_ids': sorted({r['device_id'] for r in tr}),
              'classes': CLASSES, 'model': models['fusion'], 'evaluation': report,
              'bounds': {k: [min(r['features'][k] for r in tr), max(r['features'][k] for r in tr)] for k in names}}
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def predict(bundle, row):
    """Predict one exported qualified window, requiring a fresh personal baseline.

    Row may omit new arousal/valence ratings; existing emotion self-report is
    validated for compatibility but never used as a predictor.
    """
    validate_rows([row])
    if bundle.get('schema') != MODEL_SCHEMA or bundle.get('feature_schema') != FEATURE_SCHEMA:
        raise ValueError('实验模型格式不兼容')
    if row['subject_id'] != bundle['subject_id'] or row['device_id'] not in bundle['device_ids']:
        raise ValueError('模型仅适用于训练时的参与者和设备')
    names = tuple(bundle['feature_names'])
    if names != FEATURE_SETS['fusion']:
        raise ValueError('特征契约不匹配')
    result = {'scope': 'personal_exploratory_only', 'emotion': None, 'status': 'uncertain',
              'probability_calibrated': False, 'reason': '模型分数不足，暂不推测'}
    for k in names:
        lo, hi = bundle['bounds'][k]
        allowance = max(abs(lo), abs(hi), hi - lo, 1e-6) * .5
        if not lo - allowance <= row['features'][k] <= hi + allowance:
            return {**result, 'reason': '特征超出训练范围，请检查佩戴或重新采集'}
    probabilities = bundle['model'].predict_proba([[row['features'][k] for k in names]])[0]
    index = max(range(len(probabilities)), key=lambda n: probabilities[n])
    score = float(probabilities[index])
    result['model_score'] = score
    if score >= .65:
        result.update(status='estimated', arousal=CLASSES[int(bundle['model'].classes_[index])],
                      reason='仅推测生理唤醒程度；具体感受以本人表达为准')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('--data', required=True)
    fit.add_argument('--output', required=True)
    fit.add_argument('--test-day', required=True)
    infer = sub.add_parser('predict')
    infer.add_argument('--model', required=True, help='只加载自己生成的可信joblib文件')
    infer.add_argument('--window', required=True, help='一个有效窗口的JSON文件')
    args = parser.parse_args()
    if args.command == 'train':
        rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines() if line.strip()]
        result = train(rows, args.output, args.test_day)
    else:
        import joblib
        result = predict(joblib.load(args.model), json.loads(Path(args.window).read_text(encoding='utf-8')))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
