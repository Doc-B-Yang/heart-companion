"""Train on independent, non-overlapping, human-labelled multichannel windows.

Explicit JSONL input only. Group-held-out test subjects never appear in fitting
or probability calibration. Repeated windows/auto-labels are rejected. These
metrics describe this dataset and do not establish real-world/clinical accuracy.
"""
import argparse
import json
import math
import sys
import hashlib
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'硬件开发'/'.runtime'))
from hrv_companion_agent.core.sensor_pipeline import FEATURE_NAMES, FEATURE_SCHEMA, RAW_FEATURE_NAMES, finite
from hrv_companion_agent.core.sensor_calibration import SensorCalibration
from hrv_companion_agent.algorithms.multimodal_emotion import CLASSES, deployment_failures


def load_rows(path):
    return validate_rows([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()])


def validate_rows(input_rows):
    rows = []
    latest = {}
    for row in input_rows:
        if not isinstance(row, dict):
            raise ValueError('训练行必须为JSON对象')
        if (row.get('label_source') != 'independent_self_report'
                or row.get('feature_schema') != FEATURE_SCHEMA
                or row.get('label') not in CLASSES or row.get('simulation') is not False
                or not row.get('quality',{}).get('quality_ok')):
            raise ValueError('拒绝自动标签、模拟标签、旧特征或低质量训练行')
        if not all(isinstance(row.get(k), str) and 1 <= len(row[k]) <= 64 for k in ('subject_id', 'session_id', 'device_id')):
            raise ValueError('训练行必须有受试者、设备及会话标识')
        values = row.get('features',{})
        if any(not finite(values.get(k)) for k in FEATURE_NAMES):
            raise ValueError('训练特征缺失/非有限；禁止以0补充缺失通道')
        start,end = row.get('window_start_ms'),row.get('window_end_ms')
        if not finite(start) or not finite(end) or start < 0 or not 55000 <= end-start <= 61000:
            raise ValueError('训练窗口需约60秒（55~61秒）并提供准确起止时间')
        calibration = row.get('calibration', {})
        baseline = calibration.get('baseline', {})
        baseline_end = calibration.get('baseline_end_ms')
        if (calibration.get('contact_confirmed') is not True or calibration.get('rest_confirmed') is not True
                or not finite(baseline_end) or baseline_end < 0 or baseline_end > start
                or any(not finite(baseline.get(k)) for k in RAW_FEATURE_NAMES)
                or baseline['hr_bpm'] <= 0 or not .02 <= baseline['gsr_mean_v'] <= 3.28):
            raise ValueError('缺少独立静息基线、电极确认或窗口与基线重叠')
        check = SensorCalibration()
        check.baseline = baseline
        derived = check.transform(values)
        if any(not math.isclose(values[k], derived[k], rel_tol=1e-6, abs_tol=1e-8) for k in FEATURE_NAMES if k not in RAW_FEATURE_NAMES):
            raise ValueError('相对基线特征与原始量不一致')
        key = (row['subject_id'],row['device_id'],row['session_id'])
        latest.setdefault(key,[]).append((start,end))
        rows.append(row)
    for windows in latest.values():
        ordered=sorted(windows)
        if any(b[0]<a[1] for a,b in zip(ordered,ordered[1:])):
            raise ValueError('同一受试者/设备/会话存在重叠或重复标注窗口')
    return rows


def train(rows, output, seed=42):
    rows = validate_rows(rows)
    import numpy as np
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, log_loss, recall_score
    output = Path(output)
    if output.exists() or output.with_suffix('.evaluation.json').exists():
        raise ValueError('输出模型或评估已存在，请选择新文件名')
    counts=Counter(r['label'] for r in rows)
    subjects={r['subject_id'] for r in rows}
    if len(subjects)<8 or any(counts[c]<10 for c in CLASSES):
        raise ValueError('至少8名独立受试者、每类至少10个有效独立窗口；不足时不能训练五类模型')
    X=np.array([[r['features'][k] for k in FEATURE_NAMES] for r in rows])
    y=np.array([r['label'] for r in rows])
    groups=np.array([r['subject_id'] for r in rows])
    split=GroupShuffleSplit(n_splits=30,test_size=0.25,random_state=seed)
    selected=None
    for tr,te in split.split(X,y,groups):
        if set(y[tr])==set(CLASSES) and set(y[te])==set(CLASSES):
            selected=(tr,te)
            break
    if selected is None:
        raise ValueError('受试者分组后无法在训练/测试两侧覆盖五类，请补充平衡数据')
    tr,te=selected
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(X[tr],y[tr],groups[tr]))
    if any(set(y[tr][a])!=set(CLASSES) or set(y[tr][b])!=set(CLASSES) for a,b in folds):
        raise ValueError('分组概率校准折缺少类别，请补充不同受试者的标签')
    model=CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=160,max_depth=6,min_samples_leaf=3,
                               class_weight='balanced',random_state=seed,n_jobs=1),
        method='sigmoid',cv=folds)
    model.fit(X[tr],y[tr])
    predicted=model.predict(X[te])
    proba=model.predict_proba(X[te])
    ordered=np.sort(proba,axis=1)
    accepted=(ordered[:,-1]>=0.7)&((ordered[:,-1]-ordered[:,-2])>=0.15)
    evaluation={'group_holdout':True,'calibration_grouped':True,
        'train_subjects':sorted(set(groups[tr])),'test_subjects':sorted(set(groups[te])),
        'train_windows':len(tr),'test_windows':len(te),
        'balanced_accuracy':float(balanced_accuracy_score(y[te],predicted)),
        'macro_f1':float(f1_score(y[te],predicted,average='macro')),
        'accuracy':float(np.mean(predicted==y[te])),
        'min_class_recall':float(min(recall_score(y[te], predicted, labels=CLASSES, average=None))),
        'test_class_counts':dict(Counter(y[te])),
        'log_loss':float(log_loss(y[te],proba,labels=model.classes_)),
        'coverage':float(accepted.mean()),
        'accepted_accuracy':float(np.mean(predicted[accepted]==y[te][accepted])) if accepted.any() else None,
        'confusion_matrix':confusion_matrix(y[te],predicted,labels=CLASSES).tolist(),
        'target_85_met':bool(np.mean(predicted==y[te])>=0.85),
        'note':'固定留出受试者，仅本数据集验证；非临床准确率，部署须独立复核'}
    evaluation['deployment_blockers'] = deployment_failures(evaluation)
    evaluation['deployment_ready'] = not evaluation['deployment_blockers']
    span=np.ptp(X[tr], axis=0)
    padding=np.maximum(span*.25, .005)
    bounds={k:[float(X[tr,i].min()-padding[i]),float(X[tr,i].max()+padding[i])] for i,k in enumerate(FEATURE_NAMES)}
    bundle={'schema':FEATURE_SCHEMA,'feature_names':FEATURE_NAMES,'classes':CLASSES,
            'label_source':'independent_self_report','model':model,'confidence_threshold':0.7,
            'evaluation':evaluation,'seed':seed,'class_counts':dict(counts),
            'feature_bounds':bounds,
            'training_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise ValueError('输出模型已存在，请选择新文件名，避免覆盖已验证模型')
    joblib.dump(bundle,output)
    output.with_suffix('.evaluation.json').write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding='utf-8')
    return evaluation


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',required=True,help='网页独立自评标签JSONL')
    ap.add_argument('--output',default=str(Path(__file__).parent/'multimodal_emotion.joblib'))
    args=ap.parse_args()
    try:
        print(json.dumps(train(load_rows(args.data),args.output),ensure_ascii=False,indent=2))
    except (ValueError,FileNotFoundError) as exc:
        raise SystemExit(str(exc))

if __name__=='__main__':
    main()
