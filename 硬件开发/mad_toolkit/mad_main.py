"""
MAD 数据集完整分析流水线
=======================
使用 MAD 数据集 (或合成数据) 验证 HRV→情绪分类的完整链路。

对应 PPT 中的技术路线:
  STEP 01: 生理信号采集    → 已由 MAD 数据集完成
  STEP 02: HRV 特征提取    → HRVExtractor
  STEP 03: 情绪状态识别    → EmotionClassifier (规则/模糊/ML)
  STEP 04: 评估与对比      → 准确率、混淆矩阵、ECG vs PPG 对比

运行方式:
  # 使用合成数据快速验证
  python mad_main.py --synthetic --n_subjects 18 --n_per_subject 16

  # 使用 MAD 数据集
  python mad_main.py --data_path E:/MAD_dataset_v1.0

  # 仅评估已有模型
  python mad_main.py --evaluate --model_path models/emotion_rf.joblib

输出:
  results/
    confusion_matrix.png       ← 混淆矩阵图
    ecg_vs_ppg_comparison.csv  ← ECG vs PPG HRV 对比
    emotion_model_rf.joblib    ← 训练好的随机森林模型
    classification_report.txt  ← 分类报告
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mad_dataset_loader import MADLoader, generate_synthetic_ecg_ppg
from mad_hrv_extractor import HRVExtractor
from mad_emotion_classifier import EmotionClassifier, EMOTION_NAMES

# ── 配置 ──
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_synthetic_pipeline(args):
    """
    使用合成数据运行完整流水线

    优点: 无需下载数据集，立即验证算法
    缺点: 不能反映真实 PPG 信号的复杂性
    """
    print("=" * 70)
    print("合成数据流水线 (无需 MAD 数据集)")
    print("=" * 70)

    n_subjects = args.n_subjects or 18
    n_samples_per_emotion = args.n_per_subject or 8  # 每种情绪每被试

    emotions = list(EMOTION_NAMES.values())
    extractor = HRVExtractor(window_sec=300.0)

    # ── Phase 1: 数据生成 + HRV 提取 ──
    print(f"\n[Phase 1] 生成 {n_subjects} 个被试 × {len(emotions)} 种情绪...")

    all_features = []
    all_labels = []
    all_comparisons = []

    for subj in range(1, n_subjects + 1):
        for eid, emotion in enumerate(emotions):
            # 每个情绪生成 2 个样本
            for sample_i in range(2):
                seed = subj * 100 + eid * 10 + sample_i
                data = generate_synthetic_ecg_ppg(
                    duration_sec=300.0,
                    emotion_label=emotion,
                    seed=seed,
                )

                # ECG → HRV (金标准)
                ecg_feat = extractor.extract_from_ecg(data["ecg"], fs=data["fs_ecg"])

                # PPG → HRV (模拟 MAX30102)
                ppg_feat = extractor.extract_from_ppg(data["ppg"], fs=data["fs_ppg"])

                # 对比
                comp = extractor.compare_ecg_ppg(ecg_feat, ppg_feat)
                comp["emotion"] = emotion
                comp["subject"] = subj
                all_comparisons.append(comp)

                # 用 ECG 特征训练 (作为金标准)
                all_features.append(ecg_feat)
                all_labels.append(eid)

                # 可选: 也保存 PPG 特征 (用于模拟真实传感器场景)
                ppg_feat_copy = ppg_feat.copy()
                ppg_feat_copy["emotion"] = emotion
                ppg_feat_copy["subject"] = subj

    print(f"  总数: {len(all_features)} 个样本")

    # ── Phase 2: 保存 ECG vs PPG 对比 ──
    print(f"\n[Phase 2] ECG vs PPG HRV 对比分析...")
    df_comp = pd.DataFrame(all_comparisons)

    # 按情绪聚合
    emotion_comparison = df_comp.groupby("emotion").agg({
        "hr_error_pct": "mean",
        "sdnn_error_pct": "mean",
        "rmssd_error_pct": "mean",
        "avg_error_pct": "mean",
        "agreement": lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A",
    }).round(1)

    print("\n  ECG vs PPG 对比 (按情绪):")
    print(emotion_comparison.to_string())

    comp_path = RESULTS_DIR / "ecg_vs_ppg_comparison.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"\n  对比结果已保存: {comp_path}")

    # ── Phase 3: 规则/模糊层测试 ──
    print(f"\n[Phase 3] 规则+模糊层测试...")
    clf = EmotionClassifier()

    rule_correct = 0
    fuzzy_correct = 0
    for features, label in zip(all_features, all_labels):
        result = clf.recognize(features)
        if result.emotion_id == label:
            if result.layer.startswith("rule"):
                rule_correct += 1
            elif result.layer == "fuzzy":
                fuzzy_correct += 1

    n = len(all_features)
    print(f"  规则层准确率: {rule_correct/n*100:.1f}% ({rule_correct}/{n})")
    print(f"  模糊层命中: {fuzzy_correct/n*100:.1f}% ({fuzzy_correct}/{n})")

    # ── Phase 4: ML 训练 ──
    print(f"\n[Phase 4] ML 分类器训练...")

    from mad_emotion_classifier import MLClassifier
    ml = MLClassifier(model_type="rf")
    X_feat = ml.features_to_array(all_features)
    y_arr = np.array(all_labels)
    report = ml.train(X_feat, y_arr)

    # 将训练好的模型注入 EmotionClassifier
    clf.set_ml_model(ml)

    print(f"\n  分类报告:")
    print(report["report"])

    # 保存模型
    model_path = RESULTS_DIR / "emotion_model_rf.joblib"
    ml.save(str(model_path))

    # 保存报告
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MAD Emotion Classification Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Samples: {len(all_labels)}\n")
        f.write(f"CV Accuracy: {report['cv_accuracy_mean']:.3f} ± {report['cv_accuracy_std']:.3f}\n\n")
        f.write(report["report"])
    print(f"  报告已保存: {report_path}")

    # ── 总结 ──
    print(f"\n{'='*70}")
    print(f"流水线完成!")
    print(f"{'='*70}")
    print(f"  输出文件:")
    print(f"    {comp_path}")
    print(f"    {model_path}")
    print(f"    {report_path}")
    print(f"\n  关键指标:")
    print(f"    规则+模糊层准确率: {rule_correct/(n)*100:.1f}%")
    print(f"    ML模型 CV 准确率: {report['cv_accuracy_mean']*100:.1f}%")
    print(f"    PPG vs ECG 平均误差: {df_comp['avg_error_pct'].mean():.1f}%")


def run_mad_pipeline(args):
    """使用真实 MAD 数据集运行流水线"""
    if not args.data_path:
        print("错误: 需要 --data_path 指定 MAD 数据集路径")
        return

    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"错误: 路径不存在: {data_path}")
        return

    print("=" * 70)
    print("MAD 数据集真实流水线")
    print("=" * 70)

    # 加载数据
    loader = MADLoader(str(data_path))
    subjects = loader.list_subjects()
    print(f"发现 {len(subjects)} 个被试")

    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    extractor = HRVExtractor(window_sec=300.0)
    all_features = []
    all_labels = []
    all_comparisons = []

    # 遍历所有被试和 session
    for sid in subjects:
        for ssid in loader.list_sessions(sid):
            try:
                data = loader.load_full_subject(sid, ssid)
                signals = data["signals"]
                labels = data["labels"]

                if "ecg" not in signals or "ppg" not in signals:
                    continue

                # 分段提取 HRV (每 300 秒一段)
                ecg = signals["ecg"]
                ppg = signals["ppg"]
                fs_ecg = signals.get("fs_ecg", 1000.0)
                fs_ppg = signals.get("fs_ppg", 34.0)

                # 将长信号分段
                segment_sec = 300.0
                n_segments_ecg = int(len(ecg) / (segment_sec * fs_ecg))
                n_segments_ppg = int(len(ppg) / (segment_sec * fs_ppg))
                n_segments = max(1, min(n_segments_ecg, n_segments_ppg,
                                       len(labels.get("stimulus", [1]))))

                for seg_idx in range(n_segments):
                    start_ecg = int(seg_idx * segment_sec * fs_ecg)
                    end_ecg = int((seg_idx + 1) * segment_sec * fs_ecg)
                    start_ppg = int(seg_idx * segment_sec * fs_ppg)
                    end_ppg = int((seg_idx + 1) * segment_sec * fs_ppg)

                    seg_ecg = ecg[start_ecg:end_ecg]
                    seg_ppg = ppg[start_ppg:end_ppg]

                    if len(seg_ecg) < fs_ecg * 30 or len(seg_ppg) < fs_ppg * 30:
                        continue  # 跳过太短的段

                    # ECG HRV
                    ecg_feat = extractor.extract_from_ecg(seg_ecg, fs=fs_ecg)
                    # PPG HRV
                    ppg_feat = extractor.extract_from_ppg(seg_ppg, fs=fs_ppg)

                    if ecg_feat["hr_bpm"] == 0:
                        continue

                    # 对应标签
                    emotion_id = (labels.get("stimulus", [0]) + [0])[min(seg_idx, len(labels.get("stimulus", [0])) - 1)]

                    all_features.append(ecg_feat)
                    all_labels.append(emotion_id)

                    comp = extractor.compare_ecg_ppg(ecg_feat, ppg_feat)
                    comp["subject"] = sid
                    comp["session"] = ssid
                    comp["segment"] = seg_idx
                    all_comparisons.append(comp)

            except Exception as e:
                print(f"  [ERROR] subject_{sid:03d}/session_{ssid:02d}: {e}")

    print(f"\n  累计有效样本: {len(all_features)}")

    if len(all_features) == 0:
        print("错误: 没有提取到有效样本. 请检查数据格式和路径.")
        return

    # 保存对比
    df_comp = pd.DataFrame(all_comparisons)
    comp_path = RESULTS_DIR / "ecg_vs_ppg_comparison_mad.csv"
    df_comp.to_csv(comp_path, index=False)
    print(f"  对比结果保存: {comp_path}")

    # 训练 ML
    from mad_emotion_classifier import MLClassifier
    ml = MLClassifier(model_type="rf")
    X = ml.features_to_array(all_features)
    y = np.array(all_labels)
    report = ml.train(X, y)

    model_path = RESULTS_DIR / "emotion_model_rf_mad.joblib"
    ml.save(str(model_path))

    report_path = RESULTS_DIR / "classification_report_mad.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MAD Dataset Emotion Classification Report\n")
        f.write(f"Subjects: {len(subjects)}\n")
        f.write(f"Samples: {len(all_labels)}\n")
        f.write(f"CV Accuracy: {report['cv_accuracy_mean']:.3f} ± {report['cv_accuracy_std']:.3f}\n\n")
        f.write(report["report"])
    print(f"  报告保存: {report_path}")

    print(f"\n{'='*70}")
    print(f"流水线完成! 模型: {model_path}")


def run_evaluate(args):
    """评估已有模型"""
    if not args.model_path:
        print("错误: 需要 --model_path 指定模型文件")
        return

    print("=" * 70)
    print("模型评估")
    print("=" * 70)

    clf = EmotionClassifier()
    clf.load_ml_model(args.model_path)

    # 生成测试样本
    test_cases = [
        {"hr_bpm": 72, "sdnn_ms": 55, "rmssd_ms": 42, "pnn50_pct": 18, "lf_hf": 1.8,
         "sd1_ms": 30, "sd2_ms": 65, "sd_ratio": 2.2},
        {"hr_bpm": 95, "sdnn_ms": 25, "rmssd_ms": 18, "pnn50_pct": 3,  "lf_hf": 4.2,
         "sd1_ms": 12, "sd2_ms": 30, "sd_ratio": 2.5},
        {"hr_bpm": 78, "sdnn_ms": 48, "rmssd_ms": 36, "pnn50_pct": 14, "lf_hf": 2.0,
         "sd1_ms": 26, "sd2_ms": 58, "sd_ratio": 2.2},
        {"hr_bpm": 88, "sdnn_ms": 32, "rmssd_ms": 22, "pnn50_pct": 8,  "lf_hf": 3.0,
         "sd1_ms": 16, "sd2_ms": 38, "sd_ratio": 2.4},
    ]

    labels = ["Neutral", "Anxiety", "Happy", "Angry"]

    for i, features in enumerate(test_cases):
        result = clf.recognize(features)
        match = "✅" if result.emotion == labels[i] else "❌"
        print(f"\n{match} {labels[i]:10s} → {result.emotion:10s} "
              f"({result.layer:6s}, conf={result.confidence:.3f}, risk={result.risk_score:.0f})")


def main():
    parser = argparse.ArgumentParser(
        description="MAD 数据集情绪识别流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 合成数据快速测试
  python mad_main.py --synthetic --n_subjects 18

  # 真实 MAD 数据
  python mad_main.py --data_path E:/datasets/MAD_v1.0 --max_subjects 5

  # 仅评估
  python mad_main.py --evaluate --model_path results/emotion_model_rf.joblib
        """
    )

    parser.add_argument("--synthetic", action="store_true",
                       help="使用合成数据 (无需下载数据集)")
    parser.add_argument("--data_path", type=str, default=None,
                       help="MAD 数据集根目录路径")
    parser.add_argument("--evaluate", action="store_true",
                       help="仅评估已有模型")
    parser.add_argument("--model_path", type=str, default=None,
                       help="模型文件路径 (.joblib)")
    parser.add_argument("--n_subjects", type=int, default=None,
                       help="合成数据: 被试数量 (默认18)")
    parser.add_argument("--n_per_subject", type=int, default=None,
                       help="合成数据: 每种情绪每被试样本数")
    parser.add_argument("--max_subjects", type=int, default=None,
                       help="MAD 数据: 最多加载几个被试")

    args = parser.parse_args()

    if args.evaluate:
        run_evaluate(args)
    elif args.data_path:
        run_mad_pipeline(args)
    elif args.synthetic or not args.data_path:
        run_synthetic_pipeline(args)
    else:
        # 默认: 合成数据
        args.synthetic = True
        args.n_subjects = 18
        run_synthetic_pipeline(args)


if __name__ == "__main__":
    main()
