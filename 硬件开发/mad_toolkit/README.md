# MAD 数据集情绪识别工具包

> 为「基于实时心率变异性驱动的主动式情感陪伴系统」大创项目开发  
> 北京交通大学 · 2026

---

## 快速开始

```bash
# 安装依赖
pip install pandas numpy scipy scikit-learn

# 使用合成数据快速测试（无需下载MAD数据集）
python mad_main.py --synthetic --n_subjects 5

# 使用真实MAD数据集
python mad_main.py --data_path /path/to/MAD_dataset_v1.0 --max_subjects 5

# 评估已有模型
python mad_main.py --evaluate --model_path results/emotion_model_rf.joblib
```

---

## 文件说明

| 文件 | 功能 | 对应PPT |
|------|------|---------|
| `mad_dataset_loader.py` | MAD数据集解析 + 合成数据生成 | STEP 01: 生理信号采集 |
| `mad_hrv_extractor.py` | HRV特征提取 (ECG金标准 + PPG模拟) | STEP 02: HRV特征提取 |
| `mad_emotion_classifier.py` | 三层情绪识别引擎 (规则/模糊/ML) | STEP 03: 情绪识别引擎 |
| `mad_main.py` | 完整流水线主程序 | 全流程验证 |

---

## 三层情绪识别架构

```
HRV特征 → Layer 1(规则状态机) → confidence≥0.7 → 直接输出
                     │
                     ├ confidence<0.7
                     ▼
              Layer 2(模糊推理) → confidence≥0.6 → 输出
                     │
                     ├ confidence<0.6
                     ▼
              Layer 3(ML分类器) → 最终输出
```

- **Layer 1**: 极低计算开销，适合ESP32边缘部署
- **Layer 2**: 处理边界模糊区域
- **Layer 3**: 最高精度，需要PC/服务器运行

---

## 输出文件（在 `results/` 目录)

| 文件 | 说明 |
|------|------|
| `ecg_vs_ppg_comparison.csv` | ECG vs PPG 各指标的对比误差 |
| `emotion_model_rf.joblib` | 训练好的随机森林模型 |
| `classification_report.txt` | 分类准确率报告 |

---

## 核心API示例

```python
from mad_dataset_loader import generate_synthetic_ecg_ppg
from mad_hrv_extractor import HRVExtractor
from mad_emotion_classifier import EmotionClassifier

# 生成模拟数据
data = generate_synthetic_ecg_ppg(
    duration_sec=300.0,
    emotion_label="Anxiety",
    seed=42
)

# 提取HRV特征
extractor = HRVExtractor(window_sec=300.0)
ecg_features = extractor.extract_from_ecg(data["ecg"], fs=1000.0)
ppg_features = extractor.extract_from_ppg(data["ppg"], fs=34.0)

# 情绪识别
clf = EmotionClassifier()
result = clf.recognize(ecg_features)
print(f"情绪: {result.emotion}, 置信度: {result.confidence:.2f}, 风险: {result.risk_score:.0f}")

# ECG vs PPG 对比
comparison = extractor.compare_ecg_ppg(ecg_features, ppg_features)
print(f"一致度: {comparison['agreement']}, 平均误差: {comparison['avg_error_pct']:.1f}%")
```

---

## 接入你的ESP32硬件后

当ESP32固件调试完毕后，将以下流程嵌入你的后端：

```python
# 1. 后端收到BLE数据
ble_data = {
    "hr": 85, "ibi": [720, 740, 710], "sdnn": 32.5,
    "rmssd": 22.1, "pnn50": 8.3, "rr": 18.5, "q": 0.88
}

# 2. 转换为HRV特征字典
features = {
    "hr_bpm": ble_data["hr"],
    "sdnn_ms": ble_data["sdnn"],
    "rmssd_ms": ble_data["rmssd"],
    "pnn50_pct": ble_data["pnn50"],
    "lf_hf": 3.2,  # ESP32目前不计算此值，用默认值
}

# 3. 情绪识别
clf = EmotionClassifier()
clf.load_ml_model("emotion_model_rf.joblib")
result = clf.recognize(features)

# 4. 存入CompanionAgent的上下文
agent_context = f"当前情绪: {result.emotion}, 风险评分: {result.risk_score:.0f}/100"
```
