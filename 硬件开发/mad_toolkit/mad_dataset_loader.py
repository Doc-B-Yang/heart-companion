"""
MAD 数据集加载器
===============
解析 MAD (Multimodal Affective Dataset) 的目录结构，
提取 PPG/ECG 信号和情绪标签。

MAD 数据集目录结构:
  subject_xxx/
    session_xx/
      PPG_BCG/          ← CSV, 34Hz
      ECG_EMG_EOG/      ← EDF+TXT, 1000Hz
      EEG/              ← CSV, 128Hz
  labels/               ← 情绪标签文件
  session_datetime.csv  ← 时序信息

用法:
  loader = MADLoader("path/to/MAD_dataset")
  signals, labels = loader.load_subject(1, 1)  # subject_001, session_01
"""

import os
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MADLoader:
    """MAD 数据集加载器"""

    # 刺激标签映射（0-6）
    STIMULUS_LABELS = {
        0: "Neutral",    # 中性
        1: "Happy",      # 开心
        2: "Anxiety",    # 焦虑 ← 你的项目核心
        3: "Fearful",    # 恐惧
        4: "Sad",        # 悲伤
        5: "Angry",      # 愤怒
        6: "Disgust",    # 厌恶
    }

    # 自评标签映射（含强度）
    SELF_ASSESSED_MAP = {
        0: ("Neutral", 0),
        1: ("Happy", 1),    2: ("Happy", 2),    11: ("Happy", 2),
        2: ("Anxiety", 1),  22: ("Anxiety", 2),
        3: ("Fearful", 1),  33: ("Fearful", 2),
        4: ("Sad", 1),      44: ("Sad", 2),
        5: ("Angry", 1),    55: ("Angry", 2),
        6: ("Disgust", 1),  66: ("Disgust", 2),
    }

    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: MAD 数据集根目录路径
        """
        self.root = Path(dataset_path)
        if not self.root.exists():
            raise FileNotFoundError(f"数据集路径不存在: {self.root}")

        # 加载全局元数据
        self._load_metadata()

    def _load_metadata(self):
        """加载 session_datetime.csv"""
        meta_path = self.root / "session_datetime.csv"
        if meta_path.exists():
            self.metadata = pd.read_csv(meta_path)
            print(f"[MADLoader] 元数据加载: {len(self.metadata)} 条记录")
        else:
            self.metadata = None
            print("[MADLoader] 未找到 session_datetime.csv")

    def list_subjects(self) -> List[int]:
        """列出所有被试编号"""
        subjects = []
        for d in self.root.iterdir():
            if d.is_dir() and d.name.startswith("subject_"):
                num = d.name.split("_")[1]
                if num.isdigit():
                    subjects.append(int(num))
        return sorted(subjects)

    def list_sessions(self, subject_id: int) -> List[int]:
        """列出指定被试的所有 session 编号"""
        subj_dir = self.root / f"subject_{subject_id:03d}"
        sessions = []
        for d in subj_dir.iterdir():
            if d.is_dir() and d.name.startswith("session_"):
                num = d.name.split("_")[1]
                if num.isdigit():
                    sessions.append(int(num))
        return sorted(sessions)

    def load_ppg_ecg(self, subject_id: int, session_id: int) -> Dict[str, np.ndarray]:
        """
        加载指定被试/session 的 PPG 和 ECG 数据

        Returns:
            {
                "ppg":  np.ndarray,   # PPG 信号 (34Hz)
                "ecg":  np.ndarray,   # ECG 信号 (1000Hz)
                "fs_ppg": float,      # PPG 采样率
                "fs_ecg": float,      # ECG 采样率
                "ppg_time": np.ndarray, # PPG 时间轴 (秒)
                "ecg_time": np.ndarray, # ECG 时间轴 (秒)
            }
        """
        session_dir = self.root / f"subject_{subject_id:03d}" / f"session_{session_id:02d}"

        result = {}

        # --- 加载 PPG (34Hz CSV) ---
        ppg_dir = session_dir / "PPG_BCG"
        ppg_files = list(ppg_dir.glob("*.csv")) if ppg_dir.exists() else []

        if ppg_files:
            # MAD 的 PPG CSV 格式: 可能有时间戳列 + PPG信号列
            df = pd.read_csv(ppg_files[0])
            # 尝试自动识别 PPG 信号列（通常是数值列）
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                ppg_data = df[numeric_cols[0]].values
            else:
                ppg_data = df.iloc[:, 0].values

            result["ppg"] = ppg_data
            result["fs_ppg"] = 34.0  # MAD 标称采样率
            result["ppg_time"] = np.arange(len(ppg_data)) / 34.0
            print(f"  [PPG] {len(ppg_data)} 点, {len(ppg_data)/34:.1f}秒")
        else:
            print(f"  [PPG] 无数据")

        # --- 加载 ECG (1000Hz) ---
        ecg_dir = session_dir / "ECG_EMG_EOG"
        ecg_edf = list(ecg_dir.glob("*.edf")) if ecg_dir.exists() else []
        ecg_txt = list(ecg_dir.glob("*.txt")) if ecg_dir.exists() else []

        if ecg_edf:
            try:
                import pyedflib
                f = pyedflib.EdfReader(str(ecg_edf[0]))
                n_signals = f.signals_in_file
                # ECG 通常是第一个通道
                ecg_data = f.readSignal(0)
                result["ecg"] = ecg_data
                result["fs_ecg"] = f.getSampleFrequency(0)
                result["ecg_time"] = np.arange(len(ecg_data)) / result["fs_ecg"]
                f.close()
                print(f"  [ECG] EDF {len(ecg_data)} 点, {len(ecg_data)/result['fs_ecg']:.1f}秒")
            except ImportError:
                print("  [ECG] pyedflib 未安装，尝试读取 TXT...")
                ecg_edf = []  # fallback
            except Exception as e:
                print(f"  [ECG] EDF 读取失败: {e}")
                ecg_edf = []

        if not ecg_edf and ecg_txt:
            # 回退方案: 读 TXT 文件
            df = pd.read_csv(ecg_txt[0], header=None)
            ecg_data = df.iloc[:, 0].values
            result["ecg"] = ecg_data
            result["fs_ecg"] = 1000.0
            result["ecg_time"] = np.arange(len(ecg_data)) / 1000.0
            print(f"  [ECG] TXT {len(ecg_data)} 点")

        if "ecg" not in result:
            print(f"  [ECG] 无数据 — 只能用 PPG 做 HRV")

        return result

    def load_emotion_labels(self, subject_id: int, session_id: int) -> Dict:
        """
        加载情绪标签（三层标注）

        Returns:
            {
                "stimulus":       List[int],  # 刺激标签 0-6
                "self_assessed":  List[int],  # 自评标签
                "expression":     List[int],  # 表情标签 (0/1/2)
                "emotion_names":  List[str],  # 情绪名称
            }
        """
        labels_dir = self.root / "labels"

        result = {
            "stimulus": [],
            "self_assessed": [],
            "expression": [],
            "emotion_names": [],
        }

        # 尝试多种标签文件命名格式
        possible_files = [
            labels_dir / f"subject_{subject_id:03d}_session_{session_id:02d}.csv",
            labels_dir / f"labels_subject_{subject_id:03d}.csv",
            labels_dir / "stimulus_labels.csv",
            labels_dir / "all_labels.csv",
        ]

        labels_df = None
        for pf in possible_files:
            if pf.exists():
                labels_df = pd.read_csv(pf)
                print(f"  [Labels] 从 {pf.name} 加载")
                break

        if labels_df is None:
            # 遍历 labels 目录找文件
            csv_files = list(labels_dir.glob("*.csv"))
            if csv_files:
                labels_df = pd.read_csv(csv_files[0])
                print(f"  [Labels] 从 {csv_files[0].name} 加载(自动发现)")

        if labels_df is not None:
            # 自动识别标签列
            for col in labels_df.columns:
                col_lower = col.lower()
                if "stimulus" in col_lower:
                    result["stimulus"] = labels_df[col].dropna().astype(int).tolist()
                elif "self" in col_lower or "assessed" in col_lower:
                    result["self_assessed"] = labels_df[col].dropna().astype(int).tolist()
                elif "expression" in col_lower or "express" in col_lower:
                    result["expression"] = labels_df[col].dropna().astype(int).tolist()

            # 映射到情绪名称
            for s in result["stimulus"]:
                result["emotion_names"].append(self.STIMULUS_LABELS.get(s, f"Unknown-{s}"))

        return result

    def load_full_subject(self, subject_id: int, session_id: int) -> Dict:
        """
        一次加载指定被试/session 的所有数据

        Returns:
            {"signals": ..., "labels": ..., "subject_id": ..., "session_id": ...}
        """
        print(f"\n加载 subject_{subject_id:03d} / session_{session_id:02d}...")

        signals = self.load_ppg_ecg(subject_id, session_id)
        labels = self.load_emotion_labels(subject_id, session_id)

        return {
            "signals": signals,
            "labels": labels,
            "subject_id": subject_id,
            "session_id": session_id,
        }

    def load_all(self, max_subjects: Optional[int] = None) -> List[Dict]:
        """
        加载全部可用数据

        Args:
            max_subjects: 限制最多加载几个被试（None=全部）
        """
        all_data = []
        subjects = self.list_subjects()

        if max_subjects:
            subjects = subjects[:max_subjects]

        for sid in subjects:
            for ssid in self.list_sessions(sid):
                try:
                    data = self.load_full_subject(sid, ssid)
                    all_data.append(data)
                except Exception as e:
                    print(f"  [ERROR] subject_{sid:03d}/session_{ssid:02d}: {e}")

        print(f"\n✅ 累计加载 {len(all_data)} 个 session")
        return all_data


# ── 简化版：不依赖 EDF 库的独立数据生成 ──

def generate_synthetic_ecg_ppg(
    duration_sec: float = 300.0,
    hr_mean: float = 75.0,
    hr_std: float = 5.0,
    emotion_label: str = "Neutral",
    fs_ecg: float = 1000.0,
    fs_ppg: float = 34.0,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    生成合成 ECG + PPG 数据（用于在没有MAD数据集时测试算法）

    不同情绪对应不同的 HRV 参数：
      Neutral:  HR=70, SDNN=55, RMSSD=45
      Anxiety:  HR=90, SDNN=28, RMSSD=22  ← 心率高、HRV低
      Happy:    HR=75, SDNN=50, RMSSD=40
      Sad:      HR=65, SDNN=40, RMSSD=35

    Returns:
        {"ecg": array, "ppg": array, "fs_ecg": float, "fs_ppg": float,
         "ecg_time": array, "ppg_time": array, "true_hrv": dict}
    """
    rng = np.random.default_rng(seed)

    # 情绪 → 生理参数映射
    EMOTION_PARAMS = {
        "Neutral":  {"hr": 70, "sdnn": 55, "rmssd": 45, "lf_hf": 1.8},
        "Happy":    {"hr": 75, "sdnn": 50, "rmssd": 40, "lf_hf": 2.2},
        "Anxiety":  {"hr": 90, "sdnn": 28, "rmssd": 22, "lf_hf": 3.5},
        "Fearful":  {"hr": 95, "sdnn": 25, "rmssd": 20, "lf_hf": 4.0},
        "Sad":      {"hr": 65, "sdnn": 40, "rmssd": 35, "lf_hf": 1.5},
        "Angry":    {"hr": 85, "sdnn": 32, "rmssd": 26, "lf_hf": 3.0},
        "Disgust":  {"hr": 72, "sdnn": 38, "rmssd": 30, "lf_hf": 2.0},
    }

    params = EMOTION_PARAMS.get(emotion_label, EMOTION_PARAMS["Neutral"])
    target_hr = rng.normal(params["hr"], 3)
    target_sdnn = params["sdnn"] * rng.uniform(0.85, 1.15)

    # 生成 NN 间隔序列 (ms)
    n_beats = int(duration_sec * target_hr / 60)
    mean_nn = 60000.0 / target_hr  # 平均 NN 间隔 (ms)

    # HRV 模拟: NN = mean + 随机波动(正态分布, 标准差 = SDNN)
    nn_intervals = rng.normal(mean_nn, target_sdnn, n_beats)
    nn_intervals = np.clip(nn_intervals, 300, 2000)  # 生理合理范围

    # 生成 RR 间隔序列的累积时间 (秒)
    nn_cumsum = np.cumsum(nn_intervals) / 1000.0
    # 只取在 duration_sec 内的部分
    valid_idx = nn_cumsum <= duration_sec
    nn_intervals = nn_intervals[valid_idx]
    beat_times = nn_cumsum[valid_idx]

    # ── ECG 合成 (1000Hz) ──
    n_ecg = int(duration_sec * fs_ecg)
    t_ecg = np.arange(n_ecg) / fs_ecg
    ecg = np.zeros(n_ecg)

    # 在每个 R 峰位置放一个 QRS 波形
    for bt in beat_times:
        idx = int(bt * fs_ecg)
        if idx < n_ecg:
            # QRS 复合波: 高斯脉冲
            half_width = 20  # samples (~20ms)
            start = max(0, idx - half_width)
            end = min(n_ecg, idx + half_width)
            t_local = np.arange(end - start) - (idx - start)
            qrs = np.exp(-0.5 * (t_local / 8) ** 2) * 1.5
            # 加 T 波
            t_start = idx + 30
            t_end = min(n_ecg, idx + 120)
            if t_end > t_start:
                t_local_t = np.arange(t_end - t_start)
                t_wave = np.exp(-0.5 * ((t_local_t - 30) / 15) ** 2) * 0.4
                ecg[t_start:t_end] += t_wave[:len(ecg[t_start:t_end])]
            ecg[start:end] += qrs[:len(ecg[start:end])]

    # 加噪声
    ecg += rng.normal(0, 0.05, n_ecg)

    # ── PPG 合成 (34Hz) ──
    n_ppg = int(duration_sec * fs_ppg)
    t_ppg = np.arange(n_ppg) / fs_ppg
    ppg = np.zeros(n_ppg)

    # PPG = 低频基线 + 心跳脉动
    for bt in beat_times:
        idx = int(bt * fs_ppg)
        if idx < n_ppg:
            pulse_width = int(0.3 * fs_ppg)  # ~300ms
            start = max(0, idx - pulse_width // 2)
            end = min(n_ppg, idx + pulse_width)
            t_local = np.arange(end - start) - (idx - start)
            pulse = np.exp(-0.5 * (t_local / (pulse_width / 3)) ** 2) * 0.8
            ppg[start:end] += pulse[:len(ppg[start:end])]

    # 加基线漂移模拟呼吸
    resp_freq = rng.uniform(0.2, 0.35)  # 12-21 bpm
    baseline = 0.15 * np.sin(2 * np.pi * resp_freq * t_ppg) + 1.0
    ppg = ppg * 0.5 + baseline
    ppg += rng.normal(0, 0.03, n_ppg)

    return {
        "ecg": ecg,
        "ppg": ppg,
        "fs_ecg": fs_ecg,
        "fs_ppg": fs_ppg,
        "ecg_time": t_ecg,
        "ppg_time": t_ppg,
        "true_hrv": {
            "hr_mean": target_hr,
            "sdnn": target_sdnn,
            "rmssd": params["rmssd"],
            "lf_hf": params["lf_hf"],
            "emotion": emotion_label,
        },
        "beat_times": beat_times,
        "nn_intervals": nn_intervals,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("MAD 数据集加载器 - 快速测试")
    print("=" * 60)

    # 测试 1: 合成数据生成
    print("\n[测试1] 合成数据生成")
    for emotion in ["Neutral", "Anxiety", "Happy", "Sad"]:
        data = generate_synthetic_ecg_ppg(
            duration_sec=60.0,
            emotion_label=emotion,
            seed=42
        )
        hrv = data["true_hrv"]
        print(f"  {emotion:8s} | HR={hrv['hr_mean']:.0f} SDNN={hrv['sdnn']:.0f} RMSSD={hrv['rmssd']:.0f}")

    print("\n✅ 合成数据生成测试通过!")
    print("\n💡 使用说明:")
    print("  1. 有MAD数据集: loader = MADLoader('path/to/dataset'); data = loader.load_full_subject(1,1)")
    print("  2. 没有数据集: data = generate_synthetic_ecg_ppg(emotion_label='Anxiety')")
