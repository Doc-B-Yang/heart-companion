"""
HRV 特征提取器
==============
从 ECG (金标准) 和 PPG (模拟 MAX30102) 信号中提取心率变异性特征。

核心指标（对应 PPT 中 STEP 02）:
  SDNN  — NN间期标准差 (ms)       — 反映总自主神经活性
  RMSSD — 相邻NN差值均方根 (ms)   — 反映副交感(迷走神经)活性
  pNN50 — >50ms差值百分比 (%)     — 短时自主神经调节
  LF/HF — 低频/高频功率比         — 交感-副交感平衡

输入: ECG 或 PPG 一维信号 + 采样率
输出: HRV 特征字典

用法:
  from mad_hrv_extractor import HRVExtractor
  extractor = HRVExtractor()
  features = extractor.extract_from_ecg(ecg_signal, fs=1000, window_sec=300)
  features = extractor.extract_from_ppg(ppg_signal, fs=34, window_sec=300)
"""

import numpy as np
from scipy import signal, interpolate
from scipy.signal import find_peaks, welch, savgol_filter, butter, filtfilt
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional


class HRVExtractor:
    """
    心率变异性 (HRV) 特征提取器

    支持两种输入:
    1. ECG 信号 (1000Hz 金标准)
    2. PPG 信号 (34Hz, 模拟 MAX30102)
    """

    # 频域参数
    VLF_BAND = (0.0033, 0.04)   # 极低频 (Hz)
    LF_BAND  = (0.04, 0.15)     # 低频 (Hz) — 反映交感+副交感
    HF_BAND  = (0.15, 0.40)     # 高频 (Hz) — 反映副交感

    # 合理的 NN 间隔范围 (ms)
    NN_MIN = 300   # 对应心率 200 bpm
    NN_MAX = 2000  # 对应心率 30 bpm

    def __init__(self, window_sec: float = 300.0):
        """
        Args:
            window_sec: HRV 计算窗口 (秒)，标准为 5 分钟 = 300s
        """
        self.window_sec = window_sec

    # ═══════════════════════════════════════════════════════════
    # ECG 处理流程
    # ═══════════════════════════════════════════════════════════

    def extract_from_ecg(self, ecg_signal: np.ndarray, fs: float,
                         window_sec: Optional[float] = None) -> Dict:
        """
        从 ECG 信号中提取 HRV 特征 (金标准)

        Args:
            ecg_signal: 一维 ECG 信号
            fs: 采样率 (Hz), MAD 数据集为 1000Hz
            window_sec: 窗口长度，默认使用 __init__ 设置的值

        Returns:
            HRV 特征字典
        """
        win = window_sec or self.window_sec
        n_samples = int(min(win * fs, len(ecg_signal)))
        ecg = ecg_signal[:n_samples]

        print(f"  [ECG→HRV] {len(ecg)} 点 @ {fs}Hz = {n_samples/fs:.1f}秒")

        # Step 1: 检测 R 波峰值 (QRS 复合波)
        r_peaks = self._detect_r_peaks(ecg, fs)

        # Step 2: 计算 NN 间隔序列 (ms)
        nn_intervals = self._r_peaks_to_nn(r_peaks, fs)
        nn_intervals = self._filter_abnormal_nn(nn_intervals)

        if len(nn_intervals) < 3:
            print(f"  [ECG] 有效 NN 间隔不足 (<3)")
            return self._empty_features()

        # Step 3: 时域特征
        hr_mean = 60000.0 / np.mean(nn_intervals)  # bpm
        sdnn = np.std(nn_intervals)
        rmssd = np.sqrt(np.mean(np.diff(nn_intervals) ** 2))
        pnn50 = np.sum(np.abs(np.diff(nn_intervals)) > 50) / (len(nn_intervals) - 1) * 100

        # Step 4: 频域特征
        lf_hf = self._calculate_lf_hf(nn_intervals)

        # Step 5: 非线性特征
        sd1, sd2 = self._poincare_features(nn_intervals)

        return {
            "source": "ECG",
            "fs": fs,
            "window_sec": win,
            "n_beats": len(r_peaks),
            "n_valid_nn": len(nn_intervals),
            "hr_bpm": round(hr_mean, 1),
            "sdnn_ms": round(sdnn, 1),
            "rmssd_ms": round(rmssd, 1),
            "pnn50_pct": round(pnn50, 1),
            "lf_hf": round(lf_hf, 2),
            "sd1_ms": round(sd1, 1),
            "sd2_ms": round(sd2, 1),
            "sd_ratio": round(sd2 / sd1, 2) if sd1 > 0 else 0,
            "nn_intervals": nn_intervals,
            "r_peaks_indices": r_peaks,
        }

    def _detect_r_peaks(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """
        R 峰检测算法

        方法: 带通滤波 → 求导+平方 → 移动窗口积分 → 自适应阈值
        这是 Pan-Tompkins 算法的简化版本

        Args:
            ecg: 一维 ECG 信号
            fs: 采样率

        Returns:
            R 峰索引数组 (在原始信号中的位置)
        """
        # 1. 带通滤波 (5-15 Hz, QRS 能量集中的频段)
        nyq = fs / 2
        b, a = butter(2, [5.0 / nyq, 15.0 / nyq], btype='band')
        ecg_filtered = filtfilt(b, a, ecg)

        # 2. 微分 (增强斜率变化)
        ecg_diff = np.diff(ecg_filtered)
        ecg_diff = np.append(ecg_diff, 0)

        # 3. 平方 (放大 QRS)
        ecg_sq = ecg_diff ** 2

        # 4. 移动窗口积分 (平滑)
        window_size = int(0.15 * fs)  # 150ms 窗口
        ecg_mwi = np.convolve(ecg_sq, np.ones(window_size) / window_size, mode='same')

        # 5. 自适应阈值检测峰值
        threshold = np.mean(ecg_mwi) * 0.6
        peaks, _ = find_peaks(ecg_mwi, height=threshold,
                             distance=int(0.25 * fs))  # 最小间距250ms (心率<240)

        # 6. 回溯到原信号找精确 R 峰位置 (在原信号±50ms内找最大值)
        refined_peaks = []
        search_win = int(0.05 * fs)  # 50ms
        for p in peaks:
            start = max(0, p - search_win)
            end = min(len(ecg), p + search_win)
            refined_peaks.append(start + np.argmax(ecg[start:end]))

        return np.array(refined_peaks)

    def _r_peaks_to_nn(self, r_peaks: np.ndarray, fs: float) -> np.ndarray:
        """R 峰索引 → NN 间隔 (ms)"""
        if len(r_peaks) < 2:
            return np.array([])
        rr_samples = np.diff(r_peaks)
        nn_intervals = rr_samples / fs * 1000  # 转换为 ms
        return nn_intervals

    def _filter_abnormal_nn(self, nn: np.ndarray) -> np.ndarray:
        """过滤异常的 NN 间隔 (伪迹)"""
        if len(nn) == 0:
            return nn
        mask = (nn >= self.NN_MIN) & (nn <= self.NN_MAX)
        return nn[mask]

    # ═══════════════════════════════════════════════════════════
    # PPG 处理流程 (模拟 MAX30102)
    # ═══════════════════════════════════════════════════════════

    def extract_from_ppg(self, ppg_signal: np.ndarray, fs: float,
                         window_sec: Optional[float] = None) -> Dict:
        """
        从 PPG 信号中提取 HRV 特征 (模拟 MAX30102 输出)

        与 ECG 的主要区别:
        - PPG 采样率低 (34Hz vs 1000Hz)，峰值检测精度较低
        - PPG 有基线漂移 (呼吸导致)，需要额外滤除
        - 受运动伪迹影响大

        Args:
            ppg_signal: 一维 PPG 信号
            fs: 采样率 (Hz), MAD 数据集为 34Hz
            window_sec: 窗口长度

        Returns:
            HRV 特征字典
        """
        win = window_sec or self.window_sec
        n_samples = int(min(win * fs, len(ppg_signal)))
        ppg = ppg_signal[:n_samples]

        print(f"  [PPG→HRV] {len(ppg)} 点 @ {fs}Hz = {n_samples/fs:.1f}秒")

        # Step 1: 带通滤波去除基线漂移和噪声
        ppg_filtered = self._preprocess_ppg(ppg, fs)

        # Step 2: 检测 PPG 收缩期峰值
        systolic_peaks = self._detect_ppg_peaks(ppg_filtered, fs)

        # Step 3: 计算脉冲间隔 (PPI)
        ppi = self._r_peaks_to_nn(systolic_peaks, fs)
        ppi = self._filter_abnormal_nn(ppi)

        if len(ppi) < 3:
            print(f"  [PPG] 有效脉冲间隔不足 (<3)")
            return self._empty_features()

        # Step 4: 时域特征
        hr_mean = 60000.0 / np.mean(ppi)
        sdnn = np.std(ppi)
        rmssd = np.sqrt(np.mean(np.diff(ppi) ** 2))
        pnn50 = np.sum(np.abs(np.diff(ppi)) > 50) / max(1, len(ppi) - 1) * 100

        # Step 5: 频域特征
        lf_hf = self._calculate_lf_hf(ppi)

        # Step 6: 非线性特征
        sd1, sd2 = self._poincare_features(ppi)

        # 信号质量评估
        quality = self._assess_ppg_quality(ppg_filtered, systolic_peaks)

        return {
            "source": "PPG",
            "fs": fs,
            "window_sec": win,
            "n_peaks": len(systolic_peaks),
            "n_valid_ppi": len(ppi),
            "hr_bpm": round(hr_mean, 1),
            "sdnn_ms": round(sdnn, 1),
            "rmssd_ms": round(rmssd, 1),
            "pnn50_pct": round(pnn50, 1),
            "lf_hf": round(lf_hf, 2),
            "sd1_ms": round(sd1, 1),
            "sd2_ms": round(sd2, 1),
            "sd_ratio": round(sd2 / sd1, 2) if sd1 > 0 else 0,
            "signal_quality": round(quality, 2),  # 0-1
            "ppi_intervals": ppi,
            "ppg_peaks": systolic_peaks,
        }

    def _preprocess_ppg(self, ppg: np.ndarray, fs: float) -> np.ndarray:
        """PPG 信号预处理: 去基线 + 带通滤波"""
        # Savitzky-Golay 平滑 (保留波形形状)
        window = min(int(0.25 * fs), 7)  # 250ms 窗口
        if window % 2 == 0:
            window += 1  # 必须是奇数
        ppg_smooth = savgol_filter(ppg, window, 2)

        # 带通滤波 0.5-10 Hz
        nyq = fs / 2
        b, a = butter(2, [0.5 / nyq, 10.0 / nyq], btype='band')
        ppg_filtered = filtfilt(b, a, ppg_smooth)

        return ppg_filtered

    def _detect_ppg_peaks(self, ppg: np.ndarray, fs: float) -> np.ndarray:
        """检测 PPG 信号中的收缩期峰值 (对应心跳)"""
        # 使用 scipy find_peaks
        min_distance = int(0.35 * fs)  # 最小心率 ~170 bpm
        height_threshold = np.mean(ppg) + 0.3 * np.std(ppg)

        peaks, properties = find_peaks(
            ppg,
            height=height_threshold,
            distance=min_distance,
            prominence=0.1 * np.std(ppg)
        )

        return peaks

    def _assess_ppg_quality(self, ppg: np.ndarray, peaks: np.ndarray) -> float:
        """
        评估 PPG 信号质量 (模拟 ESP32 固件中的 signalQuality)

        评估标准:
        1. 峰值数量是否在合理范围内
        2. 峰值间距的均匀性
        3. 信号幅度是否足够

        Returns:
            0-1 之间的质量评分
        """
        if len(peaks) < 2:
            return 0.0

        # 信号幅度
        amplitude = np.std(ppg)
        if amplitude < 0.02:
            return 0.0

        # 峰值间距均匀性 (变异系数越小越好)
        ppi = np.diff(peaks)
        if len(ppi) > 1:
            cv = np.std(ppi) / np.mean(ppi)
            uniformity_score = max(0, 1 - cv)
        else:
            uniformity_score = 0.5

        # 综合评分
        amplitude_score = min(1.0, amplitude * 10)
        quality = 0.4 * amplitude_score + 0.6 * uniformity_score
        return float(np.clip(quality, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════
    # 通用频域分析
    # ═══════════════════════════════════════════════════════════

    def _calculate_lf_hf(self, nn_intervals: np.ndarray,
                         fs_resample: float = 4.0) -> float:
        """
        计算 LF/HF 比值 (频域 HRV 核心指标)

        方法:
        1. 将不等间隔的 NN 序列重新插值为等间隔 (4Hz)
        2. Welch 周期图法估计功率谱
        3. 积分 LF 和 HF 频段功率

        Args:
            nn_intervals: NN 间隔序列 (ms)
            fs_resample: 重采样频率 (Hz)

        Returns:
            LF/HF 比值
        """
        if len(nn_intervals) < 10:
            return 0.0

        # Step 1: 构建 NN 序列的时间轴 (从 NN 间隔累积)
        nn_times = np.cumsum(nn_intervals) / 1000.0  # 秒

        # Step 2: 重采样为等间隔
        t_new = np.arange(0, nn_times[-1], 1.0 / fs_resample)
        f_interp = interp1d(nn_times, nn_intervals, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
        nn_resampled = f_interp(t_new)

        # Step 3: 去趋势 (去除缓慢变化)
        nn_detrended = nn_resampled - np.mean(nn_resampled)

        # Step 4: Welch 周期图
        nperseg = min(256, len(nn_detrended))
        if nperseg < 64:
            return 0.0

        freqs, psd = welch(nn_detrended, fs=fs_resample, nperseg=nperseg,
                          scaling='density')

        # Step 5: 积分各频段
        lf_power = self._band_power(freqs, psd, self.LF_BAND[0], self.LF_BAND[1])
        hf_power = self._band_power(freqs, psd, self.HF_BAND[0], self.HF_BAND[1])

        if hf_power > 0:
            return lf_power / hf_power
        return 0.0

    @staticmethod
    def _band_power(freqs: np.ndarray, psd: np.ndarray,
                    low: float, high: float) -> float:
        """积分某频段的功率"""
        mask = (freqs >= low) & (freqs <= high)
        if np.sum(mask) == 0:
            return 0.0
        return float(np.trapezoid(psd[mask], freqs[mask]))

    # ═══════════════════════════════════════════════════════════
    # Poincaré 图 (非线性 HRV)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _poincare_features(nn: np.ndarray) -> Tuple[float, float]:
        """
        Poincaré 散点图特征

        SD1: 短轴标准差 — 反映短期(副交感)变异性
        SD2: 长轴标准差 — 反映长期(整体)变异性
        """
        if len(nn) < 2:
            return 0.0, 0.0

        nn_i = nn[:-1]
        nn_ip1 = nn[1:]

        # 旋转 45°
        nn_diff = nn_ip1 - nn_i
        nn_sum = nn_ip1 + nn_i

        sd1 = np.sqrt(np.var(nn_diff) / 2)
        sd2 = np.sqrt(np.var(nn_sum) / 2)

        return float(sd1), float(sd2)

    # ═══════════════════════════════════════════════════════════
    # 工具方法
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _empty_features() -> Dict:
        """返回空的 HRV 特征 (无效信号时)"""
        return {
            "hr_bpm": 0, "sdnn_ms": 0, "rmssd_ms": 0,
            "pnn50_pct": 0, "lf_hf": 0,
            "sd1_ms": 0, "sd2_ms": 0, "sd_ratio": 0,
            "signal_quality": 0,
        }

    def compare_ecg_ppg(self, ecg_features: Dict,
                        ppg_features: Dict) -> Dict:
        """
        对比 ECG (金标准) 和 PPG 的 HRV 特征差异

        用于评估你的 PPG 传感器能多准确地反映真实 HRV

        Returns:
            {
                "hr_error_pct": ...,     # 心率误差百分比
                "sdnn_error_pct": ...,   # SDNN 误差百分比
                "rmssd_error_pct": ...,  # RMSSD 误差百分比
                "agreement": "excellent|good|moderate|poor"
            }
        """
        hr_ecg = ecg_features.get("hr_bpm", 0)
        hr_ppg = ppg_features.get("hr_bpm", 0)
        sdnn_ecg = ecg_features.get("sdnn_ms", 0)
        sdnn_ppg = ppg_features.get("sdnn_ms", 0)
        rmssd_ecg = ecg_features.get("rmssd_ms", 0)
        rmssd_ppg = ppg_features.get("rmssd_ms", 0)

        def pct_err(gold, test):
            if gold == 0:
                return 100.0
            return abs(test - gold) / gold * 100

        hr_err = pct_err(hr_ecg, hr_ppg)
        sdnn_err = pct_err(sdnn_ecg, sdnn_ppg)
        rmssd_err = pct_err(rmssd_ecg, rmssd_ppg)

        avg_err = (hr_err + abs(sdnn_err) + abs(rmssd_err)) / 3

        if avg_err < 5:
            agreement = "excellent"
        elif avg_err < 10:
            agreement = "good"
        elif avg_err < 20:
            agreement = "moderate"
        else:
            agreement = "poor"

        return {
            "hr_error_pct": round(hr_err, 1),
            "sdnn_error_pct": round(sdnn_err, 1),
            "rmssd_error_pct": round(rmssd_err, 1),
            "avg_error_pct": round(avg_err, 1),
            "agreement": agreement,
        }


# ── 快速测试 ──
if __name__ == "__main__":
    print("=" * 60)
    print("HRV 特征提取器 - 快速测试")
    print("=" * 60)

    from mad_dataset_loader import generate_synthetic_ecg_ppg

    # 生成合成数据
    data = generate_synthetic_ecg_ppg(
        duration_sec=300.0,
        emotion_label="Anxiety",
        seed=42
    )

    extractor = HRVExtractor(window_sec=300.0)

    # ECG → HRV (金标准)
    print("\n[ECG → HRV] 金标准:")
    ecg_feat = extractor.extract_from_ecg(data["ecg"], fs=data["fs_ecg"])
    for k, v in ecg_feat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"  {k:15s}: {v}")

    # PPG → HRV (模拟 MAX30102)
    print("\n[PPG → HRV] 模拟 MAX30102:")
    ppg_feat = extractor.extract_from_ppg(data["ppg"], fs=data["fs_ppg"])
    for k, v in ppg_feat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            print(f"  {k:15s}: {v}")

    # 对比
    print("\n[ECG vs PPG 对比]:")
    comparison = extractor.compare_ecg_ppg(ecg_feat, ppg_feat)
    for k, v in comparison.items():
        print(f"  {k:18s}: {v}")

    # 真实值
    true = data["true_hrv"]
    print(f"\n[真实值 (合成参数)]:")
    for k, v in true.items():
        print(f"  {k:15s}: {v}")
