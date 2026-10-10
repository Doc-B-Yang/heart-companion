# -*- coding: utf-8 -*-
"""
HRV 特征提取模块（v2 · 专业化增强版）
=====================================
对应计划书 §4.2「算法子模块 1」/ PPT STEP 02。

相对 v1 的增强（依据《文献》文件夹，逐条可追溯）：
--------------------------------------------------------------------------------
1. 【频域方法学修正 —— 重要】
   v1 以「心跳序号」为时间轴做 Lomb-Scargle，频率轴物理意义错误：
   横轴单位实为 cycle/beat，按 850 ms 平均 IBI 折算，真实频率被缩放约
   0.85 倍（即标称 LF 0.04–0.15 Hz 实际落在 0.047–0.176 Hz）。
   v2 改为 HRV 频域分析的标准流程（Task Force 1996 及文献2 采用）：
     IBI 序列 → 构建累积时间轴 → 4 Hz 等间隔线性插值重采样（tachogram）
              → Hann 窗分段（默认 256 点 / 50% 重叠）→ Welch 平均周期图
   重采样到 4 Hz 使 Nyquist = 2 Hz，远超 HF 上限 0.4 Hz，无混叠风险。
   参考：文献2《基于心率变异性特征和 PCA-SVR 的 PAD 维度情感预测分析》
        （北京化工大学学报 2021, 48(5)）—— 明确采用 Welch 功率谱。

2. 【频带标准化】
   VLF 0.003–0.04 / LF 0.04–0.15 / HF 0.15–0.40 / 总功率 0.003–0.40 Hz。
   LF 与 HF 边界在两篇文献中完全一致（0.15 Hz 为分界）；
   VLF 采信文献2 的标准定义（文献1 原文 VLF 写作 0.04–0.15 与 LF 重叠，
   判定为原文排版错误，不予采信）。

3. 【时域特征扩展】v1 仅 SDNN/RMSSD/pNN50，v2 补齐文献1 的 9 项 + 文献2 的 8 项：
   SDSD、NN50（绝对计数）、SDNN index（1 min 分段）、IBI 极差 max−min、
   平均 IBI、心率标准差 SD_HR、HRV 三角指数（RR 间期总个数 ÷ 直方图高度）。

4. 【非线性特征扩展】文献2 的 4 项：
   * Poincaré SD1/SD2 使用旋转坐标标准差（对 (IBI_i, IBI_i+1) 散点做 45° 正交
     旋转后取两方向标准差），不再用 v1 的 SD1 = RMSSD/√2 近似式；
     同时保留 sd1_rmssd_ms 作为对照（近似式在文献中被广泛使用，二者差异可监控）。
   * 样本熵 SampEn（m=2, r=0.2×SD，Chebyshev 距离）
   * DFA 标度指数 α1（4–16 拍，短程）与 α2（16–64 拍，长程）

5. 【个体基线相对化】新增 relative_features()，实现文献1 的「相对特征值法」：
       相对特征值 = (平静基线值 − 当前值) / 平静基线值
   文献1 在其数据集上比较了相对特征；本项目仍需独立真实数据验证。
   仿真链路的分类结果不构成真实情绪准确率，也不能与论文互证。
   MAX30102 得到的是脉搏间期 PRV；短窗口频域值不能解释成可靠的自主神经指标。

6. 【信号质量门禁】新增 quality_metrics()：
   RR 抖动指数 jitter = RMSSD / mean(IBI)（与 ESP32 固件 hrv.py 的 rr_jitter()
   定义完全一致），以及 ectopic_ratio（伪迹剔除比例）。
   依据实测结论：「HR 正常但 HRV 是假的」是 PPG 最隐蔽的失效模式，
   HR 只看峰间距均值（抖动被平均掉），HRV 看逐拍差分（抖动全额进入），
   二者可信度解耦。低质量数据必须被下游拒绝，而不是硬算出一个数。

7. 【向后兼容】所有 v1 键名与数值语义原样保留（含 sdnn_ms/rmssd_ms/pnn50_pct
   长名双写修复），纯增量扩展，既有调用方与测试无需改动。
   numpy 可用时自动加速频域与非线性的 O(n²) 部分，不可用时纯 Python 降级，
   零硬依赖设计不变。

参考文献
--------
[1] 易慧, 陈瑞娟, 邓光华, 等. 基于心率变异性的情绪识别研究[J].
    生物医学工程研究, 2020, 39(2): 128-132.
[2] 韩永明, 张明星, 耿志强. 基于心率变异性特征和 PCA-SVR 的 PAD 维度
    情感预测分析[J]. 北京化工大学学报(自然科学版), 2021, 48(5): 102-110.
[3] Task Force of ESC/NASPE. Heart rate variability: standards of measurement,
    physiological interpretation and clinical use[J]. Circulation, 1996, 93(5).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .. import config

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False


class HRVExtractor:
    """HRV 特征提取器（纯 Python 可用，numpy 可选加速）

    典型用法::

        ex = HRVExtractor(window_sec=300)
        feats = ex.extract_from_ibi(ibi_list)          # 全部特征
        rel   = HRVExtractor.relative_features(feats, baseline)   # 个体相对化
        q     = ex.quality_metrics(ibi_list, feats)    # 质量门禁
    """

    # ------------------------------------------------------------ 常量定义
    #: 频带定义（Hz）—— 文献2 标准划分
    BAND_VLF: Tuple[float, float] = (0.003, 0.04)
    BAND_LF: Tuple[float, float] = (0.04, 0.15)
    BAND_HF: Tuple[float, float] = (0.15, 0.40)
    BAND_TOTAL: Tuple[float, float] = (0.003, 0.40)

    #: 重采样频率（Hz）—— 4 Hz 远高于 HF 上限，无混叠
    RESAMPLE_HZ: float = 4.0
    #: Welch 分段长度（样本点，@4Hz 即 64 s）与重叠率
    WELCH_NPERSEG: int = 256
    WELCH_OVERLAP: float = 0.5

    #: 样本熵参数（m=2, r=0.2×SD 为 HRV 文献通用取值）
    SAMPEN_M: int = 2
    SAMPEN_R_RATIO: float = 0.2

    #: DFA 标度分段（文献通用 α1: 4–16 拍, α2: 16–64 拍）
    DFA_ALPHA1_RANGE: Tuple[int, int] = (4, 16)
    DFA_ALPHA2_RANGE: Tuple[int, int] = (16, 64)

    #: SDNN index 分段时长（秒）—— 文献1/2 均为 1 min
    SDNN_INDEX_SEG_SEC: float = 60.0

    #: HRV 三角指数直方图分箱宽度（ms）—— Task Force 建议 7.8125 ms
    HRV_INDEX_BIN_MS: float = 7.8125

    # ------------------------------------------------------------------
    # 【兼容修复】特征键名双写（v1 引入，v2 保留）
    # 本模块历史上存在两套命名：
    #   规范名(长名) sdnn_ms / rmssd_ms / pnn50_pct —— time_domain 输出、
    #                 EMOTION_RULES、FEATURE_WEIGHTS、ML_FEATURE_NAMES 用它；
    #   契约名(短名) sdnn / rmssd / pnn50            —— ESP32 数据包用它。
    # 早期实现把长名 pop 成短名只留一套，导致规则层 `if k in feats` 静默跳过
    # SDNN/RMSSD →「平静」与「疲劳」永远同分，ML 层 8 维有 3 维恒 0。
    # 修复方式：两套键名同时输出（纯增量，不删除任何既有键）。
    # ------------------------------------------------------------------
    _ALIASES = (("sdnn", "sdnn_ms"),
                ("rmssd", "rmssd_ms"),
                ("pnn50", "pnn50_pct"))

    @classmethod
    def add_aliases(cls, feats: Dict) -> Dict:
        for short, long in cls._ALIASES:
            if long not in feats and feats.get(short) is not None:
                feats[long] = feats[short]
        return feats

    def __init__(self, window_sec: float = 300.0,
                 resample_hz: Optional[float] = None,
                 welch_nperseg: Optional[int] = None):
        self.window_sec = float(window_sec)
        self.resample_hz = float(resample_hz or self.RESAMPLE_HZ)
        self.welch_nperseg = int(welch_nperseg or self.WELCH_NPERSEG)

    # ==================================================================
    # 伪迹过滤
    # ==================================================================
    @staticmethod
    def _median_outlier_reject(values: List[float],
                               tol: float = 0.30) -> List[float]:
        """迭代剔除偏离中位数超过 tol 的离群值（Kubios 式伪迹处理）。

        为什么必须做: RMSSD = √(mean(ΔIBI²)) 由**最大**的几个差值主导，
        单次 250ms 的运动伪迹就能把静息 RMSSD 从 50 抬到 85(实测)，
        进而把情绪判成完全相反的类别。正常呼吸性窦性节律不齐(RSA)
        波动约 ±20%, 不会被 30% 门限误剔。
        """
        vals = list(values)
        while len(vals) >= 5:
            med = sorted(vals)[len(vals) // 2]
            if med <= 0:
                break
            worst_i, worst_d = -1, 0.0
            for i, x in enumerate(vals):
                d = abs(x - med)
                if d > tol * med and d > worst_d:
                    worst_d, worst_i = d, i
            if worst_i < 0:
                break
            vals.pop(worst_i)
        return vals

    @staticmethod
    def clean_ibi(ibi_list: List[float], ibi_min: float = 300.0,
                  ibi_max: float = 2000.0) -> List[float]:
        """剔除生理范围之外的异常 IBI（运动伪迹/检测错误）

        三级过滤，逐级收紧：
          ① 生理范围 [ibi_min, ibi_max]
          ② 相邻差值 > 300 ms 的跳变（检测器错位/漏搏）
          ③ 迭代中位数离群剔除（拦住"幅度 <300ms 但足以污染 RMSSD"的尖峰）
        """
        cleaned = [v for v in ibi_list if ibi_min <= v <= ibi_max]
        if len(cleaned) >= 2:
            prev = cleaned[0]
            kept = [prev]
            for v in cleaned[1:]:
                if abs(v - prev) <= 300.0:
                    kept.append(v)
                prev = v
            cleaned = kept
        return HRVExtractor._median_outlier_reject(cleaned)

    @classmethod
    def clean_ibi_report(cls, ibi_list: List[float], ibi_min: float = 300.0,
                         ibi_max: float = 2000.0) -> Dict:
        """带诊断信息的清理（供质量门禁使用）。

        Returns:
            {"cleaned": [...], "n_in": 输入拍数, "n_out": 保留拍数,
             "rejected": 剔除拍数, "ectopic_ratio": 剔除比例}
        """
        n_in = len(ibi_list)
        cleaned = cls.clean_ibi(ibi_list, ibi_min, ibi_max)
        n_out = len(cleaned)
        rejected = max(0, n_in - n_out)
        return {
            "cleaned": cleaned,
            "n_in": n_in,
            "n_out": n_out,
            "rejected": rejected,
            "ectopic_ratio": round(rejected / n_in, 4) if n_in else 0.0,
        }

    # ==================================================================
    # 时域特征（文献1 的 9 项 + 文献2 的 8 项）
    # ==================================================================
    def time_domain(self, ibi_list: List[float]) -> Dict[str, float]:
        """时域特征：SDNN / RMSSD / SDSD / pNN50 / NN50 / SDNN index /
        IBI 极差 / 平均 IBI / 心率标准差 / HRV 三角指数 / Poincaré(SD1,SD2)
        """
        empty = {"sdnn_ms": 0.0, "rmssd_ms": 0.0, "pnn50_pct": 0.0,
                 "mean_hr_bpm": 0.0, "sd1_ms": 0.0, "sd2_ms": 0.0,
                 "sd_ratio": 0.0, "n_beats": len(ibi_list),
                 "sdsd_ms": 0.0, "nn50": 0, "sdnn_index_ms": 0.0,
                 "ibi_range_ms": 0.0, "mean_ibi_ms": 0.0,
                 "sd_hr_bpm": 0.0, "hrvi": 0.0}
        if len(ibi_list) < 3:
            return empty

        def _mean(xs: Sequence[float]) -> float:
            return sum(xs) / len(xs)

        n = len(ibi_list)
        mean_ibi = _mean(ibi_list)
        var = _mean([(v - mean_ibi) ** 2 for v in ibi_list])
        sdnn = math.sqrt(var)

        diffs = [ibi_list[i + 1] - ibi_list[i] for i in range(n - 1)]
        abs_diffs = [abs(d) for d in diffs]
        rmssd = math.sqrt(_mean([d ** 2 for d in diffs]))
        # SDSD: 相邻 RR 差值的标准差（针对 ΔRR 序列，与 RMSSD 不同）
        #   文献1/2 均将其列为独立特征：RMSSD 是 ΔRR 的均方根，
        #   SDSD 是 ΔRR 的标准差（扣掉均值），反映 ΔRR 自身的离散程度。
        sdsd = math.sqrt(_mean([(d - _mean(diffs)) ** 2 for d in diffs])) \
            if len(diffs) >= 2 else 0.0
        nn50 = sum(1 for d in abs_diffs if d > 50.0)
        pnn50 = nn50 / len(abs_diffs) * 100.0

        mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0
        # SD_HR: 逐拍瞬时心率的标准差（文献2 第 4 项独立特征）
        #   注：IBI 域标准差(SDNN) 与 HR 域标准差并非单调等价换算，
        #   瞬时心率 60/IBI 是非线性变换，故单列一维。
        inst_hr = [60000.0 / v for v in ibi_list if v > 0]
        sd_hr = math.sqrt(_mean([(h - _mean(inst_hr)) ** 2 for h in inst_hr])) \
            if len(inst_hr) >= 2 else 0.0

        # SDNN index: 按 1 min 分段求 SDNN 再平均（文献1/2 均含此特征）
        sdnn_index = self._sdnn_index(ibi_list)

        # HRV 三角指数: RR 间期总个数 ÷ RR 直方图高度（Task Force 定义）
        hrvi = self._hrv_triangular_index(ibi_list)

        # Poincaré 椭圆拟合 SD1/SD2
        sd1, sd2 = self.poincare_sd(ibi_list)
        sd1_rmssd = rmssd / math.sqrt(2.0)      # v1 近似式，保留作对照
        sd_ratio = (sd1 / sd2) if sd2 > 1e-6 else 0.0

        return {
            # ---- v1 既有键（数值语义不变）----
            "sdnn_ms": round(sdnn, 3),
            "rmssd_ms": round(rmssd, 3),
            "pnn50_pct": round(pnn50, 3),
            "mean_hr_bpm": round(mean_hr, 2),
            "sd1_ms": round(sd1, 3),
            "sd2_ms": round(sd2, 3),
            "sd_ratio": round(sd_ratio, 3),
            "n_beats": n,
            # ---- v2 新增（文献1 / 文献2）----
            "sdsd_ms": round(sdsd, 3),
            "nn50": nn50,
            "sdnn_index_ms": round(sdnn_index, 3),
            "ibi_range_ms": round(max(ibi_list) - min(ibi_list), 3),
            "mean_ibi_ms": round(mean_ibi, 3),
            "sd_hr_bpm": round(sd_hr, 3),
            "hrvi": round(hrvi, 3),
            "sd1_rmssd_ms": round(sd1_rmssd, 3),
        }

    def _sdnn_index(self, ibi_list: List[float]) -> float:
        """SDNN index：按 1 min 分段计算 SDNN 后取平均（文献1/2）"""
        if len(ibi_list) < 6:
            return 0.0
        segs: List[float] = []
        cur: List[float] = []
        acc_ms = 0.0
        limit = self.SDNN_INDEX_SEG_SEC * 1000.0
        for v in ibi_list:
            cur.append(v)
            acc_ms += v
            if acc_ms >= limit:
                if len(cur) >= 3:
                    m = sum(cur) / len(cur)
                    segs.append(math.sqrt(sum((x - m) ** 2 for x in cur) / len(cur)))
                cur, acc_ms = [], 0.0
        if len(cur) >= 3:
            m = sum(cur) / len(cur)
            segs.append(math.sqrt(sum((x - m) ** 2 for x in cur) / len(cur)))
        return sum(segs) / len(segs) if segs else 0.0

    def _hrv_triangular_index(self, ibi_list: List[float]) -> float:
        """HRV 三角指数 = RR 间期总数 ÷ 主直方图高度（Task Force 定义）

        直方图分箱宽度 7.8125 ms（1/128 s）为 HRV 标准做法。
        """
        if len(ibi_list) < 5:
            return 0.0
        bin_w = self.HRV_INDEX_BIN_MS
        hist: Dict[int, int] = {}
        for v in ibi_list:
            k = int(v / bin_w)
            hist[k] = hist.get(k, 0) + 1
        return len(ibi_list) / max(hist.values()) if hist else 0.0

    @staticmethod
    def poincare_sd(ibi_list: List[float]) -> Tuple[float, float]:
        """Poincaré 图真椭圆拟合：SD1（短轴/瞬时变异）、SD2（长轴/连续变异）

        文献2 使用「椭圆拟合半短轴/半长轴」而非 v1 的 RMSSD/√2 近似式。
        做法：将 (RR_i, RR_{i+1}) 散点绕原点旋转 −45°，使主对角线成为 x 轴，
        旋转后两个方向的标准差即为 SD2 与 SD1。

        数学上与解析式等价：
            SD1² = ½·Var(RR_i − RR_{i+1}) = ½·SDSD²
            SD2² = 2·SDNN² − ½·SDSD²
        这里采用**直接散点统计**实现，避免解析式在 SDNN 主导时累积误差。
        """
        n = len(ibi_list)
        if n < 3:
            return 0.0, 0.0
        xs = ibi_list[:-1]
        ys = ibi_list[1:]
        m = len(xs)
        # 旋转 −45°: u = (x + y)/√2 为主对角线方向(SD2), v = (y - x)/√2 为垂直方向(SD1)
        u = [(xs[i] + ys[i]) / math.sqrt(2.0) for i in range(m)]
        v = [(ys[i] - xs[i]) / math.sqrt(2.0) for i in range(m)]
        mu = sum(u) / m
        mv = sum(v) / m
        sd2 = math.sqrt(sum((x - mu) ** 2 for x in u) / m)
        sd1 = math.sqrt(sum((x - mv) ** 2 for x in v) / m)
        return sd1, sd2

    # ==================================================================
    # 频域特征（重采样 + Welch —— v2 方法学修正）
    # ==================================================================
    @staticmethod
    def _hann(n: int) -> List[float]:
        """Hann 窗（周期图标准窗，抑制频谱泄漏）"""
        if n <= 1:
            return [1.0] * max(n, 1)
        return [0.5 - 0.5 * math.cos(2.0 * math.pi * i / (n - 1))
                for i in range(n)]

    @staticmethod
    def resample_tachogram(ibi_list: List[float],
                           fs: float = 4.0) -> Tuple[List[float], List[float]]:
        """把不等间隔的 IBI 序列插值为等间隔 tachogram（HRV 频域标准前置步骤）。

        原理：IBI 序列本身是「非均匀采样」的时间序列（每次心跳一个点）。
        直接对其做 FFT 会因采样不均导致频率轴失真，故标准流程先构建
        累积时间轴，再线性插值到固定采样率网格。

        Args:
            ibi_list: 心跳间隔序列（ms）
            fs: 目标采样率（Hz），默认 4 Hz（Nyquist 2 Hz ≫ HF 上限 0.4 Hz）

        Returns:
            (t_grid, y_grid) —— 时间轴（s）与插值后的 IBI 值（ms）
        """
        n = len(ibi_list)
        if n < 2:
            return [], []
        # 累积时间轴：第 k 个样本的时间 = 前 k 个 IBI 之和
        t: List[float] = [0.0]
        for v in ibi_list[:-1]:
            t.append(t[-1] + v / 1000.0)
        total = t[-1]
        if total <= 0:
            return [], []
        steps = int(total * fs)
        if steps < 4:
            return [], []
        t_grid = [i / fs for i in range(steps + 1)]
        y_grid: List[float] = []
        j = 0
        for gt in t_grid:
            while j < n - 2 and t[j + 1] < gt:
                j += 1
            t0, t1 = t[j], t[j + 1] if j + 1 < n else t[j]
            y0, y1 = ibi_list[j], ibi_list[j + 1] if j + 1 < n else ibi_list[j]
            dt = t1 - t0
            y_grid.append(y0 if dt <= 1e-12 else y0 + (y1 - y0) * (gt - t0) / dt)
        return t_grid, y_grid

    @staticmethod
    def _dft_power_at(sig: Sequence[float], freqs: Sequence[float],
                      fs: float) -> List[float]:
        """逐频点功率 |X(f)|²（纯 Python 用 Goertzel 递推，O(N) 每频点）

        只在感兴趣的窄频段（0.003–0.4 Hz，约 30 个频点）上求值，
        避免了整段 FFT 的 O(N log N) 且无需 FFT 库。
        """
        n = len(sig)
        out: List[float] = []
        for f in freqs:
            w = 2.0 * math.pi * f / fs
            c = 2.0 * math.cos(w)
            s1 = s2 = 0.0
            for x in sig:
                s0 = x + c * s1 - s2
                s2 = s1
                s1 = s0
            # Goertzel 输出的能量 |X(f)|²
            out.append(max(0.0, s1 * s1 + s2 * s2 - c * s1 * s2))
        return out

    def _welch_psd(self, y: Sequence[float],
                   fs: float) -> Tuple[List[float], List[float]]:
        """Welch 平均周期图 → (freqs, psd)，psd 单位为 ms²/Hz。

        流程：分段（默认 256 点、50% 重叠）→ 加 Hann 窗 → 各段功率谱
        → 段间平均。相比单段周期图方差更小（Welch 的核心收益）。

        归一化：PSD = |X(f)|² / (fs · Σw²)，使得 ∫PSD df 还原序列方差。
        """
        n = len(y)
        if n < 8:
            return [], []
        nperseg = self.welch_nperseg
        if nperseg > n:
            nperseg = max(16, 2 ** int(math.log2(n)))
        step = max(1, int(nperseg * (1.0 - self.WELCH_OVERLAP)))
        win = self._hann(nperseg)
        win_norm = sum(w * w for w in win) or 1.0

        segs: List[List[float]] = []
        start = 0
        while start + nperseg <= n:
            segs.append([y[start + i] * win[i] for i in range(nperseg)])
            start += step
        if not segs:
            m = min(nperseg, n)
            segs = [[y[i] * win[i] for i in range(m)]]

        nf = nperseg // 2 + 1
        freqs = [i * fs / nperseg for i in range(nf)]
        # 只关心 0–0.45 Hz（HF 上限 0.4 Hz 留一点余量），显著减少计算量
        idx = [i for i, f in enumerate(freqs) if f <= 0.45 + 1e-9]
        sub_freqs = [freqs[i] for i in idx]

        acc = [0.0] * len(sub_freqs)
        if _HAS_NUMPY:
            for seg in segs:
                arr = _np.asarray(seg, dtype=float)
                spec = _np.abs(_np.fft.rfft(arr)) ** 2
                for k, i in enumerate(idx):
                    acc[k] += float(spec[i])
        else:
            for seg in segs:
                pw = self._dft_power_at(seg, sub_freqs, fs)
                for k in range(len(sub_freqs)):
                    acc[k] += pw[k]

        k_seg = len(segs)
        psd = [(acc[k] / k_seg) / (fs * win_norm) for k in range(len(sub_freqs))]
        return sub_freqs, psd

    def frequency_domain(self, ibi_list: List[float]) -> Dict[str, float]:
        """频域特征：VLF / LF / HF / 总功率 / 归一化 LF·HF / LF-HF 比值

        频带（文献2 标准）：VLF 0.003–0.04，LF 0.04–0.15，HF 0.15–0.40 Hz。
        输出功率单位 ms²；同时给出 Task Force 的归一化单位 nu（LF/(LF+HF)×100）。
        """
        blank = {"vlf_ms2": 0.0, "lf_ms2": 0.0, "hf_ms2": 0.0,
                 "total_power_ms2": 0.0, "lf_hf": 0.0,
                 "lf_nu": 0.0, "hf_nu": 0.0}
        if len(ibi_list) < 8:
            return blank

        _, y = self.resample_tachogram(ibi_list, self.resample_hz)
        if len(y) < 8:
            return blank

        fs = self.resample_hz
        freqs, psd = self._welch_psd(y, fs)
        if len(freqs) < 3:
            return blank
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0

        def _band_power(lo: float, hi: float) -> float:
            return sum(psd[i] * df for i, f in enumerate(freqs)
                       if lo - 1e-9 <= f <= hi + 1e-9)

        vlf = _band_power(*self.BAND_VLF)
        lf = _band_power(*self.BAND_LF)
        hf = _band_power(*self.BAND_HF)
        total = _band_power(*self.BAND_TOTAL)

        lf_hf = (lf / hf) if hf > 1e-12 else 0.0
        denom = lf + hf
        lf_nu = (lf / denom * 100.0) if denom > 1e-12 else 0.0
        hf_nu = (hf / denom * 100.0) if denom > 1e-12 else 0.0

        return {"vlf_ms2": round(vlf, 4), "lf_ms2": round(lf, 4),
                "hf_ms2": round(hf, 4), "total_power_ms2": round(total, 4),
                "lf_hf": round(lf_hf, 3),
                "lf_nu": round(lf_nu, 2), "hf_nu": round(hf_nu, 2)}

    # ==================================================================
    # 非线性特征（文献2 的 4 项）
    # ==================================================================
    @staticmethod
    def sample_entropy(x: Sequence[float], m: int = 2,
                       r: Optional[float] = None) -> float:
        """样本熵 SampEn(m, r)：衡量时间序列的复杂度/不规则度。

        定义：SampEn = −ln(A/B)，B 为 m 维模板匹配对数，A 为 m+1 维匹配对数。
        使用 Chebyshev 距离（各维差值绝对值的最大值）与 r = 0.2×SD 的
        文献通用取值。SampEn 越大表示序列越不规则（健康 HRV 一般较大，
        病理或高应激状态下降）。

        注意：样本熵**不计自匹配**（不同于近似熵 ApEn），因此无偏且对
        数据长度依赖更小。返回值越大越复杂；数据不足时返回 0。
        """
        n = len(x)
        if n < m + 3:
            return 0.0
        if r is None:
            mean = sum(x) / n
            sd = math.sqrt(sum((v - mean) ** 2 for v in x) / n)
            r = 0.2 * sd
        if r <= 1e-12:
            return 0.0

        def _count(mm: int) -> int:
            cnt = 0
            limit = n - mm
            for i in range(limit - 1):
                xi = x[i:i + mm]
                for j in range(i + 1, limit):
                    ok = True
                    for k in range(mm):
                        if abs(xi[k] - x[j + k]) > r:
                            ok = False
                            break
                    if ok:
                        cnt += 1
            return cnt

        b = _count(m)
        a = _count(m + 1)
        if b <= 0:
            return 0.0
        if a <= 0:
            # A = 0 → 理论上 SampEn → ∞；工程上用一个有限上界表示"极不规则"
            return float(round(math.log(b + 1.0), 3))
        return round(-math.log(a / b), 3)

    @staticmethod
    def dfa_alpha(x: Sequence[float], scale_min: int = 4,
                  scale_max: int = 16, n_scales: int = 7) -> float:
        """DFA（去趋势波动分析）标度指数 α：量化序列的长程幂律相关性。

        步骤（Peng 等 1995 原始定义）：
          ① 去均值后积分 y(k) = Σ_{i≤k} (x_i − x̄)
          ② 在尺度 s 上把 y 切成 ⌊N/s⌋ 段，每段做最小二乘直线拟合并去趋势
          ③ 求各段残差均方根 F(s)
          ④ 在 log F(s) ~ log s 上做线性回归，斜率即 α

        判读（HRV 文献通用）：
          α1 ≈ 1.0（4–16 拍）：健康短程相关；α1 降低常见于应激/病理
          α2 ≈ 1.0（16–64 拍）：长程相关
          白噪声 α ≈ 0.5，完全相关的 1/f 噪声 α ≈ 1.0
        """
        n = len(x)
        if n < scale_max * 2:
            scale_max = max(scale_min + 2, n // 4)
        if scale_max <= scale_min:
            return 0.0
        mean = sum(x) / n
        y: List[float] = []
        acc = 0.0
        for v in x:
            acc += v - mean
            y.append(acc)

        # 尺度按几何级数分布（log 轴均匀 → 回归更稳）
        scales: List[int] = []
        if n_scales <= 1 or scale_max <= scale_min:
            scales = [int(scale_min)]
        else:
            ratio = (scale_max / scale_min) ** (1.0 / (n_scales - 1))
            s = float(scale_min)
            while s <= scale_max + 1e-9:
                scales.append(int(round(s)))
                s *= ratio
            scales = sorted({max(scale_min, v) for v in scales})

        logs_s: List[float] = []
        logs_f: List[float] = []
        for s in scales:
            nseg = n // s
            if nseg < 2:
                continue
            sq: List[float] = []
            for k in range(nseg):
                seg = y[k * s:(k + 1) * s]
                # 最小二乘直线拟合（x = 0..s-1）
                sx = (s - 1) * s / 2.0
                sxx = (s - 1) * s * (2 * s - 1) / 6.0
                sy = sum(seg)
                sxy = sum(i * seg[i] for i in range(s))
                den = s * sxx - sx * sx
                if abs(den) < 1e-12:
                    continue
                slope = (s * sxy - sx * sy) / den
                inter = (sy - slope * sx) / s
                resid = [(seg[i] - (slope * i + inter)) ** 2 for i in range(s)]
                sq.append(sum(resid) / s)
            if not sq:
                continue
            f_s = math.sqrt(sum(sq) / len(sq))
            if f_s > 1e-12:
                logs_s.append(math.log(s))
                logs_f.append(math.log(f_s))

        if len(logs_s) < 3:
            return 0.0
        m = len(logs_s)
        mx = sum(logs_s) / m
        my = sum(logs_f) / m
        num = sum((logs_s[i] - mx) * (logs_f[i] - my) for i in range(m))
        den2 = sum((logs_s[i] - mx) ** 2 for i in range(m))
        if den2 <= 1e-12:
            return 0.0
        return round(num / den2, 4)

    def nonlinear_domain(self, ibi_list: List[float]) -> Dict[str, float]:
        """非线性特征：样本熵 / DFA α1·α2（Poincaré 已在 time_domain 输出）"""
        blank = {"sampen": 0.0, "dfa_alpha1": 0.0, "dfa_alpha2": 0.0}
        if len(ibi_list) < 20:
            return blank
        return {
            "sampen": self.sample_entropy(ibi_list, self.SAMPEN_M),
            "dfa_alpha1": self.dfa_alpha(ibi_list, *self.DFA_ALPHA1_RANGE),
            "dfa_alpha2": self.dfa_alpha(ibi_list, *self.DFA_ALPHA2_RANGE),
        }

    # ==================================================================
    # 信号质量门禁（与 ESP32 固件 hrv.py 对齐）
    # ==================================================================
    def quality_metrics(self, ibi_list: List[float],
                        feats: Optional[Dict] = None,
                        clean: bool = True) -> Dict:
        """信号质量指标（低质量数据应被下游拒绝，而非硬算出情绪）

        核心指标 **RR 抖动指数 jitter = RMSSD / mean(IBI)**：
        这是唯一能在「HR 完全正常」的假象下把问题揪出来的指标。
        因为 HR 只看峰间距均值（抖动被平均掉），而 HRV 看逐拍差分
        （抖动全额进入）—— 二者可信度天然解耦。

        实测参考（本项目 2026-09-30 指腹贴放）：
            静息合格    jitter ≤ 0.08
            勉强        0.08 ~ 0.12
            不可用      > 0.12（实测指腹接触不良时达 0.455）
        """
        report = self.clean_ibi_report(ibi_list) if clean else {
            "cleaned": list(ibi_list), "n_in": len(ibi_list),
            "n_out": len(ibi_list), "rejected": 0, "ectopic_ratio": 0.0}
        vals = report["cleaned"]
        if len(vals) < 5:
            return {**report, "jitter": 0.0, "quality_ok": False,
                    "quality_level": "不足", "reason": "有效拍数不足"}
        f = feats or self.time_domain(vals)
        mean_ibi = f.get("mean_ibi_ms") or (sum(vals) / len(vals))
        rmssd = f.get("rmssd_ms", 0.0)
        jitter = (rmssd / mean_ibi) if mean_ibi > 0 else 0.0

        # 阈值一律取自 config，不在此处写死 —— 否则运维改了配置却不生效，
        # 现场排查时"配置显示已收紧、行为却照旧"是最难定位的一类问题。
        j_good = config.QUALITY_JITTER_GOOD
        j_poor = config.QUALITY_JITTER_POOR
        ect_good = config.QUALITY_ECTOPIC_GOOD
        ect_max = config.QUALITY_ECTOPIC_MAX
        ectopic = report["ectopic_ratio"]

        if jitter <= j_good and ectopic <= ect_good:
            level, ok = "合格", True
        elif jitter <= j_poor and ectopic <= ect_max:
            level, ok = "勉强", True
        else:
            level, ok = "不可用", False

        # 说明不可用的**首要**原因（jitter 与伪迹率可能同时超标，
        # 但改善措施完全不同：前者是接触/灌注问题，后者是异位搏动或
        # 漏检/双计问题）。只说"不可用"会让用户无从下手。
        if ok:
            reason = ""
        elif ectopic > ect_max and jitter <= j_poor:
            reason = (f"伪迹比例过高（{ectopic:.0%} > {ect_max:.0%}），"
                      f"请检查电极/指腹接触是否稳定")
        elif jitter > j_poor:
            reason = (f"逐拍抖动过大（{jitter:.3f} > {j_poor}），"
                      f"多为接触不良或灌注不足，HR 可能仍显示正常")
        else:
            reason = "有效拍数不足或指标落在交界区"

        return {**report, "jitter": round(jitter, 4),
                "quality_ok": ok, "quality_level": level,
                "reason": reason,
                "mean_ibi_ms": round(mean_ibi, 2)}

    # ==================================================================
    # 个体基线相对化（文献1「相对特征值法」）
    # ==================================================================
    #: 参与相对化的特征（需要"越大越好"还是"越小越好"由下游规则决定，
    #: 本方法只做线性变换，不引入方向性假设）
    RELATIVE_KEYS = ("hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "lf_hf",
                     "sdsd_ms", "sd1_ms", "sd2_ms", "sd_hr_bpm",
                     "sdnn_index_ms", "sampen", "dfa_alpha1",
                     "total_power_ms2", "lf_nu")

    @classmethod
    def relative_features(cls, feats: Dict[str, float],
                          baseline: Optional[Dict[str, float]],
                          clip: float = 3.0) -> Dict[str, float]:
        """个体基线相对化：相对值 = (基线值 − 当前值) / 基线值

        依据文献1《基于心率变异性的情绪识别研究》—— 该文的"相对特征值法"
        是其应对个体差异的核心手段（每个人先测平静基线，所有特征转成
        相对基线的变化率后再送入分类器）。

        符号约定：正值 = 当前值低于基线，负值 = 高于基线。
        例如"高焦虑"时 RMSSD 下降 → 相对值为正。

        仿真结果仅用于检查代码；真实情绪效果须由独立自评及留出数据评估。

        Args:
            feats: 当前特征字典
            baseline: 个体平静期基线字典（键名需与 feats 一致）
            clip: 相对值钳位幅度（默认 ±3，避免基线过小时数值爆炸）

        Returns:
            **只包含真正完成相对化的特征**（形如 ``{"rmssd_ms": 0.41, ...}``），
            值域 ±clip。缺基线、缺当前值、基线过小的特征**不会出现在结果中**。

        .. warning::
           返回结果里**绝不含原始量纲的值**。这是 v2 修正的一处真实缺陷：
           原实现以 ``out = dict(feats)`` 起手，把"没有基线、只能保留原值"的
           特征一并返回。而基线通常只有 7 个键（hr_bpm/sdnn_ms/rmssd_ms/
           lf_hf/pnn50_pct/sd_hr_bpm/sdnn_index_ms），``RELATIVE_KEYS`` 却有
           14 个 —— 于是 sampen=1.5、dfa_alpha1=1.02 这类**原始值**会被
           消费方当成"相对基线偏移 +1.5"来读。

           后果不是报错而是**静默错误**：``pad_tool._explain`` 会把
           "样本熵 1.5" 解读为"显著低于平时基线"，并且因为它数值最大
           而挤掉真正有意义的偏移，排在归因列表第一位。
           混在同一套键空间里的两种量纲没有任何标记可以区分，
           所以正确做法是**不放进来**，而不是"保持原值"。
        """
        if not baseline:
            return {}
        out: Dict[str, float] = {}
        for k in cls.RELATIVE_KEYS:
            cur = feats.get(k)
            base = baseline.get(k)
            if cur is None or base is None:
                continue
            try:
                cur_f, base_f = float(cur), float(base)
            except (TypeError, ValueError):
                continue
            if base_f <= 1e-9:
                continue
            rel = (base_f - cur_f) / base_f
            out[k] = round(max(-clip, min(clip, rel)), 4)
        return out

    # ==================================================================
    # 总入口
    # ==================================================================
    def extract_from_ibi(self, ibi_list: List[float],
                         with_nonlinear: bool = True) -> Dict[str, float]:
        """IBI 序列 → 完整 HRV 特征字典（算法层标准特征格式）

        Args:
            ibi_list: 心跳间隔序列（ms）
            with_nonlinear: 是否计算样本熵/DFA（O(n²)，300 拍约 0.2~0.5 s；
                高频率调用场景可置 False 以省算力）
        """
        cleaned = self.clean_ibi(ibi_list)
        feats = self.time_domain(cleaned)
        feats.update(self.frequency_domain(cleaned))
        if with_nonlinear:
            feats.update(self.nonlinear_domain(cleaned))
        else:
            feats.update({"sampen": 0.0, "dfa_alpha1": 0.0, "dfa_alpha2": 0.0})
        # 统一键名：与 ESP32 数据包 / 工具链特征对齐
        feats["hr_bpm"] = feats.pop("mean_hr_bpm")
        feats["sdnn"] = feats.pop("sdnn_ms")
        feats["rmssd"] = feats.pop("rmssd_ms")
        feats["pnn50"] = feats.pop("pnn50_pct")
        feats["ibi_list"] = list(cleaned)
        # 抖动指数：与固件 rr_jitter() 同定义，供下游质量门禁
        if feats.get("mean_ibi_ms"):
            feats["jitter"] = round(feats["rmssd"] / feats["mean_ibi_ms"], 4)
        return self.add_aliases(feats)

    def extract_from_ble(self, packet: Dict) -> Dict[str, float]:
        """
        从 ESP32 数据包提取特征（计划书 §3.3 数据包格式）:
          {hr, ibi[], sdnn, rmssd, pnn50, lf_hf, rr, sig_quality, t,
           motion, moving, jitter, ...}

        策略：优先使用固件已算值（边缘侧算力已投入，且与固件窗口一致），
        缺失字段从 IBI 重算；完全缺失时按心率估算兜底。

        v2 变更：新增字段（motion/jitter/quality）透传；ibis 重算时
        一并产出 v2 扩展特征，使后端与固件口径可对照。
        """
        feats: Dict[str, float] = {}
        feats["hr_bpm"] = float(packet.get("hr") or 0.0)
        feats["sdnn"] = float(packet.get("sdnn") or 0.0)
        feats["rmssd"] = float(packet.get("rmssd") or 0.0)
        feats["pnn50"] = float(packet.get("pnn50") or 0.0)
        feats["lf_hf"] = float(packet.get("lf_hf") or 0.0)
        feats["rr_bpm"] = float(packet.get("rr") or 0.0)
        feats["sig_quality"] = float(packet.get("sig_quality") or 0.0)
        feats["timestamp"] = float(packet.get("t") or 0.0)
        # v2：透传固件的运动门控与质量字段（缺省时视为无运动、质量未知）
        if packet.get("jitter") is not None:
            feats["jitter"] = float(packet.get("jitter"))
        feats["motion"] = float(packet.get("motion") or 0.0)
        feats["moving"] = bool(packet.get("moving"))

        ibi = [float(v) for v in (packet.get("ibi") or [])
               if isinstance(v, (int, float))]
        if ibi:
            computed = self.extract_from_ibi(ibi)
            for key in ("sdnn", "rmssd", "pnn50", "lf_hf", "hr_bpm",
                        "sd1_ms", "sd2_ms", "sd_ratio", "sdsd_ms",
                        "nn50", "sdnn_index_ms", "ibi_range_ms",
                        "mean_ibi_ms", "sd_hr_bpm", "hrvi", "vlf_ms2",
                        "total_power_ms2", "lf_nu", "hf_nu",
                        "sampen", "dfa_alpha1", "dfa_alpha2", "jitter"):
                if feats.get(key) in (None, 0.0) and computed.get(key):
                    feats[key] = computed[key]
            feats["n_beats"] = computed.get("n_beats", 0)
            feats["ibi_list"] = computed["ibi_list"]

        # 兜底: 仅有心率时按经验公式估算 SDNN/RMSSD
        if packet.get('_simulation') is True and feats["hr_bpm"] > 0 and feats.get("sdnn", 0.0) <= 0:
            hr = feats["hr_bpm"]
            feats["sdnn"] = round(max(8.0, 2100.0 / hr), 2)
            feats["rmssd"] = round(max(6.0, 1600.0 / hr), 2)
            feats["pnn50"] = 0.0

        # Poincaré 派生（缺省时由 RMSSD 估算，供 ML 层使用）
        if feats.get("sd1_ms", 0.0) <= 0 and feats["rmssd"] > 0:
            feats["sd1_ms"] = round(feats["rmssd"] / math.sqrt(2.0), 3)
            feats["sd2_ms"] = round(math.sqrt(max(0.0, 2 * feats["sdnn"] ** 2
                                                  - feats["sd1_ms"] ** 2 / 2)), 3)
            feats["sd_ratio"] = round(feats["sd1_ms"] / feats["sd2_ms"], 3) \
                if feats["sd2_ms"] > 1e-6 else 0.0
        # 长名双写（见类注释「兼容修复」）：必须在所有回落/派生计算之后，
        # 否则规则层会读到未回落的 0 值。
        return self.add_aliases(feats)


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    import random
    random.seed(7)
    print("=" * 66)
    print("HRV 特征提取器 v2 自测")
    print("=" * 66)

    # 模拟平静状态 IBI（约 850ms 均值，抖动 30ms）
    calm = [850 + random.gauss(0, 30) for _ in range(300)]
    # 模拟焦虑状态 IBI（约 700ms 均值，抖动 12ms —— HR 高、HRV 低）
    anxious = [700 + random.gauss(0, 12) for _ in range(300)]

    ex = HRVExtractor()
    for label, ibi in (("平静", calm), ("焦虑", anxious)):
        f = ex.extract_from_ibi(ibi)
        q = ex.quality_metrics(ibi, f)
        keys = ("hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "sdsd_ms",
                "nn50", "sdnn_index_ms", "ibi_range_ms", "sd_hr_bpm", "hrvi",
                "sd1_ms", "sd2_ms", "lf_ms2", "hf_ms2", "lf_hf", "lf_nu",
                "sampen", "dfa_alpha1", "dfa_alpha2", "jitter")
        print(f"\n[{label}]")
        for k in keys:
            print(f"  {k:<18} {f.get(k)}")
        print(f"  {'质量等级':<18} {q['quality_level']} "
              f"(jitter={q['jitter']}, 伪迹率={q['ectopic_ratio']})")

    # 相对化演示
    base = ex.extract_from_ibi(calm)
    cur = ex.extract_from_ibi(anxious)
    rel = HRVExtractor.relative_features(cur, base)
    print("\n[个体基线相对化] 以平静为基线，焦虑状态的相对值：")
    for k in ("hr_bpm", "rmssd_ms", "sdnn_ms", "lf_hf"):
        print(f"  {k:<14} 绝对 {cur.get(k):<10} → 相对 {rel.get(k)}")
