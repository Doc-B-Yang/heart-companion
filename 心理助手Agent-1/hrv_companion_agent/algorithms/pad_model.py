# -*- coding: utf-8 -*-
"""
PAD 维度情感模型（Pleasure–Arousal–Dominance）
=============================================
对应计划书 §4.2「算法子模块 2.6」—— 为情绪识别增加**连续维度输出**。

为什么要从「离散分类」扩到「维度回归」
--------------------------------------------------------------------------------
v1 只输出 5 类离散标签（平静/低焦虑/高焦虑/愉悦/疲劳），存在两个结构性问题：

  1. **粒度损失**：两类之间没有中间态。"有点开心"和"非常开心"同属「愉悦」，
     下游无法据此调整陪伴强度 —— 而陪伴恰恰最需要"程度感"。
  2. **单模态天花板**：文献1《基于心率变异性的情绪识别研究》实测，
     仅用 HRV 做 8 类离散分类准确率只有 **13.75%**，四分类最好 47.5%，
     二分类 60~75%。离散硬分类在 HRV 上天然吃亏。

文献2《基于心率变异性特征和 PCA-SVR 的 PAD 维度情感预测分析》
（韩永明等，北京化工大学学报 2021, 48(5): 102-110）给出的替代路线是
**回归连续维度**，实测（混合情感状态，PCA-SVR）：

    | 维度 | MSE  | ρ    | ρc(CCC) |
    |------|------|------|---------|
    | P    | 0.37 | 0.80 | **0.78** |
    | A    | 1.76 | 0.87 | 0.46    |
    | D    | 0.96 | 0.66 | 0.30    |
    | 平均 | 1.03 | 0.78 | 0.51    |

关键结论（必须诚实对待）：
  * **P（愉悦度）与 A（唤醒度）是 HRV 能预测的**；
  * **D（优势度）效果最差**（文献原话："能很好地在愉悦度和唤醒度上识别和
    预测情感，在优势度上较差"）。本模块因此给 D 标注 **低置信**，
    并建议 D 主要交由对话文本补充，而非依赖生理信号。

本模块提供两条并联路径
--------------------------------------------------------------------------------
  A. **规则映射**（``RulePAD``，零依赖，始终可用）
     基于 HRV 特征的生理学含义映射到 P/A/D，并把个体基线相对化纳入
     （文献1 的"相对特征值法"）。有文献方向性依据，但**系数是本项目的
     工程标定值，不是文献给出的公式** —— 引用时须注明。

  B. **PCA-SVR 回归**（``PADRegressor``，需 numpy/sklearn，有训练数据时启用）
     严格复现文献2 的流程：z-score 标准化 → PCA 降维 → SVR(RBF)。
     文献2 报告的两档累计贡献度阈值均支持（0.85 为常规准则，0.99 为其实用值）。

⚠️ 关于 PCA 的重要提醒（文献2 自证）
    文献2 在**单一情绪**状态下 PCA 反而**变差**（放松：PCA-SVR ρc 0.23
    vs 纯 SVR 0.54；焦虑：0.15 vs 0.41），原因是单状态样本太少。
    故本模块默认 ``use_pca`` 由样本量决定，小样本时**不做降维**。

参考文献
--------
[2] 韩永明, 张明星, 耿志强. 基于心率变异性特征和 PCA-SVR 的 PAD 维度
    情感预测分析[J]. 北京化工大学学报(自然科学版), 2021, 48(5): 102-110.
[1] 易慧, 陈瑞娟, 邓光华, 等. 基于心率变异性的情绪识别研究[J].
    生物医学工程研究, 2020, 39(2): 128-132.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None
    _HAS_NUMPY = False

try:  # sklearn 为可选依赖，缺失时仅启用规则路径
    from sklearn.decomposition import PCA as _PCA
    from sklearn.preprocessing import StandardScaler as _StandardScaler
    from sklearn.svm import SVR as _SVR
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _PCA = _StandardScaler = _SVR = None
    _HAS_SKLEARN = False


# ======================================================================
# 一、PAD 自评量表（文献2 式(1) 的 12 项中文简化版）
# ======================================================================
#: 12 组对立形容词，按 P/A/D 三轴各 4 项交替排列
#: 计分取值 −4 ~ +4（左端 −4，右端 +4）
#:
#: ⚠️ **左右端顺序不是随意写的**：文献2 式(1) 对 12 项分别赋予 ±1 的系数，
#:    其中 V2/V4/V6/V8/V9/V12 为**反向题**（防止被试的固定反应偏差，
#:    心理测量学的标准做法）。为让"填表 12 个分值 → 公式 → PAD"全程自洽，
#:    本表的左右端已按式(1) 的系数符号对齐：每项右端(+4) 所指的属性，
#:    恰好是该项系数所"奖励"的方向。填写时数值越大 = 越靠右端的词。
PAD_SCALE_ITEMS: Tuple[Tuple[str, str, str], ...] = (
    ("P", "愤怒的", "感兴趣的"),      # V1  系数 +1
    ("A", "清醒的", "困倦的"),        # V2  系数 −1（反向题）
    ("D", "受控的", "主控的"),        # V3  系数 +1
    ("P", "友好的", "轻蔑的"),        # V4  系数 −1（反向题）
    ("A", "平静的", "激动的"),        # V5  系数 +1
    ("D", "支配的", "被支配的"),      # V6  系数 −1（反向题）
    ("P", "不满的", "满足的"),        # V7  系数 +1
    ("A", "专注的", "分心的"),        # V8  系数 −1（反向题）
    ("D", "被影响的", "有影响力的"),  # V9  系数 +1
    ("P", "愉快的", "不快的"),        # V10 系数 −1（反向题）
    ("A", "沉静的", "兴奋的"),        # V11 系数 +1
    ("D", "重要的", "被忽视的"),      # V12 系数 −1（反向题）
)


def pad_from_questionnaire(v: Sequence[float]) -> Dict[str, float]:
    """12 项自评 → PAD 三维分值（文献2 式(1)）

    文献2 原文公式::

        ValueP = (V1 − V4 + V7  − V10) / 4
        ValueA = (−V2 + V5 − V8  + V11) / 4
        ValueD = (V3 − V6 + V9  − V12) / 4

    Args:
        v: 长度为 12 的原始打分辨列（−4 ~ +4），**必须按 PAD_SCALE_ITEMS
           的顺序**填写（即：数值越大越靠该行的右端词）

    Returns:
        {"P": .., "A": .., "D": ..}，各自 ∈ [−4, +4]

    Raises:
        ValueError: 分列长度不足 12
    """
    if len(v) < 12:
        raise ValueError(f"PAD 量表需要 12 项打分，收到 {len(v)} 项")
    val = [float(x) for x in v[:12]]
    return {
        "P": round((val[0] - val[3] + val[6] - val[9]) / 4.0, 4),
        "A": round((-val[1] + val[4] - val[7] + val[10]) / 4.0, 4),
        "D": round((val[2] - val[5] + val[8] - val[11]) / 4.0, 4),
    }


def pad_to_emotion(p: float, a: float, d: float) -> str:
    """PAD 三维 → 本项目 5 类离散情绪

    判据来自文献2 对"焦虑"与"放松"的明确定义，其余由唤醒度与愉悦度组合推出::

        焦虑：P<0, A>0, D<0        （文献2 原文）
        放松：P>0, A<0, D>0        （文献2 原文）

        本项目扩展映射：
          高焦虑 ← P<0 且 A 高
          低焦虑 ← P<0 且 A 中
          疲劳   ← P<0 且 A 低
          愉悦   ← P>0 且 A 高
          平静   ← P>0 且 A 低

    ⚠️ 注意：此映射只使用 P 与 A 两轴 —— 因为文献2 实测 D 轴 ρc 仅 0.30，
       用它参与离散化会引入噪声。D 仅在 risk 评估中作弱修正项。

    分界说明：文献2 只给出了"焦虑(P<0,A>0,D<0)"与"放松(P>0,A<0,D>0)"
    两个区域的**符号判据**，并未定义中间带。本项目的 5 类映射需要在
    "低焦虑"与"疲劳"之间划一条线，故引入经验分界 ``A_HI``（±0.5，
    即 ±12.5% 量程）—— **属工程取值，需实验标定**。
    """
    a_hi = 0.5      # 唤醒度分界（−4~+4 量纲），经验值
    if p < 0:
        if a >= a_hi:
            return "高焦虑"
        if a >= -a_hi:
            return "低焦虑"
        return "疲劳"
    if a >= a_hi:
        return "愉悦"
    return "平静"


# ======================================================================
# 二、规则映射（零依赖路径）
# ======================================================================
@dataclass
class PADResult:
    """PAD 估计结果（含方法与置信信息，便于下游判断可信度）"""
    P: float
    A: float
    D: float
    method: str                # rule / svr / hybrid
    confidence: float          # 整体置信度 0~1
    d_confidence: float        # D 轴单独置信度（文献2 表明其天然偏低）
    emotion_hint: str = ""     # 由 P/A 推出的离散情绪建议
    detail: Dict = None

    def to_dict(self) -> Dict:
        return {"P": round(self.P, 3), "A": round(self.A, 3),
                "D": round(self.D, 3), "method": self.method,
                "confidence": round(self.confidence, 3),
                "d_confidence": round(self.d_confidence, 3),
                "emotion_hint": self.emotion_hint,
                "detail": self.detail or {}}


class RulePAD:
    """基于 HRV 生理学含义的 PAD 规则估计（零依赖，永远可用）

    映射依据（方向性来自文献，系数为本项目工程标定值）：

    **A 唤醒度** —— 交感激活/整体代谢水平
        * 心率相对个体基线的升高量（最强、最稳的唤醒指标）
        * LF/HF 升高（交感-副交感平衡向交感偏移）
        * SD_HR（心率自身的波动幅度，文献2 第 4 特征）

    **P 愉悦度** —— 副交感张力/恢复状态
        * RMSSD 相对基线（副交感活性的核心指标，情绪效价的主要生理相关）
        * SDNN、pNN50 同向
        * 注：效价在纯生理信号上本就不易分离 —— 这正是文献2 报 P 轴
          ρc=0.78（三轴中最高）却仍需 SVR 而非简单阈值的原因。

    **D 优势度** —— HRV 最不敏感
        * 仅给弱先验（以 SDNN 与 HR 的组合方向作粗估）
        * ``d_confidence`` 恒设为低值，提醒下游不要依赖
    """

    #: 唤醒度权重（HR 相对变化 / LF-HF / SD_HR）
    A_WEIGHTS = {"hr_rel": 0.55, "lf_hf": 0.30, "sd_hr": 0.15}
    #: 愉悦度权重（RMSSD 相对变化为主）
    P_WEIGHTS = {"rmssd_rel": 0.60, "sdnn_rel": 0.25, "pnn50_rel": 0.15}
    #: 参考尺度：把无量纲的相对变化映射到 −4~+4
    SCALE = 4.0

    def predict(self, feats: Dict[str, float],
                baseline: Optional[Dict[str, float]] = None) -> PADResult:
        base = baseline or {}

        def _rel(key: str, fb: float) -> float:
            """相对基线变化率（文献1「相对特征值法」的符号取反版）"""
            cur = feats.get(key)
            b = base.get(key)
            try:
                cur_f = float(cur) if cur is not None else None
                b_f = float(b) if b is not None else None
            except (TypeError, ValueError):
                return 0.0
            if cur_f is None or cur_f <= 0:
                return 0.0
            if b_f is None or b_f <= 1e-9:
                # 无基线时用人群参考中值兜底（标注为弱证据）
                b_f = fb
            if b_f <= 1e-9:
                return 0.0
            return (cur_f - b_f) / b_f

        # ---- A 唤醒度 ----
        hr_rel = _rel("hr_bpm", 70.0)
        lf_hf = float(feats.get("lf_hf") or 0.0)
        # LF/HF 以 1.5 为静息参考中点，按 ±1.5 归一
        lf_hf_norm = (lf_hf - 1.5) / 1.5 if lf_hf > 0 else 0.0
        sd_hr = float(feats.get("sd_hr_bpm") or 0.0)
        sd_hr_norm = (sd_hr - 3.0) / 3.0 if sd_hr > 0 else 0.0

        a_raw = (self.A_WEIGHTS["hr_rel"] * hr_rel
                 + self.A_WEIGHTS["lf_hf"] * lf_hf_norm
                 + self.A_WEIGHTS["sd_hr"] * max(-1.0, min(1.0, sd_hr_norm)))
        A = max(-1.0, min(1.0, a_raw)) * self.SCALE

        # ---- P 愉悦度 ----
        rmssd_rel = _rel("rmssd_ms", 42.0)
        sdnn_rel = _rel("sdnn_ms", 50.0)
        pnn50_rel = _rel("pnn50_pct", 15.0)
        p_raw = (self.P_WEIGHTS["rmssd_rel"] * rmssd_rel
                 + self.P_WEIGHTS["sdnn_rel"] * sdnn_rel
                 + self.P_WEIGHTS["pnn50_rel"] * pnn50_rel)
        P = max(-1.0, min(1.0, p_raw)) * self.SCALE

        # ---- D 优势度（弱先验，明确低置信）----
        # 生理上优势度与"交感激活 + 副交感张力并存"有关（掌控感常伴
        # 适度唤醒与良好恢复力），故取 A 与 P 的同向组合作粗估。
        D = max(-1.0, min(1.0, 0.5 * (a_raw * 0.3 + p_raw * 0.7))) * self.SCALE

        # 置信度：有基线 → 高；无基线（用人群中值兜底）→ 低
        has_base = bool(base.get("rmssd_ms") or base.get("sdnn_ms"))
        conf = 0.55 if has_base else 0.30
        # 质量门禁：抖动指数过高时进一步降置信
        jit = float(feats.get("jitter") or 0.0)
        if jit > 0.12:
            conf *= 0.4
        elif jit > 0.08:
            conf *= 0.7

        return PADResult(
            P=P, A=A, D=D, method="rule",
            confidence=round(conf, 3),
            d_confidence=round(conf * 0.35, 3),   # 文献2: D 轴 ρc 仅 0.30
            emotion_hint=pad_to_emotion(P, A, D),
            detail={"hr_rel": round(hr_rel, 4), "rmssd_rel": round(rmssd_rel, 4),
                    "lf_hf_norm": round(lf_hf_norm, 4),
                    "based_on_baseline": has_base})


# ======================================================================
# 三、PCA-SVR 回归（文献2 流程，需 sklearn）
# ======================================================================
#: 送入 PAD 回归的 HRV 特征（对应文献2 的 19 维，取本项目可实现子集）
PAD_FEATURE_NAMES: Tuple[str, ...] = (
    "mean_ibi_ms", "sdnn_ms", "hr_bpm", "sd_hr_bpm", "rmssd_ms",
    "nn50", "pnn50_pct", "sdnn_index_ms",
    "vlf_ms2", "lf_ms2", "hf_ms2", "lf_nu", "hf_nu", "lf_hf",
    "sd1_ms", "sd2_ms", "sampen", "dfa_alpha1", "dfa_alpha2",
)


class PADRegressor:
    """PCA-SVR 三维回归（文献2 完整流程）

    流程: z-score 标准化 → (可选)PCA 降维 → RBF-SVR 逐维回归

    文献2 的 PCA 阈值给了两档：
      * 常规准则 >85%（``PCA_VAR_KEEP_STD``）
      * 论文实用值 99%（``PCA_VAR_KEEP_PAPER``）
    本实现默认 85%，并**在样本量不足时自动关闭 PCA** —— 因为文献2 自身
    证明了小样本下 PCA 会掉分（单一情绪状态 ρc 0.23 vs 0.54）。
    """

    PCA_VAR_KEEP_STD = 0.85
    PCA_VAR_KEEP_PAPER = 0.99
    MIN_SAMPLES_FOR_PCA = 100

    def __init__(self, model_path: Optional[str] = None,
                 use_pca: Optional[bool] = None):
        self.model_path = model_path
        self.use_pca = use_pca
        self._bundle = None
        self.available = False
        self._try_load()

    def _try_load(self):
        if not _HAS_SKLEARN or not self.model_path:
            return
        try:
            import joblib
            import os
            if not os.path.exists(self.model_path):
                return
            self._bundle = joblib.load(self.model_path)
            self.available = isinstance(self._bundle, dict) \
                and "models" in self._bundle
        except Exception:  # noqa: BLE001
            self.available = False

    def predict(self, feats: Dict[str, float]) -> Optional[PADResult]:
        """特征 → (P, A, D)；模型不可用时返回 None 交由规则路径兜底"""
        if not self.available:
            return None
        try:
            names = self._bundle.get("feature_names", PAD_FEATURE_NAMES)
            if feats.get("mean_ibi_ms") in (None, 0) and feats.get("hr_bpm"):
                # mean_ibi 缺失时由心率反推，保证特征向量完整
                feats = dict(feats)
                feats["mean_ibi_ms"] = 60000.0 / float(feats["hr_bpm"])
            row = [[float(feats.get(n, 0) or 0) for n in names]]
            X = _np.asarray(row, dtype=float)
            if self._bundle.get("scaler") is not None:
                X = self._bundle["scaler"].transform(X)
            if self._bundle.get("pca") is not None:
                X = self._bundle["pca"].transform(X)
            out = {}
            confs = []
            for axis in ("P", "A", "D"):
                m = self._bundle["models"].get(axis)
                if m is None:
                    out[axis] = 0.0
                    continue
                v = float(m.predict(X)[0])
                out[axis] = max(-4.0, min(4.0, v))
            meta = self._bundle.get("metadata", {}) or {}
            # SVR 不直接给概率；用训练期 CV 的 ρc 作置信度代理
            conf = float(meta.get("ccc", {}).get("mean", 0.5)) if meta else 0.5
            return PADResult(
                P=out.get("P", 0.0), A=out.get("A", 0.0), D=out.get("D", 0.0),
                method="svr", confidence=round(max(0.0, min(1.0, conf)), 3),
                d_confidence=round(max(0.0, min(1.0, conf)) * 0.6, 3),
                emotion_hint=pad_to_emotion(out.get("P", 0.0),
                                            out.get("A", 0.0),
                                            out.get("D", 0.0)),
                detail={"feature_names": list(names),
                        "pca_used": self._bundle.get("pca") is not None,
                        "source": meta.get("source", "unknown")})
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------ 训练
    def fit(self, X: Sequence[Sequence[float]],
            y: Sequence[Sequence[float]],
            feature_names: Sequence[str] = PAD_FEATURE_NAMES,
            use_pca: Optional[bool] = None,
            cv_folds: int = 5,
            verbose: bool = False) -> Dict:
        """训练 PCA-SVR（严格按文献2 的流程与评估口径）

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签矩阵 (n_samples, 3)，三列依次为 P/A/D，量纲 −4~+4
            feature_names: 特征名（写入 bundle，供 predict 对齐）
            use_pca: 是否降维；None 时按样本量自动决定
            cv_folds: 交叉验证折数（<2 时关闭 CV，只用样本内指标）

        Returns:
            训练摘要。``metrics`` 为**样本内**指标；``cv_metrics`` 为
            交叉验证指标（若可用）——**报告时应当用 cv_metrics**，
            样本内 ρc 会明显虚高（SVR 有能力记住训练点）。

        .. note::
           v2 修正了一处真实缺陷：原实现把「样本内预测」当作置信度来源，
           而 docstring 声称用的是 CV。二者不一致会导致 PAD 的 confidence
           被系统性高估（本模块自测中样本内 mean_ρc≈0.98，而 5 折 CV
           会明显更低）。现在两个口径分开返回，且 ``predict()`` 读的是
           ``metadata.ccc``（由 CV 写入）。
        """
        if not _HAS_SKLEARN:
            raise RuntimeError("训练 PAD 模型需要 scikit-learn")
        Xa = _np.asarray(X, dtype=float)
        ya = _np.asarray(y, dtype=float)
        n = Xa.shape[0]
        if use_pca is None:
            # 文献2 自证：小样本下 PCA 会掉分 → 只有样本充足时才降维
            use_pca = n >= self.MIN_SAMPLES_FOR_PCA

        # ---- 交叉验证指标（先算，因为它决定 confidence 的可信度）----
        cv_metrics = self._cross_val(Xa, ya, cv_folds, use_pca)
        cv_ccc = {a: cv_metrics[a]["ccc"] for a in ("P", "A", "D")} \
            if cv_metrics else {}

        scaler = _StandardScaler().fit(Xa)
        Xs = scaler.transform(Xa)

        pca = None
        if use_pca:
            pca = _PCA(n_components=self.PCA_VAR_KEEP_STD, svd_solver="full")
            Xs = pca.fit_transform(Xs)

        models: Dict[str, object] = {}
        metrics: Dict[str, Dict[str, float]] = {}
        for i, axis in enumerate(("P", "A", "D")):
            svr = _SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
            svr.fit(Xs, ya[:, i])
            models[axis] = svr
            pred = svr.predict(Xs)
            metrics[axis] = _regression_metrics(ya[:, i], pred)
            if verbose:
                m = metrics[axis]
                cv = (cv_metrics or {}).get(axis, {})
                print(f"  {axis}: 样本内 MSE={m['mse']:.4f} ρ={m['rho']:.4f} "
                      f"ρc={m['ccc']:.4f}"
                      + (f" | CV ρc={cv.get('ccc', float('nan')):.4f}"
                         if cv else ""))

        # 有 CV 就用 CV 的 ρc（诚实口径）；没有则退回样本内并标注
        source_ccc = cv_ccc or {a: metrics[a]["ccc"] for a in ("P", "A", "D")}
        mean_ccc = sum(source_ccc[a] for a in ("P", "A", "D")) / 3.0
        self._bundle = {
            "models": models, "scaler": scaler, "pca": pca,
            "feature_names": list(feature_names),
            "metadata": {"source": "self-trained", "n_samples": n,
                         "use_pca": use_pca,
                         "ccc_source": "cv" if cv_ccc else "in_sample",
                         "pca_explained": (float(sum(pca.explained_variance_ratio_))
                                           if pca is not None else 1.0),
                         "ccc": {"P": source_ccc["P"],
                                 "A": source_ccc["A"],
                                 "D": source_ccc["D"],
                                 "mean": round(mean_ccc, 4)}},
        }
        self.available = True
        return {"n_samples": n, "use_pca": use_pca,
                "metrics": metrics, "cv_metrics": cv_metrics,
                "ccc_source": "cv" if cv_ccc else "in_sample",
                "mean_ccc": round(mean_ccc, 4)}

    # ------------------------------------------------------------------
    @staticmethod
    def _cross_val(Xa, ya, cv_folds: int, use_pca: bool) -> Dict[str, Dict]:
        """K 折交叉验证（逐维，返回三轴的 MSE/ρ/ρc）

        为什么必须做：SVR 在样本量不大时能"记住"训练点，样本内 ρc 会
        接近 1，用它当置信度会让整个 PAD 链路显得比实际可靠得多。
        文献2 报的 P 0.78 / A 0.46 / D 0.30 **也是**交叉验证口径。
        """
        n = Xa.shape[0]
        if cv_folds < 2 or n < cv_folds * 2:
            return {}
        try:
            from sklearn.model_selection import KFold
        except ImportError:  # pragma: no cover
            return {}
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        oof = _np.zeros_like(ya, dtype=float)
        try:
            for tr, te in kf.split(Xa):
                sc = _StandardScaler().fit(Xa[tr])
                Xtr, Xte = sc.transform(Xa[tr]), sc.transform(Xa[te])
                pca = None
                if use_pca:
                    pca = _PCA(n_components=PADRegressor.PCA_VAR_KEEP_STD,
                               svd_solver="full")
                    Xtr = pca.fit_transform(Xtr)
                    Xte = pca.transform(Xte)
                for i in range(ya.shape[1]):
                    svr = _SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
                    svr.fit(Xtr, ya[tr, i])
                    oof[te, i] = svr.predict(Xte)
        except Exception:  # noqa: BLE001 小样本/退化输入时放弃 CV
            return {}
        return {axis: _regression_metrics(ya[:, i], oof[:, i])
                for i, axis in enumerate(("P", "A", "D"))}

    def save(self, path: str):
        import joblib
        if self._bundle is None:
            raise RuntimeError("尚无已训练的模型")
        joblib.dump(self._bundle, path)
        self.model_path = path


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """文献2 式(11)(12)(13)：MSE / ρ / 一致性相关系数 CCC(ρc)

    ρc = 2·ρ·σy·σŷ / (σy² + σŷ² + (ȳ − ŷ̄)²)
    ρc 同时惩罚"相关性差"与"系统性偏移"，比单纯 ρ 更能反映可用性。
    """
    yt = _np.asarray(y_true, dtype=float)
    yp = _np.asarray(y_pred, dtype=float)
    n = len(yt)
    if n == 0:
        return {"mse": 0.0, "rho": 0.0, "ccc": 0.0}
    mse = float(_np.mean((yt - yp) ** 2))
    if n < 2:
        return {"mse": round(mse, 4), "rho": 0.0, "ccc": 0.0}
    my, mp = float(_np.mean(yt)), float(_np.mean(yp))
    vy = float(_np.var(yt)), float(_np.var(yp))
    cov = float(_np.mean((yt - my) * (yp - mp)))
    sy, sp = math.sqrt(vy[0]), math.sqrt(vy[1])
    rho = cov / (sy * sp) if sy > 1e-12 and sp > 1e-12 else 0.0
    denom = vy[0] + vy[1] + (my - mp) ** 2
    ccc = (2 * cov / denom) if denom > 1e-12 else 0.0
    return {"mse": round(mse, 4), "rho": round(rho, 4), "ccc": round(ccc, 4)}


# ======================================================================
# 四、统一入口
# ======================================================================
def estimate_pad(feats: Dict[str, float],
                 baseline: Optional[Dict[str, float]] = None,
                 regressor: Optional[PADRegressor] = None,
                 prefer: str = "auto") -> PADResult:
    """PAD 统一估计入口（SVR 可用则优先，否则规则映射兜底）

    Args:
        feats: HRV 特征字典
        baseline: 个体平静期基线
        regressor: 已训练的 PADRegressor（可选）
        prefer: "auto" 优先 SVR；"rule" 强制走规则路径

    Returns:
        PADResult。下游可用 ``.confidence`` / ``.d_confidence`` 判断可信度，
        低置信时（如 <0.4）建议**不要**让 PAD 主导对话策略。
    """
    if prefer != "rule" and regressor is not None and regressor.available:
        r = regressor.predict(feats)
        if r is not None:
            return r
    return RulePAD().predict(feats, baseline)


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    print("=" * 66)
    print("PAD 维度情感模型自测（文献2）")
    print("=" * 66)

    print(f"\nnumpy={_HAS_NUMPY}  sklearn={_HAS_SKLEARN}")

    # --- 量表打分演示 ---
    # 注意 PAD_SCALE_ITEMS 中 V2/V4/V6/V8/V9/V12 是反向题，
    # 故"分值越大越靠右端词"，方向已由式(1) 的系数统一。
    # 焦虑：愤怒/轻蔑/不满/不快(P−)、清醒/激动/专注/兴奋(A+)、受控/被支配/被影响/被忽视(D−)
    anxious = [-3, -3, -3, 3, 3, 3, -3, -3, -3, 3, 3, 3]
    # 放松：感兴趣/友好/满足/愉快(P+)、困倦/平静/分心/沉静(A−)、主控/支配/有影响力/重要(D+)
    relaxed = [3, 3, 3, -3, -3, -3, 3, 3, 3, -3, -3, -3]
    for label, v in (("焦虑", anxious), ("放松", relaxed)):
        pad = pad_from_questionnaire(v)
        tags = []
        if pad["P"] < 0 < pad["A"] and pad["D"] < 0:
            tags.append("命中文献焦虑判据")
        if pad["P"] > 0 > pad["A"] and pad["D"] > 0:
            tags.append("命中文献放松判据")
        print(f"  {label}: P={pad['P']:+.2f} A={pad['A']:+.2f} D={pad['D']:+.2f}"
              f"  [{'/'.join(tags) or '—'}]  → 本项目分类: "
              f"{pad_to_emotion(pad['P'], pad['A'], pad['D'])}")

    # --- 规则映射：平静 vs 焦虑 ---
    baseline = {"hr_bpm": 70.0, "rmssd_ms": 45.0, "sdnn_ms": 55.0, "pnn50_pct": 18.0}
    calm = {"hr_bpm": 70.0, "rmssd_ms": 45.0, "sdnn_ms": 55.0, "pnn50_pct": 18.0,
            "lf_hf": 1.5, "sd_hr_bpm": 3.0}
    stress = {"hr_bpm": 92.0, "rmssd_ms": 16.0, "sdnn_ms": 22.0, "pnn50_pct": 3.0,
              "lf_hf": 4.5, "sd_hr_bpm": 6.0}
    happy = {"hr_bpm": 78.0, "rmssd_ms": 55.0, "sdnn_ms": 62.0, "pnn50_pct": 24.0,
             "lf_hf": 2.2, "sd_hr_bpm": 4.0}
    tired = {"hr_bpm": 60.0, "rmssd_ms": 28.0, "sdnn_ms": 34.0, "pnn50_pct": 7.0,
             "lf_hf": 1.2, "sd_hr_bpm": 2.0}
    rp = RulePAD()
    print("\n规则映射（以个体平静基线为参照）:")
    for label, f in (("平静", calm), ("高焦虑", stress),
                     ("愉悦", happy), ("疲劳", tired)):
        r = rp.predict(f, baseline)
        print(f"  {label:<4} P={r.P:+5.2f} A={r.A:+5.2f} D={r.D:+5.2f} "
              f"conf={r.confidence:.2f} → {r.emotion_hint}")

    # --- SVR 流程自检（合成数据）---
    if _HAS_SKLEARN:
        rng = _np.random.RandomState(42)
        n = 240
        X = rng.randn(n, len(PAD_FEATURE_NAMES)) * 10 + 50
        # 让标签与若干特征线性相关，验证回归链路可学
        y = _np.stack([
            (X[:, 4] - 40) / 12.0,          # P ← RMSSD
            (X[:, 2] - 70) / 10.0,          # A ← HR
            (X[:, 1] - 50) / 15.0,          # D ← SDNN
        ], axis=1)
        y = _np.clip(y, -4, 4)
        reg = PADRegressor()
        print("\nPCA-SVR 训练自检（合成数据，仅验证流程与指标计算）:")
        res = reg.fit(X, y, verbose=True)
        print(f"  n={res['n_samples']} use_pca={res['use_pca']} "
              f"mean_CCC={res['mean_ccc']}")
        pred = reg.predict({n_: 50.0 for n_ in PAD_FEATURE_NAMES})
        print(f"  预测示例: {pred.to_dict()}")
