# -*- coding: utf-8 -*-
"""
情感动力学状态机（PWE · Plutchik's Wheel of Emotions）
=====================================================
对应计划书 §4.2「算法子模块 2.5」—— 为「主动式陪伴」提供**情感连续性**。

为什么需要这一层（问题陈述）
--------------------------------------------------------------------------------
v1 的三层识别（规则→模糊→ML）每次只回答「**此刻**属于哪一类」，是**无记忆的
离散硬分类**。由此产生三个在陪伴场景中很致命的问题：

  1. **情感跳变**：相邻两次推断在"平静/低焦虑"之间来回横跳，Agent 却像换了个人；
  2. **无累积**：连续 30 分钟的低落无法形成"今天状态不好"的整体判断，
     系统只会一遍遍重复同样强度的问候；
  3. **无衰减**：一次误判（如运动伪迹造成的假高焦虑）会立刻拉满警报并写入记忆，
     无法自然消退。

文献3《基于 PWE 模型的实时情绪识别与情感对话系统》（战勇胜等，
计算机系统应用 2026, 35(9): 98-106）给出的解法是：把情绪建成一个
**受刺激驱动、随时间指数衰减、并向"心境(mood)"累积**的动力学系统。

⚠️ 术语澄清（易错）：文献3 的 **PWE = Plutchik's Wheel of Emotions（普拉奇克
情绪轮）**，*不是* "Probability Weighted Entropy（概率加权熵）"。
本模块因此实现的是**情绪轮上的强度动力学**，不涉及任何熵计算。

模型结构
--------------------------------------------------------------------------------
状态量：
  * ``Et[s]``   —— 情绪 s 的当前情感强度 ∈ [0, 1]（t 时刻）
  * ``di[s]``   —— 情绪 s 的剩余持续时长（s）—— 决定衰减快慢
  * ``Mt``      —— 心境（mood）极性 ∈ [−1, +1]，跨轮次的慢变量
  * ``M(s)``    —— 心境强度 ∈ [0, 1]
  * ``m``       —— 心境极性符号 ∈ {+1(积极), 0(中性), −1(消极)}

更新方程（严格对应文献3 式(11)–(20)，编号沿用原文）
  (11) 有效刺激强度      Ci = clip( P(e_i) + m[i]·kmi[i]·M(s), 0, 1 )
  (12) 刺激强于情感时    Et = Ci ;                        di = 8.0(积极) / 5.0(其他)
  (13) 刺激弱于情感时    Et = E_{t-2} ;                   di = di · f'
       其中延长因子     f' = 1.5(积极) / 1.1(其他)   ← 积极情感衰减更慢
  (14) 被心境抑制但有残留 Et = max(E_{t-2}, P) ;           di = 5.0
  (15) 判定为干扰刺激     Et = 0 ;                         di = 0
  (17) 自然指数衰减      Et = E_{t-1} · exp( −(ln E_{t-1} − ln hi1) / di · Δt )
  (18) 心境增量          Δm = (1/n) · Σ_i m[i]·kei[i]·Ci[i]
  (19) 心境更新          Mt = clip( M_{t-2} + Δm, −1, +1 )
  (20) 心境衰减          Mt = M_{t-1}·exp(−dm·Δt)  (+0.05 偏置 当 Mt−1 ≤ −0.1)
  (16) 强度分级          unaware / mild / moderate / intense

两个设计意图（文献3 明确论述，直接服务于本项目定位）
  * **强化积极情感**：积极情感的持续时长 5 s → 8 s、衰减延长因子 1.1 → 1.5。
    对"陪伴"产品是对的——让好心情留得久一点。
  * **消极心境加速回归中性**：式(20) 的 +0.05 偏置。这是**安全性设计**，
    防止负性心境在被反复强化后长期滞留，对心理健康场景是必要的兜底。

实现边界（必须诚实标注）
  * 文献3 的`kmi[i]`（心境对情绪的影响系数）与`kei[i]`（情绪对心境的影响权重）
    **原文未给出具体数值**。本模块采用按情绪效价设定的经验值（见 ``KMI``/``KEI``），
    **属于工程假设，需在 20 人实验中标定**，不可当作文献结论引用。
  * 文献3 还把情感强度映射到 **Transformer 内部超参**（注意力头数/隐藏层维度/
    层数等，式27–33）。本项目调用**云端 API 模型**，无法改动模型内部结构，
    因此只落地**外部超参**（temperature / top_p / top_k / 重复惩罚）与
    system prompt 语气调制 —— 见 ``agent/emotion_strategy.py``。

参考文献
--------
[3] 战勇胜, 王宇鹏, 徐世晨, 等. 基于 PWE 模型的实时情绪识别与情感对话系统[J].
    计算机系统应用, 2026, 35(9): 98-106. DOI: 10.15888/j.cnki.csa.010241
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ======================================================================
# 情绪轮定义（Plutchik 8 种基础情感，文献3 采用）
# ======================================================================
#: 情绪轮上的 8 种基础情感 + 本项目补充的 boring（文献3 图中含 Boring）
WHEEL_EMOTIONS: Tuple[str, ...] = (
    "joy", "trust", "anticipation", "surprise",
    "sadness", "disgust", "fear", "anger", "boring",
)

#: 中文名（供日志与前端展示）
WHEEL_ZH: Dict[str, str] = {
    "joy": "喜悦", "trust": "信任", "anticipation": "期待", "surprise": "惊讶",
    "sadness": "悲伤", "disgust": "厌恶", "fear": "恐惧", "anger": "愤怒",
    "boring": "乏味",
}

#: 效价分类 —— 决定"积极情感强化"与"心境极性"的符号
POSITIVE_EMOTIONS = frozenset({"joy", "trust", "anticipation"})
NEGATIVE_EMOTIONS = frozenset({"sadness", "disgust", "fear", "anger", "boring"})

#: 情绪 → 效价符号 m[i]（+1 积极 / 0 中性 / −1 消极）
EMOTION_POLARITY: Dict[str, int] = {
    e: (1 if e in POSITIVE_EMOTIONS else (-1 if e in NEGATIVE_EMOTIONS else 0))
    for e in WHEEL_EMOTIONS
}

#: kmi[i] —— 心境对情绪 i 的调制系数（式11）
#: ⚠️ 文献3 未给数值，此处按"消极心境放大负性情绪、抑制正性情绪"的心理学
#:    常识设定，量级参考原文表述（心境影响是**调制项**而非主导项）。
KMI: Dict[str, float] = {
    "joy": 0.25, "trust": 0.20, "anticipation": 0.20, "surprise": 0.10,
    "sadness": 0.35, "disgust": 0.30, "fear": 0.40, "anger": 0.35,
    "boring": 0.25,
}

#: kei[i] —— 情绪 i 对心境的贡献权重（式18）
#: ⚠️ 同上，属工程经验值。量级设计原则：高唤醒负性情绪对心境冲击最大，
#:    积极情绪次之，中性情绪最小。
KEI: Dict[str, float] = {
    "joy": 0.9, "trust": 0.6, "anticipation": 0.5, "surprise": 0.3,
    "sadness": 1.0, "disgust": 0.7, "fear": 1.0, "anger": 0.9,
    "boring": 0.5,
}

#: 积极/其他 的持续时长与延长因子（式12、13）
DURATION_POSITIVE = 8.0
DURATION_OTHER = 5.0
EXTEND_FACTOR_POSITIVE = 1.5
EXTEND_FACTOR_OTHER = 1.1
#: 持续时长上限（工程加固）—— 文献式(13) 在刺激持续存在时会让 di 无界增长，
#: 导致衰减停滞、情绪被"粘住"。上限保证任何情绪都有消退通路。
DURATION_MAX = 60.0

#: 强度分级阈值 hi1 < hi2 < hi3（式16）
LEVEL_THRESHOLDS: Tuple[float, float, float] = (0.15, 0.40, 0.70)
LEVEL_NAMES: Tuple[str, ...] = ("unaware", "mild", "moderate", "intense")

#: **熄灭容差** —— 补上式(17) 离散形式的一个数学漏洞
#:
#: 式(17) 的更新可整理为 ``E_{t+1} = hi1 · (E_t / hi1)^(1 − dt/di)``，
#: 即 **hi1 是一个吸引不动点**：强度严格递减但永远只无限逼近 hi1。
#: 因此原文写的"E ≤ hi1 → 情绪熄灭"在数学上**不可达**，会带来两个后果：
#:   1. durations 永不归零 → 下游"这份难受已经跟着你一阵子了"这类
#:      描述会挂在一个早已消退的情绪上；
#:   2. 主导情绪判据必须改用 hi1（见 snapshot），否则状态层长期报
#:      "unaware 级别的非零情绪"。
#: 这里用相对容差 1e-3 收口：强度距 hi1 不足 0.1% 即认定已熄灭。
#: 该容差只影响"何时归零"，不影响任何衰减轨迹。
EXTINCT_TOL = 1e-3

#: 心境衰减因子 dm（式20）
MOOD_DECAY = 0.05
#: 消极心境回归中性的偏置（式20 的安全性设计）
NEGATIVE_MOOD_BIAS = 0.05
NEGATIVE_MOOD_GATE = -0.1


# ======================================================================
# 用户人格参数（文献3 表2 的四种人格，直接可配置）
# ======================================================================
#: 文献3 表2 原文数值。用于让"小暖"适配不同气质的用户：
#:   理想型  —— 均衡理性、情感曲线平滑可预测（默认）
#:   冷漠型  —— 情感幅度窄、高衰减、高阈值（情绪不敏感/不轻易波动）
#:   敏感型  —— 情感曲线不稳定、易走极端（交互强度 2.0 即放大刺激）
#:   执着型  —— 情感衰减小、心境平滑有惯性、不易放下
PERSONALITY_PROFILES: Dict[str, Dict[str, float]] = {
    "ideal":      {"alpha": 0.05, "beta": 0.08, "tm": 2.0, "threshold": 0.45,
                   "interaction": 1.0, "fusion": "linear", "zh": "理想型"},
    "indifferent": {"alpha": 0.15, "beta": 0.08, "tm": 2.0, "threshold": 0.65,
                    "interaction": 1.0, "fusion": "linear", "zh": "冷漠型"},
    "sensitive":  {"alpha": 0.05, "beta": 0.08, "tm": 2.0, "threshold": 0.45,
                   "interaction": 2.0, "fusion": "n-linear", "zh": "敏感型"},
    "persistent": {"alpha": 0.02, "beta": 0.08, "tm": 2.0, "threshold": 0.45,
                   "interaction": 1.0, "fusion": "adaptive", "zh": "执着型"},
}

#: **已接入**动力学的参数 —— 改动它们会真实改变输出
#:   alpha       式(17) 衰减速率的缩放（alpha / MOOD_DECAY）
#:   interaction 式(11) 对刺激强度的放大（文献3 表2 的 C-c 交互强度）
WIRED_PROFILE_PARAMS: Tuple[str, ...] = ("alpha", "interaction")

#: **保留但尚未接入**的参数 —— 改动它们**不会**改变任何输出
#:
#: ⚠️ 这一条必须写清楚，否则会误导实验设计。文献3 表2 给出的字段是
#:    (α, β, tm, s, C-c, fusion) 六个，但论文未给出 β / tm / fusion 在
#:    式(11)–(20) 中的确切位置（它们属于原文"情感融合"环节，本次精读
#:    未能定位到对应公式）。为避免**凭空实现**出一个看似能调、实则语义
#:    无据的参数，这里选择如实标注为未接入，而不是补一个自己编的公式。
#:
#: 直接后果（写进说明书与论文，不要漏）：
#:   1. 四档人格**确实**有行为差异（ideal: α=0.05/C-c=1.0；indifferent:
#:      α=0.15；sensitive: C-c=2.0；persistent: α=0.02），所以"切换人格
#:      会改变响应"这句话是成立的。
#:   2. 但「冷漠型」的**高阈值语义没有生效** —— 它当前只表现为"衰减更快"，
#:      而不表现为"需要更强的刺激才能被激起"。同理「敏感型」的
#:      n-linear 融合、「执着型」的自适应融合都未实现。
#:      因此对外描述这四档时，只能说"调节情感衰减速度与刺激放大倍数"，
#:      **不可**说"调节情感阈值"。
RESERVED_PROFILE_PARAMS: Tuple[str, ...] = ("beta", "tm", "threshold", "fusion")


@dataclass
class AffectState:
    """情感动力学状态快照（可直接注入 Prompt / 上报前端）

    ⚠️ **两套情绪标签不要混用**（这是本类最容易踩的坑）：

      * ``dominant`` / ``dominant_zh`` 是 **Plutchik 情绪轮的 9 类**
        （fear / anger / joy … → 恐惧 / 愤怒 / 喜悦 …），
        服务于动力学内部的衰减与心境计算；
      * 本项目对用户与 Prompt 用的是 **5 类中文标签**
        （平静/低焦虑/高焦虑/愉悦/疲劳）。

    内部消费（trigger_engine 的强度加分、emotion_strategy 的持续性描述）
    直接用 ``dominant`` 是正确的；但**任何要展示给人或写进 Prompt 的地方**，
    都应使用 ``emotion_zh``（本字段已由 to_dict 提供）或
    ``EmotionDynamics.effective_emotion_zh()``，
    否则会出现"系统说用户是恐惧，而系统提示词里只定义了 5 类"的错位。
    """
    intensities: Dict[str, float] = field(default_factory=dict)
    durations: Dict[str, float] = field(default_factory=dict)
    mood: float = 0.0                     # Mt ∈ [−1, 1]
    mood_strength: float = 0.0            # M(s) ∈ [0, 1]
    polarity: int = 0                     # m ∈ {−1, 0, +1}
    dominant: str = ""                    # 主导情绪（**情绪轮**键名，非项目 5 类）
    dominant_zh: str = ""                 # 上者的中文名（**仅供展示**）
    dominant_intensity: float = 0.0
    level: str = "unaware"                # unaware/mild/moderate/intense
    probabilities: Dict[str, float] = field(default_factory=dict)
    updated_at: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "intensities": {k: round(v, 4) for k, v in self.intensities.items()},
            "durations": {k: round(v, 2) for k, v in self.durations.items()},
            "mood": round(self.mood, 4),
            "mood_strength": round(self.mood_strength, 4),
            "polarity": self.polarity,
            "dominant": self.dominant,
            "dominant_zh": self.dominant_zh,
            "dominant_intensity": round(self.dominant_intensity, 4),
            # 项目 5 类标签 —— 对外展示 / 注入 Prompt 请用这个，
            # 不要用 dominant_zh（那是情绪轮的 9 类名）
            "emotion_zh": (WHEEL_TO_ZH.get(self.dominant, "平静")
                           if self.dominant else "平静"),
            "level": self.level,
            "probabilities": {k: round(v, 4)
                              for k, v in self.probabilities.items()},
            "updated_at": self.updated_at,
        }


# ======================================================================
# 5 类中文情绪 ⇄ 情绪轮 的双向映射（本项目特有，文献3 无此层）
# ======================================================================
#: 本项目 5 类 → 情绪轮的权重分解（一行和为 1）
#:   设计依据：本项目的「高焦虑」在效价-唤醒二维空间上覆盖"高唤醒+负效价"，
#:   对应情绪轮上的 fear（主导）与 anger（次要）；「疲劳」为低唤醒负效价，
#:   对应 sadness + boring。该分解使动力学层能同时服务两套标签体系。
ZH_TO_WHEEL: Dict[str, Dict[str, float]] = {
    "平静": {},
    "愉悦": {"joy": 0.75, "trust": 0.25},
    "低焦虑": {"fear": 0.55, "anticipation": 0.25, "anger": 0.20},
    "高焦虑": {"fear": 0.60, "anger": 0.30, "disgust": 0.10},
    "疲劳": {"boring": 0.45, "sadness": 0.45, "disgust": 0.10},
}

#: 情绪轮 → 本项目 5 类（用于把动力学结果反向表述给用户）
WHEEL_TO_ZH: Dict[str, str] = {
    "joy": "愉悦", "trust": "愉悦", "anticipation": "低焦虑",
    "surprise": "低焦虑", "sadness": "疲劳", "disgust": "高焦虑",
    "fear": "高焦虑", "anger": "高焦虑", "boring": "疲劳",
}


class EmotionDynamics:
    """PWE 情感动力学状态机（文献3 式(11)–(20) 的工程实现）

    用法::

        dyn = EmotionDynamics(personality="ideal")
        # 每收到一次情绪识别结果（5 类中文概率）就推进一次
        st = dyn.update_from_zh({"高焦虑": 0.6, "低焦虑": 0.3, "平静": 0.1}, dt=1.0)
        print(st.dominant_zh, st.level, st.mood)

    时间语义：``dt`` 为距上次更新的秒数。固件 1 包/秒 → dt≈1.0。
    """

    def __init__(self, personality: str = "ideal",
                 hi: Tuple[float, float, float] = LEVEL_THRESHOLDS,
                 mood_decay: float = MOOD_DECAY):
        if personality not in PERSONALITY_PROFILES:
            personality = "ideal"
        self.personality_key = personality
        self.personality = PERSONALITY_PROFILES[personality]
        # 人格的 α（衰减系数）与 C-c（交互强度）参与动力学 —— 见
        # WIRED_PROFILE_PARAMS 的说明；s（阈值）等字段目前**不参与**计算，
        # 保留在此仅为与文献3 表2 的字段一一对应，便于日后接入。
        self.alpha = float(self.personality["alpha"])
        self.threshold = float(self.personality["threshold"])   # 未接入
        self.interaction = float(self.personality["interaction"])
        self.hi = tuple(hi)
        self.mood_decay = float(mood_decay)

        self.intensities: Dict[str, float] = {e: 0.0 for e in WHEEL_EMOTIONS}
        self.durations: Dict[str, float] = {e: 0.0 for e in WHEEL_EMOTIONS}
        self._prev_intensities: Dict[str, float] = {e: 0.0 for e in WHEEL_EMOTIONS}
        self.mood = 0.0
        self.mood_strength = 0.0
        self._last_update: Optional[float] = None
        self.probabilities: Dict[str, float] = {e: 0.0 for e in WHEEL_EMOTIONS}

    # ------------------------------------------------------------ 映射层
    @staticmethod
    def zh_to_wheel(prob_zh: Dict[str, float]) -> Dict[str, float]:
        """5 类中文情绪概率 → 情绪轮 9 维概率（线性加权分解）

        注意：**不做归一化**。因为「平静」在情绪轮上不贡献任何维度
        （平静 = 无情绪刺激），若在映射后归一化，会把平静的概率份额
        按比例分摊给其余情绪 → 凭空放大刺激强度。
        保留原始量纲后，Σ P(e) ≤ 1，不足的部分正好表示"无情绪刺激"。
        """
        out = {e: 0.0 for e in WHEEL_EMOTIONS}
        for zh, p in (prob_zh or {}).items():
            w = ZH_TO_WHEEL.get(zh)
            if not w or p is None:
                continue
            try:
                pv = float(p)
            except (TypeError, ValueError):
                continue
            for e, ratio in w.items():
                out[e] += pv * ratio
        return out

    # ------------------------------------------------------------ 核心更新
    def update(self, probs: Dict[str, float], dt: float = 1.0,
               now: Optional[float] = None) -> AffectState:
        """推进一个时间步（文献3 式(11)–(20)）

        Args:
            probs: 情绪轮 9 维概率 P(e)，和不必为 1（内部按刺激强度处理）
            dt: 距上次更新秒数（必须为正；首帧自动钳位为 1.0）
            now: 当前时间戳（测试注入用）

        Returns:
            AffectState
        """
        dt = max(1e-3, min(float(dt), 600.0))   # 防止时钟跳变导致爆炸
        now = now if now is not None else time.time()
        self._last_update = now

        # 人格的交互强度放大刺激（文献3 表2「交互强度 C-c」：敏感型 = 2.0）
        stim = {e: float(probs.get(e, 0.0)) * self.interaction
                for e in WHEEL_EMOTIONS}
        self.probabilities = dict(probs)

        new_int: Dict[str, float] = {}
        new_dur: Dict[str, float] = {}
        # 保留上一拍强度用于式(13)(14) 的 E_{t-2} 语义
        prev = dict(self.intensities)

        for e in WHEEL_EMOTIONS:
            e_prev = prev.get(e, 0.0)
            p = stim.get(e, 0.0)
            # ---- 式(11) 有效刺激强度 = 情绪概率 + 心境调制 ----
            # 调制项 = 情绪效价 × kmi × mood（**mood 必须带符号**）
            #   消极心境(mood<0) → 负性情绪被放大、正性情绪被抑制
            #   积极心境(mood>0) → 正性情绪被放大、负性情绪被抑制
            # 若误用 |mood|，会得出"消极心境抑制消极情绪"的反向结论。
            mood_term = EMOTION_POLARITY[e] * KMI[e] * self.mood
            c = p + mood_term
            c = max(0.0, min(1.0, c))

            is_pos = e in POSITIVE_EMOTIONS
            if c > e_prev:
                # ---- 式(12) 刺激强于当前情感 → 直接跃升 ----
                new_int[e] = c
                new_dur[e] = DURATION_POSITIVE if is_pos else DURATION_OTHER
            elif c > 0.0:
                # ---- 式(13) 刺激弱于当前情感 → 不升高，但延长持续时间 ----
                new_int[e] = e_prev
                f = EXTEND_FACTOR_POSITIVE if is_pos else EXTEND_FACTOR_OTHER
                # 工程加固：文献式(13) 每步都乘延长因子，若刺激持续存在
                # （如长时间焦虑）di 会指数增长 → 衰减趋近停滞、情绪"粘住"。
                # 这里对 di 设上限，保证情绪总有自然消退的可能。
                new_dur[e] = min(
                    (self.durations.get(e, 0.0) or DURATION_OTHER) * f,
                    DURATION_MAX)
            elif p + e_prev > mood_term:
                # ---- 式(14) 被心境抑制但仍有残留 → 缓慢更新 ----
                new_int[e] = max(e_prev, p)
                new_dur[e] = DURATION_OTHER
            else:
                # ---- 式(15) 判定为干扰刺激 → 清零 ----
                new_int[e] = 0.0
                new_dur[e] = 0.0

        # ---- 式(17) 自然指数衰减（严格按文献原式） ----
        #     Et = E_{t-1} · exp( −(ln E_{t-1} − ln hi1) / di · Δt )
        # 注意该式的特性：**衰减速率依赖当前强度**——E 越大衰减越快，
        # 越接近阈值 hi1 衰减越慢（情绪尾巴拖长），比固定时间常数更贴近
        # "强烈情绪消退快、余韵绵长"的直觉。人格系数 α 作为速率缩放。
        hi1 = self.hi[0]
        alpha_scale = self.alpha / MOOD_DECAY      # 理想型 = 1.0
        # 熄灭门限：hi1 是不动点，必须留容差才能真的熄灭（见 EXTINCT_TOL）
        extinct_at = hi1 * (1.0 + EXTINCT_TOL)
        for e in WHEEL_EMOTIONS:
            et = new_int[e]
            di = new_dur[e]
            if et <= extinct_at:
                # 跌破"算得上情绪"的下界 → 情绪熄灭，持续时间一并归零
                new_int[e] = 0.0
                new_dur[e] = 0.0
                continue
            if di <= 0:
                new_int[e] = et
                continue
            decay_exp = (math.log(et) - math.log(hi1)) / di * dt * alpha_scale
            new_int[e] = et * math.exp(-max(0.0, decay_exp))

        self.intensities = new_int
        self.durations = new_dur

        # ---- 式(18)(19) 心境累积与更新 ----
        n = len(WHEEL_EMOTIONS)
        delta_m = sum(EMOTION_POLARITY[e] * KEI[e] * max(0.0, stim.get(e, 0.0))
                      for e in WHEEL_EMOTIONS) / n
        mood_prev = self.mood
        self.mood = max(-1.0, min(1.0, mood_prev + delta_m))

        # ---- 式(20) 心境自然衰减（消极心境额外 +0.05 加速回中性） ----
        decayed = self.mood * math.exp(-self.mood_decay * dt)
        if self.mood <= NEGATIVE_MOOD_GATE:
            decayed += NEGATIVE_MOOD_BIAS
        self.mood = max(-1.0, min(1.0, decayed))
        self.mood_strength = abs(self.mood)

        return self.snapshot()

    def update_from_zh(self, prob_zh: Dict[str, float], dt: float = 1.0,
                       now: Optional[float] = None) -> AffectState:
        """便捷入口：直接喂 5 类中文情绪概率"""
        return self.update(self.zh_to_wheel(prob_zh), dt=dt, now=now)

    # ------------------------------------------------------------ 输出
    def snapshot(self) -> AffectState:
        """生成当前状态快照（含主导情绪与强度分级）

        ⚠️ 主导情绪的判据是 ``lead_v > hi1`` 而不是 ``lead_v > 0``。
        这不是细节：式(17) 的衰减以 hi1 为**渐近下界**（见 EXTINCT_TOL），
        强度只会无限逼近 hi1 而不会真正到 0。若用 >0 判据，一个情绪
        消退十几分钟后系统仍会报出"当前主导情绪 = 恐惧"，
        而它的强度明明已经落在"无情绪（unaware）"区间 ——
        结果是 Prompt 与前端长期挂着一个不存在的情绪。
        用 hi1 作判据，正好与式(16) 的分级语义对齐：
        **低于 hi1 的强度，按定义就不构成一种情绪。**
        """
        lead = max(WHEEL_EMOTIONS, key=lambda e: self.intensities.get(e, 0.0))
        lead_v = self.intensities.get(lead, 0.0)
        hi1 = self.hi[0]
        if lead_v <= hi1:
            # 不足以构成情绪 → 主报告"平静"，强度归零（不把残余量当情绪上报）
            lead, lead_zh, lead_v = "", "平静", 0.0
        else:
            lead_zh = WHEEL_ZH.get(lead, lead)
        self.polarity = (1 if self.mood > 0.1 else (-1 if self.mood < -0.1 else 0))
        return AffectState(
            intensities=dict(self.intensities),
            durations=dict(self.durations),
            mood=self.mood,
            mood_strength=self.mood_strength,
            polarity=self.polarity,
            dominant=lead,
            dominant_zh=lead_zh,
            dominant_intensity=lead_v,
            level=classify_level(lead_v, self.hi),
            probabilities=dict(self.probabilities),
            updated_at=self._last_update or time.time(),
        )

    # ------------------------------------------------------------ 辅助
    def effective_emotion_zh(self, fallback: str = "平静") -> str:
        """把动力学主导情绪折算回本项目 5 类标签（供下游沿用旧接口）"""
        st = self.snapshot()
        if not st.dominant:
            return fallback
        return WHEEL_TO_ZH.get(st.dominant, fallback)

    def intensity_of_zh(self, emotion_zh: str) -> float:
        """查询某中文情绪对应的动力学强度（其映射维度的加权和）"""
        w = ZH_TO_WHEEL.get(emotion_zh, {})
        return sum(self.intensities.get(e, 0.0) * r for e, r in w.items())

    def reset(self):
        self.intensities = {e: 0.0 for e in WHEEL_EMOTIONS}
        self.durations = {e: 0.0 for e in WHEEL_EMOTIONS}
        self.mood = 0.0
        self.mood_strength = 0.0
        self.probabilities = {e: 0.0 for e in WHEEL_EMOTIONS}


def classify_level(intensity: float,
                   hi: Tuple[float, float, float] = LEVEL_THRESHOLDS) -> str:
    """式(16) 情感强度分级：unaware / mild / moderate / intense"""
    hi1, hi2, hi3 = hi
    if intensity <= hi1:
        return "unaware"
    if intensity <= hi2:
        return "mild"
    if intensity <= hi3:
        return "moderate"
    return "intense"


# ------------------------------------------------------------------ 自测
if __name__ == "__main__":
    print("=" * 66)
    print("PWE 情感动力学状态机自测（文献3 式(11)–(20)）")
    print("=" * 66)

    dyn = EmotionDynamics(personality="ideal")
    print("\n场景A：持续高焦虑 10 秒后转入平静，观察衰减与心境回落")
    for i in range(10):
        st = dyn.update_from_zh({"高焦虑": 0.75, "低焦虑": 0.2, "平静": 0.05}, dt=1.0)
        print(f"  t={i+1:>2}s  {st.dominant_zh:<4} 强度={st.dominant_intensity:.3f}"
              f" 级别={st.level:<9} 心境={st.mood:+.3f}")
    print("  --- 转为平静 ---")
    for i in range(12):
        st = dyn.update_from_zh({"平静": 1.0}, dt=1.0)
        print(f"  t={i+11:>2}s  {st.dominant_zh:<4} 强度={st.dominant_intensity:.3f}"
              f" 级别={st.level:<9} 心境={st.mood:+.3f}")

    print("\n场景B：积极情感强化（愉悦 → 持续时长 8s、延长因子 1.5）")
    d2 = EmotionDynamics()
    for i in range(4):
        st = d2.update_from_zh({"愉悦": 0.9}, dt=1.0)
        print(f"  t={i+1}s  {st.dominant_zh} 强度={st.dominant_intensity:.3f}"
              f" 时长={st.durations.get('joy', 0):.1f}s 心境={st.mood:+.3f}")

    print("\n场景C：四种人格对同一刺激的响应差异（连续 8 秒高焦虑）")
    for key in ("ideal", "indifferent", "sensitive", "persistent"):
        d = EmotionDynamics(personality=key)
        for _ in range(8):
            d.update_from_zh({"高焦虑": 0.6, "低焦虑": 0.3}, dt=1.0)
        st = d.snapshot()
        print(f"  {PERSONALITY_PROFILES[key]['zh']:<4} 强度={st.dominant_intensity:.3f}"
              f"  心境={st.mood:+.3f}  级别={st.level}")

    print("\n场景D：消极心境加速回中性（式20 的安全性设计）")
    d3 = EmotionDynamics()
    for _ in range(60):
        d3.update_from_zh({"高焦虑": 0.9}, dt=1.0)
    print(f"  持续高焦虑后 心境={d3.mood:+.3f}")
    for i in (5, 20, 60, 120):
        d4 = EmotionDynamics()
        for _ in range(60):
            d4.update_from_zh({"高焦虑": 0.9}, dt=1.0)
        for _ in range(i):
            d4.update_from_zh({"平静": 1.0}, dt=1.0)
        print(f"  平静 {i:>3}s 后  心境={d4.mood:+.4f}")
