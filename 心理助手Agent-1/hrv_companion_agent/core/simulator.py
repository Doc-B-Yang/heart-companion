# -*- coding: utf-8 -*-
"""
传感器数据模拟器
================
在没有 ESP32 硬件时，生成符合 BLE 数据包格式（计划书 §3.3）的合成生理数据，
用于: 算法验证 / 端到端闭环演示 / 单元测试。

数据真实性设计:
  * 按情绪场景定义目标 HR/SDNN/RMSSD 区间（与识别规则表同源）
  * IBI 序列由随机游走生成（相邻差标准差≈RMSSD），可注入运动伪迹
  * HRV 特征由 HRVExtractor 从 IBI 窗口真实计算（非直接伪造）
  * LF/HF 由计算值与场景目标值混合（真实硬件由 ESP32 计算，见说明书）
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

from ..algorithms.hrv_extractor import HRVExtractor

# 场景: (情绪, 时长秒) 序列 —— 演示"平静→压力事件→高焦虑→干预→恢复"闭环
#
# 【修复】各段时长必须显著长于滑动 IBI 窗口（SIM_IBI_WINDOW_SEC）。
#   原实现各段仅 10~30s，而窗口累积到 150 拍(≈2 分钟)，任何时刻窗口内都
#   混合着前后两种情绪 → SDNN 被拉平成一个与情绪无关的值（实测恒 ≈105ms），
#   高焦虑段永远无法落入规则表的 SDNN 10~30 区间。这不是算法问题, 是
#   仿真剧本的时长设计问题。
SIMULATION_SCENARIOS: Dict[str, List[Tuple[str, int]]] = {
    "stress_event": [  # 考试/汇报类压力事件（核心演示场景）
        ("平静", 60), ("低焦虑", 60), ("高焦虑", 120),
        ("低焦虑", 60), ("平静", 60),
    ],
    "anxiety_attack": [  # 突发焦虑
        ("平静", 60), ("高焦虑", 120), ("平静", 60),
    ],
    "afternoon_fatigue": [  # 午后疲劳
        ("平静", 60), ("疲劳", 120), ("平静", 60),
    ],
    "happy_chat": [  # 愉快聊天
        ("平静", 60), ("愉悦", 120), ("平静", 60),
    ],
}

# 滑动 IBI 窗口长度（秒）。取 45s: 既够 HRV 统计（≥30s 才有意义），
# 又短于最短场景段（60s），保证每段有"纯窗口"时段可被正确识别。
SIM_IBI_WINDOW_SEC = 45.0

# 场景目标特征区间（与 algorithms.emotion_recognizer.EMOTION_RULES 同源）
# lf_hf 取规则表区间中点: 规则层按此值区分焦虑等级。
SCENARIO_RANGES = {
    "平静": {"hr": (60, 75), "sdnn": (50, 90), "rmssd": (40, 70),
             "rr": (12, 16), "lf_hf": (1.0, 2.0)},
    "低焦虑": {"hr": (75, 90), "sdnn": (30, 50), "rmssd": (25, 40),
               "rr": (16, 20), "lf_hf": (2.0, 3.5)},
    "高焦虑": {"hr": (90, 110), "sdnn": (12, 30), "rmssd": (8, 25),
               "rr": (20, 26), "lf_hf": (3.5, 6.5)},
    "愉悦": {"hr": (65, 80), "sdnn": (40, 60), "rmssd": (35, 50),
             "rr": (14, 18), "lf_hf": (1.5, 2.5)},
    "疲劳": {"hr": (55, 70), "sdnn": (35, 45), "rmssd": (20, 30),
             "rr": (10, 15), "lf_hf": (1.0, 1.5)},
}

# RR 序列的三条振荡成分（Hz）。物理含义:
#   0.03Hz 极低频(VLF): 体温调节/血管舒缩节律, 是 SDNN 的主要来源
#   0.10Hz Mayer 波(LF): 压力反射
#   呼吸频率(RSA, HF): 副交感调制
# 必须建模 VLF, 否则仿真出的 SDNN 只能由 RMSSD 决定, 比值恒定,
# 「疲劳」(SDNN/RMSSD≈1.6) 与「平静」(≈1.27) 在 SDNN 上无法分开。
VLF_HZ = 0.03
LF_HZ = 0.10


class SensorSimulator:
    """ESP32 可穿戴终端模拟器（输出 BLE 数据包，1 包/秒）"""

    def __init__(self, scenario: str = "stress_event", seed: int = 42,
                 artifact_prob: float = 0.02):
        self.scenario = scenario
        self.phases: List[Tuple[str, int]] = SIMULATION_SCENARIOS.get(
            scenario, SIMULATION_SCENARIOS["stress_event"])
        self.artifact_prob = artifact_prob
        self.rng = random.Random(seed)
        self.extractor = HRVExtractor()

        self._phase_idx = 0
        self._phase_left = self.phases[0][1]
        self._targets = self._make_targets(self.phases[0][0])
        self._window: List[float] = []   # 滚动 IBI 窗口
        self._window_t: List[float] = []  # 与之对应的累计时间轴(秒)
        self._t_sec = 0.0                # 累计时间(秒), 作为振荡器相位基准
        self._tick = 0

    # ------------------------------------------------------------ 目标生成
    def _make_targets(self, emotion: str) -> Dict[str, float]:
        r = SCENARIO_RANGES[emotion]
        hr = self.rng.uniform(*r["hr"])
        sdnn = self.rng.uniform(*r["sdnn"])
        rmssd = min(self.rng.uniform(*r["rmssd"]), sdnn * 1.2)
        rr = self.rng.uniform(*r["rr"])
        lf_hf = self.rng.uniform(*r["lf_hf"])
        return {"emotion": emotion, "hr": hr, "sdnn": sdnn,
                "rmssd": rmssd, "rr": rr, "lf_hf": lf_hf}

    def _advance_phase(self):
        self._phase_idx += 1
        if self._phase_idx >= len(self.phases):
            return False
        self._phase_left = self.phases[self._phase_idx][1]
        self._targets = self._make_targets(self.phases[self._phase_idx][0])
        return True

    # ------------------------------------------------------------ 心跳生成
    def _next_ibi(self) -> float:
        """按场景目标 (HR / SDNN / RMSSD / 呼吸率 / LF-HF) 生成下一个 IBI。

        原实现只用 rmssd 设步长的高斯随机游走, 结果是:
          * SDNN 由随机游走在长窗口上累积, 与场景目标完全脱钩（实测恒 ≈105ms）;
          * 呼吸率只影响 tick() 里的 rr 字段, 与 IBI 序列无关 → RSA 不存在。
        ⇒ 仿真数据永远落不进规则表的高焦虑区间（SDNN 需 10~30）。

        现改为三条振荡成分 + 噪声，幅度由目标量联立解出:
          单频成分经逐拍采样后 RMSSD_i ≈ A_i·2πf_i/√2, 功率比 = A_L²/A_H²
          1) 由 RMSSD 与 LF/HF 解出 LF、HF 幅度;
          2) SDNN² 的余额交给 VLF（VLF 对 RMSSD 贡献小但主导 SDNN）;
          3) 回代一轮修正 RMSSD 预算。
        """
        t = self._targets
        base = 60000.0 / t["hr"]
        f_h = t["rr"] / 60.0                # 呼吸率 Hz → HF(RSA) 中心频率
        f_l = LF_HZ
        f_v = VLF_HZ
        ratio = t.get("lf_hf", 1.5)

        rmssd_t = t["rmssd"]
        sigma = rmssd_t * 0.25
        # RMSSD 的"逐拍差分增益": 对幅度 A、频率 f 的正弦, 相邻两拍之差约为
        # A·2·sin(π·f·Δt), 其中 Δt 是**逐拍采样间隔**（≈当前 RR，不是 1 秒）。
        # 原实现用低频近似 2πf 且忽略 Δt, 会让仿真出的 RMSSD 系统性偏低
        # 15~30%（呼吸越快越明显）。
        dt_s = base / 1000.0
        g_v = 2.0 * math.sin(math.pi * f_v * dt_s)
        g_l = 2.0 * math.sin(math.pi * f_l * dt_s)
        g_h = 2.0 * math.sin(math.pi * f_h * dt_s)

        a_v = 0.0
        a_l = a_h = 0.0
        for _ in range(2):
            v_rem = max(0.0, rmssd_t ** 2 - 2 * sigma ** 2 - (a_v * g_v) ** 2)
            denom = ratio * g_l ** 2 + g_h ** 2
            a_h = math.sqrt(2.0 * v_rem / denom) if denom > 0 else 0.0
            a_l = a_h * math.sqrt(ratio)
            used = (a_l ** 2 + a_h ** 2) / 2.0 + sigma ** 2
            rest = t["sdnn"] ** 2 - used
            a_v = math.sqrt(2.0 * rest) if rest > 0 else 0.0

        tsec = self._t_sec                  # 秒级时间轴, 保证相位连续
        nxt = base \
            + a_v * math.sin(2 * math.pi * f_v * tsec) \
            + a_l * math.sin(2 * math.pi * f_l * tsec) \
            + a_h * math.sin(2 * math.pi * f_h * tsec) \
            + self.rng.gauss(0, sigma)
        # 运动伪迹注入
        if self.rng.random() < self.artifact_prob:
            nxt += self.rng.choice([-1, 1]) * self.rng.uniform(150, 350)
        return max(300.0, min(2000.0, nxt))

    # ------------------------------------------------------------ 数据包
    def tick(self, ts: Optional[float] = None) -> Optional[Dict]:
        """产生 1 秒数据包；场景播完返回 None"""
        if self._phase_left <= 0 and not self._advance_phase():
            return None
        self._phase_left -= 1
        self._tick += 1
        ts = ts if ts is not None else time.time()

        # 每秒产生的心搏数应与当前心率一致（60~110bpm → 1~1.8 次/秒）。
        # 原实现固定 1 次/秒, 相当于把所有心率都当成 60bpm, 使 IBI 序列的
        # 时间轴与真实时间脱钩（进而使 RSA 频率、RR 序列长度都失真）。
        produced = 0
        elapsed = 0.0
        while elapsed < 1.0 and produced < 4:
            rr = self._next_ibi()
            self._t_sec += rr / 1000.0
            elapsed += rr / 1000.0
            produced += 1
            self._window.append(rr)
            self._window_t.append(self._t_sec)
        # 窗口按时间裁剪（不是按拍数）, 保证窗口只覆盖"当前状态"
        cut = self._window_t[-1] - SIM_IBI_WINDOW_SEC
        n_drop = 0
        for wt in self._window_t:
            if wt >= cut:
                break
            n_drop += 1
        if n_drop:
            del self._window[:n_drop]
            del self._window_t[:n_drop]

        feats = self.extractor.extract_from_ibi(self._window)
        # 瞬时心率: 用最近 5 拍估算（真实 MAX30102 报告近瞬时心率，
        # 而整窗均值会被之前的平静期拉低、造成显示滞后）
        recent = self._window[-5:]
        hr_now = 60000.0 / (sum(recent) / len(recent)) if recent else feats["hr_bpm"]
        # LF/HF：取仿真 RR 序列的频域计算结果与场景目标的混合。
        # 本项目的 HRVExtractor.frequency_domain 以"心跳序号"为时间轴做
        # Lomb-Scargle, 频率轴无绝对生理意义, 仅能作相对量; 故与目标值混合,
        # 使该字段的量纲与 EMOTION_RULES 一致(实测硬件则由 bridge.py 用
        # 均匀重采样的 RR 波图给出, 见 esp32_firmware/bridge.py)。
        t = self._targets
        lf_hf = 0.5 * feats.get("lf_hf", 0.0) + 0.5 * (t["lf_hf"]
                + self.rng.gauss(0, 0.3))

        return {
            "t": round(ts, 3),
            "hr": round(hr_now, 1),
            "ibi": [round(v, 1) for v in feats["ibi_list"][-3:]],
            "sdnn": round(feats["sdnn"], 2),
            "rmssd": round(feats["rmssd"], 2),
            "pnn50": round(feats["pnn50"], 2),
            "lf_hf": round(max(0.2, lf_hf), 3),
            "rr": round(t["rr"] + self.rng.gauss(0, 0.5), 1),
            "sig_quality": round(self.rng.uniform(0.85, 1.0), 3),
            "_simulation": True,
            "_scenario_emotion": t["emotion"],   # 仅模拟器内部标注，真实硬件无此字段
        }

    def run_all(self) -> Iterator[Dict]:
        """逐秒产出整个场景的数据包"""
        while True:
            pkt = self.tick()
            if pkt is None:
                return
            yield pkt
