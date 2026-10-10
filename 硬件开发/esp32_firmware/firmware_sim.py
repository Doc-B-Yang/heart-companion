# -*- coding: utf-8 -*-
"""
firmware_sim.py - 固件级仿真器（PC 端运行，不需要 ESP32 / MAX30102）

用途
----
在硬件到货前/调试期间，产生与 ESP32 串口**完全同格式**的 JSON 行（1 行/秒），
用于离线验证 link:  bridge.py → 心理助手 Agent。

与 core.simulator.SensorSimulator 的区别（重要）:
  SensorSimulator 直接"编造" HRV 特征值（绕过信号处理）；
  本仿真器**生成 PPG 原始波形**，再走与固件同一份 hrv.HRVAnalyzer 算法链，
  因此能验证的是"整条链路 + 算法"而不只是"接口格式"。二者用途不同，不可互替。

用法
----
    # 打印到屏幕（模拟串口输出）
    python firmware_sim.py --scenario stress_event --seconds 120

    # 存为文件，供反复回放
    python firmware_sim.py --scenario high_anx --seconds 60 --out demo.jsonl

    # 带运动伪迹（验证鲁棒性）
    python firmware_sim.py --scenario calm --artifact 0.05
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time

sys.path.insert(0, ".")
try:
    import hrv as hrv_mod
except ImportError:  # 允许从上层目录导入
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hrv as hrv_mod


# ----------------------------------------------------------------- 场景
# 目标: 静息心率(hr) / SDNN / RMSSD / 呼吸率(rr) / LF-HF 功率比
# 数值取自 Agent 的 EMOTION_RULES 参考表区间中点（计划书 §4.2）。
# 必须同时给 SDNN 与 RMSSD: 生理上 SDNN 主要由极低频(VLF)成分驱动,
# RMSSD 只对逐拍快变化敏感 —— 二者不是同一个量, 不能只指定 RMSSD。
#   例: 「疲劳」的 SDNN/RMSSD(40/25=1.6) 明显高于「平静」(70/55=1.27),
#       若只建模 RMSSD, 仿真数据永远无法落入疲劳的 SDNN 区间。
SCENARIOS = {
    "calm":     {"hr": 68, "sdnn": 70, "rmssd": 55, "rr": 14, "lf_hf": 1.5},
    "happy":    {"hr": 74, "sdnn": 50, "rmssd": 43, "rr": 16, "lf_hf": 2.0},
    "fatigue":  {"hr": 62, "sdnn": 40, "rmssd": 25, "rr": 12, "lf_hf": 1.2},
    "low_anx":  {"hr": 82, "sdnn": 40, "rmssd": 32, "rr": 17, "lf_hf": 2.8},
    "high_anx": {"hr": 98, "sdnn": 20, "rmssd": 16, "rr": 22, "lf_hf": 4.5},
}

# 时序剧本: (情绪, 持续秒数) —— 复刻"平静 → 压力事件 → 高焦虑 → 干预 → 恢复"
# 注: 各段时长刻意长于 HRV 统计窗(60s)。Agent 自带模拟器用的是 15/12/25/15/20s
#     的紧凑剧本, 但那样每段都短于统计窗, 窗口内始终混合两种状态, 无法验证
#     "状态切换后指标能否收敛" —— 这是本项目响应性(延迟<1s)的关键。
SEQUENCES = {
    "stress_event":   [("calm", 40), ("low_anx", 30), ("high_anx", 70),
                       ("low_anx", 30), ("calm", 40)],
    "anxiety_attack": [("calm", 40), ("high_anx", 70), ("calm", 40)],
    "afternoon_fatigue": [("calm", 40), ("fatigue", 70), ("calm", 40)],
    "happy_chat":     [("calm", 40), ("happy", 70), ("calm", 40)],
}

FS = 100            # 采样率 Hz
DC_LEVEL = 90000    # IR 直流电平（实测正常值量级）
AC_AMP = 1000       # 脉搏交流幅度（实测正常值量级 ≈1000）
NOISE_SIGMA = 22    # 传感器噪声（18-bit ADC 折算）

# 经验标定系数: 固件信号链(低通 EMA + 抛物线插值 + RR 清洗)会把低频 RR 抖动
# 轻微平滑掉, 实测输出约为目标值的 0.77 倍。此处反向补偿, 使仿真数据落到
# EMOTION_RULES 的目标区间, 从而让"仿真 → 识别"闭环可作准确率核对。
RMSSD_CALIB = 1.30
SDNN_CALIB = 1.00


def _phase_profile(ph: float) -> float:
    """单搏 PPG 波形模板（归一化幅度）。

    ph ∈ [0,1) 为当前心搏周期内的相位。形状 = 收缩期主峰 + 重搏波次峰，
    与真实 MAX30102 指腹波形一致（含 0.30~0.35s 处的次峰，用于验证不应期逻辑）。
    """
    main = math.exp(-((ph - 0.15) / 0.090) ** 2)
    dicrotic = 0.30 * math.exp(-((ph - 0.40) / 0.120) ** 2)
    return main - dicrotic


class FirmwareSim:
    """模拟 ESP32 固件：合成 PPG → hrv.HRVAnalyzer → 串口 JSON 行"""

    def __init__(self, scenario: str = "calm", seed: int = 7,
                 artifact: float = 0.0, seconds: int = 0):
        self.rng = random.Random(seed)
        self.artifact = artifact
        self.seconds = seconds

        if scenario in SEQUENCES:
            self.phases = list(SEQUENCES[scenario])
        elif scenario in SCENARIOS:
            self.phases = [(scenario, seconds or 60)]
        else:
            raise ValueError("未知场景 %s，可选: %s / %s"
                             % (scenario, list(SCENARIOS), list(SEQUENCES)))

        self.analyzer = hrv_mod.HRVAnalyzer(
            fs=FS, window_sec=60, min_peak_ms=450,
            max_peak_ms=2000, peak_ratio=0.5)

        self._phase_i = 0
        self._left = self.phases[0][1]
        self._target = dict(SCENARIOS[self.phases[0][0]])
        self._target["emotion"] = self.phases[0][0]

        # 生理状态
        self._t_ms = 0.0
        self._last_beat_ms = 0.0
        self._cur_rr = 60000.0 / self._target["hr"]
        self._next_beat_ms = 0.0
        self._drift_ph = 0.0

        self._second = 0
        self._lines = 0
        self._gen = None

    # ------------------------------------------------------- 内部
    def _advance_phase(self) -> bool:
        self._phase_i += 1
        if self._phase_i >= len(self.phases):
            return False
        name, dur = self.phases[self._phase_i]
        self._left = dur
        self._target = dict(SCENARIOS[name])
        self._target["emotion"] = name
        return True

    def _next_rr(self) -> float:
        """按目标 HR / SDNN / RMSSD / 呼吸率 / LF-HF 生成下一个 RR 间期。

        RR 序列建模（三个振荡器 + 噪声）:
            VLF 0.03Hz : 极低频, 主要贡献 SDNN（体温调节/血管舒缩节律）
            LF  0.10Hz : Mayer 波, 交感受压反射
            HF  呼吸频率: RSA, 副交感
            高斯噪声    : 传感器/检测抖动

        求解顺序:
          1) 由目标 RMSSD 与目标 LF/HF 解出 LF、HF 幅度
             （单正弦采样序列 RMSSD_A ≈ A·2πf/√2, 功率比 = A_L²/A_H²）
          2) 目标 SDNN² 减去 LF+HF+噪声已贡献的方差, 余额交给 VLF
          3) VLF 对 RMSSD 也有少量贡献, 回代修正一轮（RMSSD 预算扣除它）
        """
        t = self._target
        base = 60000.0 / t["hr"]
        f_h = t["rr"] / 60.0                          # 呼吸率 Hz → HF 中心频率
        f_l = 0.10                                    # Mayer 波 → LF 中心频率
        f_v = 0.03                                    # 极低频 → SDNN 主要来源
        ratio = t.get("lf_hf", 1.5)

        rmssd_t = t["rmssd"] * RMSSD_CALIB            # 补偿信号链平滑损失
        sdnn_t = t["sdnn"] * SDNN_CALIB
        sigma = rmssd_t * 0.25                        # 随机抖动

        w_v = 2.0 * math.pi * f_v
        w_l = 2.0 * math.pi * f_l
        w_h = 2.0 * math.pi * f_h

        a_v = 0.0
        for _ in range(2):                            # 两轮迭代即可收敛
            v_rem = max(0.0, rmssd_t ** 2 - 2 * sigma ** 2
                        - (a_v * w_v) ** 2)
            denom = ratio * f_l ** 2 + f_h ** 2
            k = math.sqrt(2.0 * v_rem / denom) if denom > 0 else 0.0
            a_h = k / (2.0 * math.pi)
            a_l = a_h * math.sqrt(ratio)
            # LF+HF+噪声 已占用的方差
            used = (a_l ** 2 + a_h ** 2) / 2.0 + sigma ** 2
            rest = sdnn_t ** 2 - used
            a_v = math.sqrt(2.0 * rest) if rest > 0 else 0.0

        tsec = self._t_ms / 1000.0
        rr = base \
            + a_v * math.sin(2 * math.pi * f_v * tsec) \
            + a_l * math.sin(2 * math.pi * f_l * tsec) \
            + a_h * math.sin(2 * math.pi * f_h * tsec) \
            + self.rng.gauss(0, sigma)

        if self.rng.random() < self.artifact:         # 运动伪迹
            rr += self.rng.choice([-1, 1]) * self.rng.uniform(120, 300)
        return max(320.0, min(1900.0, rr))

    def _sample(self) -> int:
        """产生一个 18-bit IR 样本"""
        if self._t_ms >= self._next_beat_ms:
            self._cur_rr = self._next_rr()
            self._last_beat_ms = self._t_ms
            self._next_beat_ms = self._t_ms + self._cur_rr

        ph = (self._t_ms - self._last_beat_ms) / self._cur_rr
        if ph >= 1.0:
            ph = 0.999
        pulse = AC_AMP * _phase_profile(ph)

        # 缓慢基线漂移（呼吸/体动引起的直流起伏）
        self._drift_ph += 2 * math.pi * 0.05 / FS
        drift = 0.004 * DC_LEVEL * math.sin(self._drift_ph)

        v = DC_LEVEL + drift + pulse + self.rng.gauss(0, NOISE_SIGMA)
        return int(max(0, min(262143, v)))

    # ------------------------------------------------------- 对外
    def tick(self, step_ms: int = 10):
        """推进 10ms（一个采样点），返回 None"""
        for _ in range(step_ms // 10):
            self.analyzer.add_sample(self._sample())
            self._t_ms += 10.0

    def report(self):
        """推进 1 秒并返回一行"串口 JSON"（格式与 main.py 完全一致）"""
        if self._left <= 0 and not self._advance_phase():
            return None
        self._left -= 1

        self.tick(1000)
        m = self.analyzer.metrics()
        self._second += 1
        self._lines += 1

        out = {
            "t": self._second,
            "ppg": True,
            "hr": m["hr"],
            "sdnn": m["sdnn"],
            "rmssd": m["rmssd"],
            "pnn50": m["pnn50"],
            "ibi": m["ibi"],
            "ibi_new": m["ibi_new"],
            "sig_quality": m["sig_quality"],
            "nn": m["nn"],
            "ac": m["ac"],
            "pi": m["pi"],
            "rej": m["rej"],
            "rej_tot": m["rej_tot"],
            "sat": m["sat"],
            # 仿真专有: 标注真实情绪，仅用于核对识别准确率（真机无此字段）
            "_truth": self._target["emotion"],
            "_simulation": True,
        }
        return out

    def lines(self):
        """逐秒产出 JSON 文本行"""
        while True:
            pkt = self.report()
            if pkt is None:
                return
            yield json.dumps(pkt, ensure_ascii=False)

    def next_line(self):
        """返回下一行 JSON 文本（供 bridge.py 逐步取用）；剧本播完返回 None"""
        if self._gen is None:
            self._gen = self.lines()
        return next(self._gen, None)


def main():
    ap = argparse.ArgumentParser(description="ESP32 固件仿真器（离线验证用）")
    ap.add_argument("--scenario", default="stress_event",
                    help="calm/happy/fatigue/low_anx/high_anx 或剧本名 %s"
                         % list(SEQUENCES))
    ap.add_argument("--seconds", type=int, default=60,
                    help="单情绪场景持续秒数（剧本场景忽略）")
    ap.add_argument("--artifact", type=float, default=0.0,
                    help="运动伪迹注入概率 0~1")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--realtime", action="store_true",
                    help="按真实 1 行/秒输出（默认尽快输出）")
    ap.add_argument("--out", default=None, help="写入 .jsonl 文件")
    args = ap.parse_args()

    sim = FirmwareSim(scenario=args.scenario, seed=args.seed,
                      artifact=args.artifact, seconds=args.seconds)
    fout = open(args.out, "w", encoding="utf-8") if args.out else None
    n = 0
    try:
        for line in sim.lines():
            print(line, flush=True)
            if fout:
                fout.write(line + "\n")
                fout.flush()
            n += 1
            if args.realtime:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if fout:
            fout.close()
        sys.stderr.write("\n[sim] 共输出 %d 行%s\n"
                         % (n, (" -> " + args.out) if args.out else ""))


if __name__ == "__main__":
    main()
