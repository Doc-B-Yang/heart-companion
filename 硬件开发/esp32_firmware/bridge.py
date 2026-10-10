# -*- coding: utf-8 -*-
"""
bridge.py - 硬件 → 心理助手 Agent 桥接器（PC 端运行）
======================================================

职责（把三样东西焊在一起）
--------------------------
    ESP32 串口 JSON (1 行/秒)
        │  ① 解析 + 契约字段组装（9 字段）
        │  ② 补两个固件算不出的特征: rr(呼吸率, RSA法) 与 lf_hf(频域)
        │  ③ 个人静息基线归一化（可选, 强烈建议）
        ▼
    HeartCompanionSystem.tick(packet)
        ▼
    情绪 / 置信度 / 风险 / 触发话术  +  干预效果自动反馈

为什么需要 ①②③（不是"多此一举"）
----------------------------------
① 契约对齐: Agent 只认 {t, hr, ibi[], sdnn, rmssd, pnn50, lf_hf, rr, sig_quality}。
   固件输出的是同名子集 + 诊断字段, 时间轴还是"开机秒数"而非 Unix 时间戳
   （若直接透传, 情绪时间线会全部落到 1970 年）。
② 固件算不出的两个量:
   - rr（呼吸率）: 原方案靠 LD2410D 雷达, 现已退役; 改为从 RR 间期序列反推
     呼吸性窦性心律不齐（RSA）, 零额外硬件。
   - lf_hf: 需要在均匀重采样的 RR 波图上做频谱分析, ESP32 上没必要做。
③ 基线归一化: Agent 的 EMOTION_RULES 用的是**人群 5 分钟标准**（平静 =
   HR 60~75 / SDNN 50~90 / RMSSD 40~70）。而本项目实测:
     - 滚动 1 分钟窗口的 SDNN 天然低于 5 分钟标准;
     - 单模态 PPG 的 RMSSD 绝对值为噪声主导（实测静息 174ms, 正常 20~50ms）。
   直接用绝对值喂规则层, 所有情绪得分都会塌缩到同一档。
   解法: 除以个人静息基线、再乘回人群参考中值 → 变成"相对自身的变化量"。
   这也是 HRV 研究里的标准做法（个体内标准化）。

用法
----
    # 1) 先采个人静息基线（3 分钟, 坐着不动、正常呼吸）
    python bridge.py --port COM6 --baseline 180

    # 2) 正常联调（自动加载上一步的基线并归一化）
    python bridge.py --port COM6 --user user001

    # 3) 无硬件离线验证（固件仿真器）
    python bridge.py --port SIM --scenario stress_event
    python bridge.py --file demo.jsonl

    # 4) 只看数据、不进 Agent
    python bridge.py --port COM6 --no-agent

    # 5) 落盘（原始包 + 情绪结论, 供答辩画图）
    python bridge.py --port COM6 --csv run.csv --jsonl run.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------ 路径
_HERE = Path(__file__).resolve().parent            # esp32_firmware/
_WORKBUDDY = _HERE.parent                          # workbuddy/
_PROJECT = _WORKBUDDY.parent                       # 大创心理助手/

DEFAULT_AGENT_DIR = _PROJECT / "心理助手Agent-1" / "hrv_companion_agent(1)"
DEFAULT_AGENT_DIR_ALT = _PROJECT / "hrv_companion_agent"
DEFAULT_BASELINE_FILE = _WORKBUDDY / "baseline" / "resting_baseline.json"
sys.path.insert(0, str(DEFAULT_AGENT_DIR))
from hrv_companion_agent.core.sensor_pipeline import SensorPipeline

# 人群参考中值（取自 Agent 的 EMOTION_RULES「平静」区间中点）
# 作用: 个人基线归一化时的"锚点", 使归一化后的数值仍落在规则表量纲上。
POP_REF = {"hr": 67.5, "sdnn": 70.0, "rmssd": 55.0, "lf_hf": 1.5}

# 生理合理区间（归一化后做钳位, 防止极端基线把数值放飞到荒唐区间）
NORM_CLAMP = {"hr": (40.0, 150.0), "sdnn": (5.0, 150.0),
              "rmssd": (3.0, 130.0), "lf_hf": (0.2, 10.0)}

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:      # pragma: no cover
    _HAS_NUMPY = False


# =================================================================== 频谱
class TachogramSpectrum:
    """RR 波图（tachogram）频谱分析 —— 同时给出 LF/HF 与 RSA 呼吸率。

    为什么必须"均匀重采样":
      RR 序列本身是"非均匀采样"的（采样点间隔就等于待测量的值）。
      直接对 beat-index 做 DFT 会把心率本身当成采样率, 频率轴无生理意义。
      标准做法: 对 RR(t) 做三次样条/线性插值 → 4Hz 均匀网格 → 频谱。

    LF 0.04~0.15Hz（交感+副交感混合） / HF 0.15~0.40Hz（副交感, 即 RSA）
    → LF/HF 反映交感-副交感平衡（规则表用它区分焦虑等级）
    → HF 峰值频率 × 60 = 呼吸率（次/分）
    """

    FS_GRID = 4.0
    MIN_SPAN_SEC = 30.0

    # LF/HF 标定系数。实测结论: **不需要标定, 取 1.0**。
    # 对拍方法: 用已知目标 LF/HF 的仿真数据(含完整固件信号链)反算
    #   calm 1.5→1.51 / happy 2.0→3.43 / fatigue 1.2→0.96 /
    #   low_anx 2.8→2.40 / high_anx 4.5→3.96
    # 五组比值的几何均值 ≈ 0.99, 即**无系统性偏移**, 残差是 ±30% 的随机散布。
    # 残差来源已定位(用两正弦最小二乘拟合 RR 序列验证):
    #   固件信号链对高频 RR 抖动(RSA, 与呼吸同频)的衰减随呼吸频率变化,
    #   呼吸越快、HF 幅度被压得越多 → LF/HF 读数偏高。
    # 影响与对策: LF/HF 是**相对指标**, 绝对值不可较真; 必须配合个人基线
    #   归一化使用(见 Baseline.normalize)。保持 1.0 可让这种不确定性如实暴露。
    LF_HF_CALIB = 1.00

    # VLF 抑制: 60s 窗内 VLF(0.03Hz) 无法可靠测量, 却会泄漏进 LF 带。
    # 用 3 级级联二阶巴特沃斯高通(fc=0.05Hz) 预先压掉, 0.1Hz 处仅衰减 <1%。
    HP_FC = 0.05
    HP_STAGES = 3

    def __init__(self, max_span_sec: float = 60.0, lfhf_calib: float = None):
        self.max_span_sec = max_span_sec
        self.lfhf_calib = self.LF_HF_CALIB if lfhf_calib is None else lfhf_calib
        self.beats: List[Tuple[float, float]] = []   # (累计时间秒, rr_ms)

    # ---------------------------------------------------------- 滤波
    def _highpass(self, x):
        """3 级级联二阶巴特沃斯高通（RBJ biquad, Q=1/√2）"""
        fs = self.FS_GRID
        w0 = 2.0 * math.pi * self.HP_FC / fs
        cw, sw = math.cos(w0), math.sin(w0)
        Q = 1.0 / math.sqrt(2.0)
        alpha = sw / (2.0 * Q)
        b0 = (1.0 + cw) / 2.0
        b1 = -(1.0 + cw)
        b2 = b0
        a0 = 1.0 + alpha
        a1 = -2.0 * cw
        a2 = 1.0 - alpha
        b0, b1, b2, a1, a2 = (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
        y = x
        for _ in range(self.HP_STAGES):
            out = np.empty_like(y)
            x1 = x2 = y1 = y2 = 0.0
            for i in range(len(y)):
                xi = float(y[i])
                yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
                x2, x1 = x1, xi
                y2, y1 = y1, yi
                out[i] = yi
            y = out
        return y

    # ---------------------------------------------------------- 数据
    def add(self, rr_ms: float):
        if not (300.0 <= rr_ms <= 2000.0):
            return
        t = self.beats[-1][0] + rr_ms / 1000.0 if self.beats else 0.0
        self.beats.append((t, rr_ms))
        horizon = t - self.max_span_sec
        while self.beats and self.beats[0][0] < horizon:
            self.beats.pop(0)

    @property
    def span(self) -> float:
        if len(self.beats) < 2:
            return 0.0
        return self.beats[-1][0] - self.beats[0][0]

    # ---------------------------------------------------------- 分析
    def analyze(self) -> Dict[str, float]:
        """返回 {lf_hf, rr_bpm, resp_conf, lf_power, hf_power, n_beats}"""
        empty = {"lf_hf": 0.0, "rr_bpm": 0.0, "resp_conf": 0.0,
                 "lf_power": 0.0, "hf_power": 0.0,
                 "n_beats": len(self.beats)}
        if len(self.beats) < 25 or self.span < self.MIN_SPAN_SEC:
            return empty

        t = np.array([b[0] for b in self.beats])
        rr = np.array([b[1] for b in self.beats])

        # 均匀重采样
        n = int(self.span * self.FS_GRID)
        if n < 64:
            return empty
        tg = np.linspace(t[0], t[-1], n)
        try:
            rr_u = np.interp(tg, t, rr)          # 线性插值（无 scipy 依赖）
        except Exception:                        # pragma: no cover
            return empty

        # 去均值 + 去线性趋势 + 高通抑制 VLF（抗基线漂移, 否则 VLF 泄漏进 LF）
        rr_u = rr_u - rr_u.mean()
        if n > 2:
            x = np.arange(n, dtype=float)
            slope = np.polyfit(x, rr_u, 1)[0]
            rr_u = rr_u - slope * x
        rr_u = self._highpass(rr_u)
        # Hann 窗（降低频谱泄漏）
        rr_u = rr_u * np.hanning(n)

        # DFT（0.01Hz 分辨率）
        freqs = np.arange(0.01, 0.51, 0.01)
        psd = np.empty_like(freqs)
        for i, f in enumerate(freqs):
            w = 2.0 * math.pi * f
            ph = w * tg
            c = float(np.dot(rr_u, np.cos(ph)))
            s = float(np.dot(rr_u, np.sin(ph)))
            psd[i] = c * c + s * s

        lf = psd[(freqs >= 0.04) & (freqs <= 0.15)]
        hf = psd[(freqs >= 0.15) & (freqs <= 0.40)]
        lf_p = float(lf.sum())
        hf_p = float(hf.sum())
        lf_hf = (lf_p / hf_p) * self.lfhf_calib if hf_p > 1e-9 else 0.0

        # 呼吸率 = HF 带峰值频率
        hf_f = freqs[(freqs >= 0.15) & (freqs <= 0.40)]
        rr_bpm = 0.0
        conf = 0.0
        if hf_p > 1e-9 and len(hf_f):
            k = int(np.argmax(hf))
            f_peak = float(hf_f[k])
            rr_bpm = f_peak * 60.0
            # 置信度 = 峰值功率 / HF 总功率（越接近 1, 呼吸节律越单一清晰）
            conf = float(hf[k] / hf_p)

        return {"lf_hf": round(lf_hf, 3), "rr_bpm": round(rr_bpm, 1),
                "resp_conf": round(conf, 3),
                "lf_power": round(lf_p, 1), "hf_power": round(hf_p, 1),
                "n_beats": len(self.beats)}


# =================================================================== 基线
class Baseline:
    """个人静息基线: 采集 / 保存 / 加载 / 归一化"""

    def __init__(self, path: Path = DEFAULT_BASELINE_FILE,
                 user_id: str = "user001"):
        self.path = Path(path)
        self.user_id = user_id
        self.data: Dict[str, float] = {}

    # ---------------------------------------------------------- 采集
    def fit(self, samples: List[Dict]) -> Dict[str, float]:
        """从静息样本中取**中位数**作为基线（中位数比均值抗残留伪迹）"""
        def med(key):
            vals = sorted(float(s[key]) for s in samples
                          if s.get(key) is not None and float(s[key]) > 0)
            if not vals:
                return 0.0
            return vals[len(vals) // 2]
        self.data = {
            "user_id": self.user_id,
            "n_samples": len(samples),
            "hr": round(med("hr"), 2),
            "sdnn": round(med("sdnn"), 2),
            "rmssd": round(med("rmssd"), 2),
            "lf_hf": round(med("lf_hf"), 3),
            "rr": round(med("rr"), 2),
            "pi": round(med("pi"), 3),
            "sig_quality": round(med("sig_quality"), 3),
            "ts": time.time(),
            "note": "静息基线(中位数); 用于个人基线归一化, 详见 硬件到Agent对接说明.md",
        }
        return self.data

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        return self.path

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            return True
        except Exception:                        # noqa: BLE001
            return False

    @property
    def ready(self) -> bool:
        return bool(self.data.get("hr")) and bool(self.data.get("rmssd"))

    # ---------------------------------------------------------- 归一化
    def normalize(self, pkt: Dict) -> Dict:
        """把绝对值改成"相对个人静息基线的变化", 再锚回人群参考量纲。

            value_norm = value_raw / baseline × population_reference

        生理含义: 焦虑/压力时 HR↑、SDNN↓、RMSSD↓、LF/HF↑, 归一化后
        这些相对变化会被放到规则表能识别的绝对量纲上。
        呼吸率(rr)**不做**归一化 —— 情绪改变呼吸频率的方向不固定,
        且规则表本身用绝对值(平静 12~16 次/分)。
        """
        if not self.ready:
            return pkt
        out = dict(pkt)
        for key in ("hr", "sdnn", "rmssd", "lf_hf"):
            base = float(self.data.get(key) or 0.0)
            raw = pkt.get(key)
            if base <= 0 or raw is None:
                continue
            val = float(raw) * POP_REF[key] / base
            lo, hi = NORM_CLAMP[key]
            out[key] = round(max(lo, min(hi, val)), 2)
        return out


# =================================================================== Agent
class AgentLink:
    """延迟导入 + 优雅降级地持有 HeartCompanionSystem"""

    def __init__(self, agent_dir: Path, user_id: str, enabled: bool = True):
        self.ok = False
        self.sys = None
        self.user_id = user_id
        self.error = ""
        self.ml_loaded = False
        self.llm_backend = "-"
        if not enabled:
            self.error = "已按 --no-agent 禁用"
            return
        try:
            root = str(agent_dir)
            if root not in sys.path:
                sys.path.insert(0, root)
            from hrv_companion_agent.core.system import HeartCompanionSystem
        except Exception as e:                   # noqa: BLE001
            self.error = "导入失败: %s" % e
            return
        try:
            self.sys = HeartCompanionSystem(user_id=user_id)
            prof = self.sys.get_profile()
            self.ml_loaded = bool(prof.get("ml_loaded"))
            self.llm_backend = prof.get("llm_backend", "-")
            self.ok = True
        except Exception as e:                   # noqa: BLE001
            self.error = "初始化失败: %s" % e

    def tick(self, packet: Dict) -> Optional[Dict]:
        if not self.ok:
            return None
        try:
            state = self.sys.ingest_sensor(packet)
            return {'should_trigger':False, 'state':state}
        except Exception as e:                   # noqa: BLE001
            self.error = "tick 异常: %s" % e
            return None

    def state(self) -> Optional[Dict]:
        return self.sys.get_state() if self.ok else None

    def feedback(self, effective=None, after_risk=None) -> Dict:
        if not self.ok:
            return {"recorded": False, "reason": "agent 未启用"}
        try:
            return self.sys.feedback(effective=effective, after_risk=after_risk)
        except Exception as e:                   # noqa: BLE001
            return {"recorded": False, "reason": str(e)}

    def seed_baseline(self, base: Dict):
        """把 PC 端采到的静息基线写入 Agent 记忆层, 供触发引擎与提示词使用"""
        if not self.ok:
            return
        try:
            self.sys.memory.update_baseline({
                "hr_bpm": base.get("hr"),
                "sdnn_ms": base.get("sdnn"),
                "rmssd_ms": base.get("rmssd"),
                "lf_hf": base.get("lf_hf"),
            })
        except Exception:                        # noqa: BLE001
            pass

    def close(self):
        if self.ok:
            try:
                self.sys.close()
            except Exception:                    # noqa: BLE001
                pass


# =================================================================== 输入源
class Source:
    """统一输入源: 串口 / 固件仿真器 / jsonl 文件 / stdin / 内存脚本"""

    def __init__(self, port: str, baud: int, scenario: str, path: Optional[str],
                 speed: float = 1.0, artifact: float = 0.0):
        self.port = port
        self.baud = baud
        self.scenario = scenario
        self.path = path
        self.speed = speed
        self.artifact = artifact
        self.ser = None
        self.fh = None
        self.sim = None
        self._f = None

    def open(self):
        p = (self.port or "").upper()
        if self.path:
            self.fh = open(self.path, "r", encoding="utf-8")
            return "文件 %s" % self.path
        if p == "SIM":
            sys.path.insert(0, str(_HERE))
            from firmware_sim import FirmwareSim
            self.sim = FirmwareSim(scenario=self.scenario, artifact=self.artifact)
            return "固件仿真器[%s]" % self.scenario
        if p in ("STDIN", "-"):
            self.fh = sys.stdin
            return "标准输入"
        try:
            import serial
        except ImportError:
            raise SystemExit("缺少 pyserial: pip install pyserial")
        self.ser = serial.Serial(self.port, self.baud, timeout=1)
        self._f = self.ser.readline
        return "%s @%d" % (self.port, self.baud)

    def readline(self) -> Optional[str]:
        """返回一行 str; 源耗尽返回 None"""
        if self.sim is not None:
            if self.speed:
                time.sleep(1.0 / self.speed)
            return self.sim.next_line()
        if self.ser is not None:
            raw = self.ser.readline()
            if not raw:
                return ""
            return raw.decode("utf-8", "replace").strip()
        if self.fh is not None:
            if self.speed:
                time.sleep(1.0 / self.speed)
            line = self.fh.readline()
            return line.strip() if line else None
        return None

    def close(self):
        if self.ser:
            self.ser.close()
        if self.fh and self.fh is not sys.stdin:
            self.fh.close()


# =================================================================== 主流程
class Bridge:
    def __init__(self, args):
        self.args = args
        self.spec = TachogramSpectrum(max_span_sec=args.spec_window,
                                      lfhf_calib=args.lfhf_calib)
        self.base = Baseline(Path(args.baseline_file), args.user)
        if self.base.load():
            self.base_loaded = True
        else:
            self.base_loaded = False
        self.agent = AgentLink(Path(args.agent_dir), args.user,
                               enabled=not args.no_agent)
        self._csv_f = None
        self._csv_w = None
        self._jl_f = None
        self._stats = {"n": 0, "ok": 0, "emotions": {}, "triggers": 0}
        self._pending = None       # 待评估的干预
        self._last_quality_warn = 0.0
        self._norm_on = bool(args.calib) and self.base.ready
        # LF/HF 的 EMA 平滑状态（见 build_packet 中的说明）
        self._lfhf_ema = None
        self._lfhf_t = None
        self.sensor_pipeline = SensorPipeline()

    # ------------------------------------------------------------ 契约包
    def build_packet(self, raw: Dict) -> Tuple[Dict, Dict]:
        """固件 JSON → (契约包, 元信息)"""
        now = time.time()
        if raw.get('schema') == 2:
            flow = self.sensor_pipeline.process(raw)
            f = flow['features']
            pkt = dict(raw)
            pkt.update({'t':now, 'ibi':list(raw.get('ibi_new', [])),
                        'hr':f.get('hr_bpm',0), 'sdnn':f.get('sdnn_ms',0),
                        'rmssd':f.get('rmssd_ms',0), 'pnn50':f.get('pnn50_pct',0),
                        'lf_hf':0.0, 'rr':0.0, 'sig_quality':raw.get('sig_quality',0),
                        'pipeline_quality':flow['quality'], 'accepted':flow['accepted'],
                        '_replay': bool(self.args.file) if hasattr(self.args, 'file') else False})
            return pkt, {'raw':raw, 'pi':raw.get('pi',0), 'nn':f.get('n_beats',0),
                         'rej_tot':raw.get('rej_tot',0), 'sat':raw.get('sat',0),
                         'resp_conf':0, 'spec_n':0, 'lf_hf_raw':0, 'truth':''}

        # ibi: 优先用增量字段(无重复), 回退到"最近3拍"
        ibi = [float(v) for v in (raw.get('ibi_new', []) if 'ibi_new' in raw else [])
               if isinstance(v, (int, float))]
        for v in ibi:
            self.spec.add(v)

        spec = self.spec.analyze()

        # LF/HF 时间平滑（EMA）。
        # 原因: LF/HF 由两个**窄带峰**的功率之比得到, 60s 窗口内峰值位置的
        #   ±1 bin(0.01Hz) 抖动就能让比值变化 ±40%（实测同一静息状态下
        #   秒级读数在 0.63~1.55 间跳）。HRV 频域指标在文献中本来就是
        #   "慢变量", 必须做时间平滑才可用。
        # 代价: 引入约 tau 秒的滞后。tau 取 20s 是"可用性 vs 响应性"的折中;
        #   HR/SDNN/RMSSD 本身已是 60s 窗均值, 无需再平滑。
        lfhf_raw = spec["lf_hf"]
        # dt 用固件的开机秒数(单调、与真实时间同尺度), 这样即使离线回放
        # (--speed 0) 平滑强度也与在线一致; 没有 t 字段时退回墙钟。
        t_src = raw.get("t")
        tick_s = float(t_src) if isinstance(t_src, (int, float)) and t_src > 0 \
            else now
        if lfhf_raw > 0:
            if self._lfhf_ema is None:
                self._lfhf_ema = lfhf_raw
            else:
                tau = max(0.001, float(getattr(self.args, "lfhf_smooth", 20.0)))
                dt = 1.0 if self._lfhf_t is None else max(0.2, tick_s - self._lfhf_t)
                a = 1.0 if tau <= 0.001 else 1.0 - math.exp(-dt / tau)
                self._lfhf_ema += a * (lfhf_raw - self._lfhf_ema)
            self._lfhf_t = tick_s
        lfhf_out = round(self._lfhf_ema, 3) if self._lfhf_ema else lfhf_raw

        pkt = {
            "t": now,                       # 必须用 Unix 时间戳(详见文件头 ①)
            "hr": float(raw.get("hr") or 0.0),
            "ibi": ibi,
            "sdnn": float(raw.get("sdnn") or 0.0),
            "rmssd": float(raw.get("rmssd") or 0.0),
            "pnn50": float(raw.get("pnn50") or 0.0),
            "lf_hf": lfhf_out,                      # 固件算不出 → 由本桥接器提供
            "rr": spec["rr_bpm"] or 0.0,            # 雷达退役 → RSA 反推
            "sig_quality": float(raw.get("sig_quality") or 0.0),
            "_simulation": raw.get('_simulation') is True,
        }
        meta = {
            "raw": raw,
            "pi": raw.get("pi", 0.0),
            "nn": raw.get("nn", 0),
            "rej_tot": raw.get("rej_tot", raw.get("rej", 0)),
            "sat": raw.get("sat", 0),
            "resp_conf": spec["resp_conf"],
            "spec_n": spec["n_beats"],
            "lf_hf_raw": lfhf_raw,
            "truth": raw.get("_truth", ""),
        }
        return pkt, meta

    # ------------------------------------------------------------ 基线模式
    def run_baseline(self, src: Source, seconds: int):
        print("\n【静息基线采集】时长 %d 秒" % seconds)
        print("要求: 坐姿安静、背靠椅背、双手垂放、自然呼吸、不要说话、不看手机。")
        print("     传感器按标准位置佩戴(指夹: 食指指腹 / 胸带: 剑突下方)。")
        print("-" * 78)
        samples = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            line = src.readline()
            if line is None:
                break
            if not line or not line.startswith("{"):
                continue
            try:
                raw = json.loads(line)
            except ValueError:
                continue
            pkt, meta = self.build_packet(raw)
            q = pkt["sig_quality"]
            left = seconds - (time.time() - t0)
            print("  剩余 %3ds | HR %5.1f | RMSSD %5.1f | SDNN %5.1f | "
                  "PI %.2f%% | Q %.2f %s" % (
                      int(left), pkt["hr"], pkt["rmssd"], pkt["sdnn"],
                      meta["pi"], q,
                      "" if q >= 0.5 else "⚠信号弱"))
            # 仅采纳高质量样本, 否则基线会被坏数据污染
            if q >= self.args.min_quality and pkt["rmssd"] > 0 \
                    and meta["nn"] >= 20 and pkt.get('pipeline_quality', {}).get('quality_ok'):
                samples.append(pkt)

        if len(samples) < 10:
            print("\n✗ 有效样本不足(%d 个)。可能原因: 未贴合/信号弱/时间太短。" % len(samples))
            print("  请检查: 1) 手指是否压稳 2) auto_gain 是否 in_range 3) 传感器是否过热")
            return 1
        base = self.base.fit(samples)
        p = self.base.save()
        print("-" * 78)
        print("✓ 基线已保存: %s" % p)
        print("  HR %.1f bpm | SDNN %.1f ms | RMSSD %.1f ms | LF/HF %.2f | "
              "呼吸 %.1f 次/分 | PI %.2f%% | Q %.2f"
              % (base["hr"], base["sdnn"], base["rmssd"], base["lf_hf"],
                 base["rr"], base["pi"], base["sig_quality"]))
        print("  有效样本 %d/%d 个" % (base["n_samples"], seconds))
        self.agent.seed_baseline(base)
        if self.agent.ok:
            print("  已同步写入 Agent 记忆层(个体基线 EMA 起点)")
        return 0

    # ------------------------------------------------------------ 主循环
    def run(self, src: Source, limit: int = 0):
        self._open_outputs()
        print("\n【实时联调】")
        print("Agent: %s | ML层: %s | LLM: %s | 基线归一化: %s"
              % ("已接入" if self.agent.ok else "未启用(%s)" % self.agent.error,
                 "已加载" if self.agent.ml_loaded else "未加载(降级为规则+模糊)",
                 self.agent.llm_backend,
                 "开启" if self._norm_on else
                 ("关闭(未加 --calib)" if self.base.ready else "关闭(无基线文件)")))
        print("-" * 78)
        n = 0
        try:
            while True:
                line = src.readline()
                if line is None:
                    break
                if not line:
                    continue
                if not line.startswith("{"):
                    print("  [dev] %s" % line)
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    print("  [raw] %s" % line)
                    continue

                pkt, meta = self.build_packet(raw)
                if pkt.get('accepted') is False:
                    continue
                feed = self.base.normalize(pkt) if self._norm_on and pkt.get('_simulation') else pkt
                n += 1
                self._stats["n"] += 1

                self._render(pkt, meta, feed, n)

                if self.agent.ok:
                    result = self.agent.tick(feed)
                    self._render_agent(result, pkt, meta)
                    if feed.get('_simulation'):
                        self._maybe_feedback(pkt)

                if self._csv_w:
                    self._write_csv(pkt, meta, feed)
                if self._jl_f:
                    self._jl_f.write(json.dumps(
                        {"packet": pkt, "feature": feed, "meta": meta,
                         "state": self.agent.state()}, ensure_ascii=False,
                        default=str) + "\n")
                    self._jl_f.flush()

                if limit and n >= limit:
                    break
        except KeyboardInterrupt:
            print("\n已中断。")
        finally:
            self._summary()
            self._close_outputs()
        return 0

    # ------------------------------------------------------------ 渲染
    def _render(self, pkt, meta, feed, n):
        if meta["nn"] == 0:
            print("[%4d] 预热中… (尚未检出稳定心搏; 请确认手指贴合/ADC未饱和)" % n)
            return
        q = pkt["sig_quality"]
        tag = "✔" if q >= self.args.min_quality else "⚠"
        print("[%4d] HR %5.1f | SDNN %5.1f RMSSD %6.1f pNN50 %5.1f (nn=%3d) | "
              "LF/HF %5.2f | 呼吸 %4.1f/min(c%.2f) | Q %.2f%s"
              % (n, pkt["hr"], pkt["sdnn"], pkt["rmssd"], pkt["pnn50"], meta["nn"],
                 pkt["lf_hf"], pkt["rr"], meta["resp_conf"], q, tag))
        if self._norm_on:
            print("      归一化→ HR %5.1f | SDNN %5.1f RMSSD %5.1f | LF/HF %5.2f"
                  "   (基线 HR %.1f / RMSSD %.1f)"
                  % (feed["hr"], feed["sdnn"], feed["rmssd"], feed["lf_hf"],
                     self.base.data.get("hr", 0), self.base.data.get("rmssd", 0)))
        if q < self.args.min_quality and time.time() - self._last_quality_warn > 30:
            self._last_quality_warn = time.time()
            print("      ⚠ 信号质量 %.2f 偏低(PI %.2f%%): 硬件情绪分类已阻止。"
                  " 检查贴合/ADC量程/手指压紧程度。" % (q, meta["pi"]))

    def _render_agent(self, result, pkt, meta):
        st = self.agent.state()
        if st:
            truth = ("  真值=%s %s" % (meta["truth"],
                     "✔" if meta["truth"] == st.get("emotion") else "✘")
                     if meta["truth"] else "")
            print("      → 情绪 %s (置信 %.2f | 风险 %.1f | 层 %s)%s"
                  % (st.get("emotion"), st.get("confidence", 0),
                     st.get("risk_score") or 0, st.get("layer"), truth))
            self._stats["emotions"][st.get("emotion")] = \
                self._stats["emotions"].get(st.get("emotion"), 0) + 1
        if result and result.get("should_trigger"):
            self._stats["triggers"] += 1
            print("      " + "─" * 66)
            print("      🔔 主动触发 [%s] 风险 %.1f | 策略: %s | %s"
                  % (result.get("trigger_level"), result.get("risk", 0),
                     (result.get("strategy") or {}).get("name", "-"),
                     result.get("reason", "")))
            msg = (result.get("message") or "").strip().replace("\n", "\n      ")
            print("      小暖: %s" % msg)
            print("      " + "─" * 66)
            self._pending = {"t": time.time(), "before": result.get("risk", 0)}

    def _maybe_feedback(self, pkt):
        if not self._pending:
            return
        if time.time() - self._pending["t"] < self.args.feedback_delay:
            return
        st = self.agent.state() or {}
        r = self.agent.feedback(after_risk=st.get("risk_score"))
        print("      ↩ 干预效果回溯: 前 %.0f → 后 %.0f | %s | 触发阈值偏移 %.1f"
              % (r.get("before_risk", 0), r.get("after_risk", 0),
                 "有效" if r.get("effective") else "无效",
                 r.get("trigger_threshold_offset", 0.0)))
        self._pending = None

    # ------------------------------------------------------------ 输出
    def _open_outputs(self):
        if self.args.csv:
            self._csv_f = open(self.args.csv, "w", newline="", encoding="utf-8-sig")
            self._csv_w = csv.writer(self._csv_f)
            self._csv_w.writerow([
                "ts", "hr", "sdnn", "rmssd", "pnn50", "lf_hf", "rr", "sig_quality",
                "pi", "nn", "rej", "sat", "resp_conf",
                "hr_n", "sdnn_n", "rmssd_n", "lf_hf_n",
                "emotion", "confidence", "risk", "layer", "truth"])
        if self.args.jsonl:
            self._jl_f = open(self.args.jsonl, "w", encoding="utf-8")

    def _write_csv(self, pkt, meta, feed):
        st = self.agent.state() or {}
        self._csv_w.writerow([
            round(pkt["t"], 3), pkt["hr"], pkt["sdnn"], pkt["rmssd"], pkt["pnn50"],
            pkt["lf_hf"], pkt["rr"], pkt["sig_quality"],
            meta["pi"], meta["nn"], meta["rej_tot"], meta["sat"], meta["resp_conf"],
            feed["hr"], feed["sdnn"], feed["rmssd"], feed["lf_hf"],
            st.get("emotion", ""), st.get("confidence", ""), st.get("risk_score", ""),
            st.get("layer", ""), meta["truth"]])
        self._csv_f.flush()

    def _close_outputs(self):
        if self._csv_f:
            self._csv_f.close()
            print("CSV 已保存: %s" % self.args.csv)
        if self._jl_f:
            self._jl_f.close()
            print("JSONL 已保存: %s" % self.args.jsonl)

    def _summary(self):
        s = self._stats
        print("-" * 78)
        print("共处理 %d 包, 触发 %d 次。" % (s["n"], s["triggers"]))
        if s["emotions"]:
            tot = sum(s["emotions"].values()) or 1
            print("情绪分布: " + " | ".join(
                "%s %.0f%%" % (k, 100.0 * v / tot)
                for k, v in sorted(s["emotions"].items(), key=lambda x: -x[1])))
        if not self.agent.ok:
            print("提示: Agent 未接入 → %s" % self.agent.error)


# =================================================================== CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="硬件 → 心理助手 Agent 桥接器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法\n----")[-1])
    ap.add_argument("--port", default="COM6", help="串口号; SIM=固件仿真器, "
                                                  "STDIN=标准输入")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument('--url', default='http://127.0.0.1:8000', help='新硬件统一后端地址')
    ap.add_argument("--file", default=None, help="改为回放 .jsonl 文件")
    ap.add_argument("--scenario", default="stress_event",
                    help="SIM 模式的场景: calm/happy/fatigue/low_anx/high_anx "
                         "/stress_event/anxiety_attack/afternoon_fatigue/happy_chat")
    ap.add_argument("--artifact", type=float, default=0.0, help="SIM 运动伪迹概率")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="回放倍速(文件/SIM); 0=最快")
    ap.add_argument("--limit", type=int, default=0, help="限包数(0=不限)")

    ap.add_argument("--user", default="user001", help="Agent 用户 ID")
    ap.add_argument("--agent-dir", default=str(DEFAULT_AGENT_DIR),
                    help="hrv_companion_agent 包的父目录")
    ap.add_argument("--no-agent", action="store_true", help="不接入 Agent, 只看数据")

    ap.add_argument("--baseline", type=int, default=0,
                    help="静息基线采集模式: 采集秒数(建议 180)")
    ap.add_argument("--baseline-file", default=str(DEFAULT_BASELINE_FILE))
    ap.add_argument("--calib", action="store_true",
                    help="启用个人基线归一化(强烈建议开启)")

    ap.add_argument("--min-quality", type=float, default=0.5,
                    help="sig_quality 阈值(低于则基线样本被剔除并告警)")
    ap.add_argument("--lfhf-calib", type=float, default=None,
                    help="LF/HF 经验标定系数(默认 %.2f, 见 TachogramSpectrum)"
                         % TachogramSpectrum.LF_HF_CALIB)
    ap.add_argument("--lfhf-smooth", type=float, default=20.0,
                    help="LF/HF 的 EMA 时间常数(秒, 0=不平滑)")
    ap.add_argument("--spec-window", type=float, default=60.0,
                    help="RR 波图频谱分析窗长(秒)。越短响应越快、越不稳; "
                         "HRV 频域分析一般需 ≥30s, 默认 60s")
    ap.add_argument("--feedback-delay", type=int, default=90,
                    help="主动触发后多少秒回溯干预效果")

    ap.add_argument("--csv", default=None, help="导出 CSV")
    ap.add_argument("--jsonl", default=None, help="导出完整 JSONL")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.port not in ('SIM', 'STDIN') and not args.file and not args.no_agent:
        if args.baseline:
            print('新硬件静息基线请在网页完成电极确认和基线保存；启动run_api.py及gateway.py。')
            return 2
        if args.csv or args.jsonl:
            print('原始采集日志请使用host_monitor.py --csv；桥接已统一为后端网关。')
            return 2
        from gateway import main as gateway_main
        gateway_main(['--port', args.port, '--baud', str(args.baud),
                      '--user', args.user, '--url', args.url])
        return 0
    agent_dir = Path(args.agent_dir)
    if not agent_dir.exists() and DEFAULT_AGENT_DIR_ALT.exists():
        agent_dir = DEFAULT_AGENT_DIR_ALT
    args.agent_dir = str(agent_dir)

    src = Source(args.port, args.baud, args.scenario, args.file,
                 speed=args.speed, artifact=args.artifact)
    print("=" * 78)
    print(" 硬件 → 心理助手 Agent 桥接器")
    print("=" * 78)
    print("数据源:", src.open())
    print("Agent 包路径:", args.agent_dir,
          "" if Path(args.agent_dir).exists() else "  ⚠ 不存在")

    b = Bridge(args)
    try:
        if args.baseline > 0:
            return b.run_baseline(src, args.baseline)
        if not args.no_agent and not b.agent.ok:
            print("⚠ Agent 未接入: %s" % b.agent.error)
        return b.run(src, limit=args.limit)
    finally:
        src.close()
        b.agent.close()


if __name__ == "__main__":
    sys.exit(main())
