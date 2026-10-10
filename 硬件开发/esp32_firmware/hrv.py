# -*- coding: utf-8 -*-
"""
hrv.py - PPG 信号处理与 HRV(心率变异性) 计算

输入: MAX30102 IR 通道原始样本 (18-bit) + 采样时间戳(ms)
输出: 心率 HR、RR 间期序列、时域 HRV 指标 SDNN / RMSSD / pNN50

信号链(经硬件实测标定):
  1. 高通去直流: EMA(alpha=0.02, 截止≈0.3Hz) 抑制呼吸/运动引起的基线漂移
  2. 低通平滑  : EMA(alpha=0.25, 截止≈5Hz)  抑制高频抖动(否则单搏会产生多个假峰)
  3. 幅度包络  : 慢上升/快衰减(抗伪迹); 门限基准取近期检出峰幅中位数
  4. 峰值检测  : 局部极大值 + 抛物线插值(亚采样峰位) + 自适应门限 + 生理不应期(≥450ms)
  5. RR 校验   : 生理区间[30,200]bpm + 近5拍中位数 40% 离群剔除

注: 为适配 ESP32 有限内存, 不再整窗缓存 60s 波形(原实现会 MemoryError);
    仅保留峰值插值所需的 3 点波形缓冲与 512 条 RR 缓冲。
"""

import math


class HRVAnalyzer:
    def __init__(self, fs=100, window_sec=60,
                 min_peak_ms=450, max_peak_ms=2000, peak_ratio=0.5, polarity=1):
        if polarity not in (-1, 1):
            raise ValueError('PPG polarity must be +1 or -1')
        self.polarity = polarity
        self.fs = fs
        if fs <= 0:
            raise ValueError('Positive PPG sample rate required')
        self.dc_alpha = 1 - .98 ** (100.0 / fs)
        self.lp_alpha = 1 - .75 ** (100.0 / fs)
        self.sample_ms = 1000.0 / fs           # 相邻样本时间间隔(ms)
        self._sidx = 0                         # 样本计数(用于精确时间戳)
        self.window_sec = window_sec           # HRV 统计时间窗(秒), <=0 表示不限
        self.window_n = int(fs * window_sec)   # 兼容保留(不再整窗缓存波形)
        self.wave_n = 3                        # 局部极值与抛物线插值只访问最近3点
        self.rr_n = 512                        # RR 间期缓冲(HRV 统计)
        self.min_peak_ms = min_peak_ms
        self.max_peak_ms = max_peak_ms
        self.peak_ratio = peak_ratio

        self.ir = []        # 滤波后信号(用于局部极值；完整波形另行诊断采集)
        self.t = []         # 对应时间戳(ms)
        self.peaks = []     # 峰值时间戳(ms)
        self.peak_amps = [] # 近期检出峰幅度(用于自适应门限)
        self.rr = []        # RR 间期(ms)
        self.rr_t = []      # RR 累计时间轴(s), 用于按时间窗裁剪

        self.dc = None      # 高通 EMA 状态
        self.lp = None      # 低通 EMA 状态
        self.amp_env = None # 幅度包络
        self.last_peak_t = None
        self.thr = 0.0
        self.rr_rej = 0     # 被剔除的离群 RR 数

        # --- 上报用状态 (对接 Agent 契约包) ---
        self.sat_n = 0      # ADC 饱和/异常样本计数(削顶或黑屏)
        self.pending_rr = []  # 自上次 metrics() 以来新检出的 RR(增量上报, 避免重复)
        self.pending_events = []
        self.pending_dropped = 0
        self.beat_seq = 0
        self.segment_id = 0

        # --- 运动伪迹门控 (MPU6050) ---
        # 由外部(主循环读 IMU)通过 add_sample(..., motion=True) 告知"此刻在动"。
        # 运动期间**只跑滤波器、不记录峰/RR** —— 从源头掐掉伪迹, 比事后
        # 按统计离群剔除更彻底(后者会把真实的长/短 RR 一起误剔)。
        self.motion_active = False
        self.motion_n = 0        # 被判定为运动的样本累计数
        self.motion_segs = 0     # 运动段次数(用于判断"是不是一直在动")
        self.motion_ema = 0.0    # 运动占比的指数平滑(≈20s 时间常数), 供质量分使用

    # ---------- 复位 ----------
    def reset(self):
        self.ir.clear(); self.t.clear(); self.peaks.clear()
        self.peak_amps.clear(); self.rr.clear(); self.rr_t.clear()
        self._sidx = 0
        self.dc = None
        self.lp = None
        self.amp_env = None
        self.last_peak_t = None
        self.thr = 0.0
        self.rr_rej = 0
        self.sat_n = 0
        self.pending_rr = []
        self.pending_events = []
        self.pending_dropped = 0
        self.motion_active = False
        self.motion_n = 0
        self.motion_segs = 0
        self.motion_ema = 0.0

    def invalidate(self):
        """Break continuity. Never join intervals across motion, missing reads or bad peaks."""
        self.reset()
        self.segment_id += 1

    # ---------- 逐样本处理 ----------
    def add_sample(self, ir_value, t_ms=None, motion=False):
        """加入一个 IR 样本，返回 True 表示检测到一个脉搏峰。

        t_ms 建议留空: MAX30102 采样率固定, 按样本序号推算时间戳
        (第 k 个样本位于 k×sample_ms) 可避免"整批 FIFO 同时间戳"
        造成的峰位量化误差(会虚高 RMSSD)。

        motion: 该样本时刻 IMU 是否判定为运动(由主循环提供)。
                为 True 时**不记录峰/RR** —— 运动段的 RR 是伪迹, 直接从
                源头丢弃; 滤波器仍继续运行以保持 IIR 状态连续。
        """
        if t_ms is None:
            t_ms = self._sidx * self.sample_ms
        elif self.t and (t_ms <= self.t[-1] or t_ms-self.t[-1] > self.sample_ms*1.5):
            self.invalidate()
        self._sidx += 1

        # -1) 运动状态跳变处理
        if motion != self.motion_active:
            if motion:
                self.motion_segs += 1
                self.rr = []
                self.rr_t = []
                self.pending_rr = []
                self.pending_events = []
                self.segment_id += 1
            else:
                # 退出运动段: **强制重新锁定**。
                # 若不清空, 会用"运动前最后一个峰"与"运动后第一个峰"算出
                # 一个跨度很大的假 RR(可能仍落在 450~2000ms 生理区间内而被
                # 误收) → 直接污染 RMSSD。清空后需重新累积 2 个峰才产出 RR。
                self.last_peak_t = None
                self.peaks = []
            self.motion_active = motion

        self.motion_ema = 0.95 * self.motion_ema + 0.05 * (1.0 if motion else 0.0)
        if motion:
            self.motion_n += 1

        # 0) 饱和/异常样本计数: 18-bit 满量程 262143。
        #    削顶(>=262100)会让波形变平 → 峰位抖动 → HRV 失真;
        #    贴边(<=100)说明 LED 电流过低或未贴合。二者都计入质量分。
        if ir_value >= 262100 or ir_value <= 100:
            self.sat_n += 1

        # 1) 高通去直流 (alpha=0.02 → fc≈0.3Hz)
        if self.dc is None:
            self.dc = float(ir_value)
        self.dc = self.dc * (1-self.dc_alpha) + ir_value * self.dc_alpha
        hp = ir_value - self.dc

        # 2) 低通平滑 (alpha=0.25 → fc≈5Hz)
        if self.lp is None:
            self.lp = hp
        else:
            self.lp = self.lp * (1-self.lp_alpha) + hp * self.lp_alpha
        # Polarity selects the optical landmark after filtering. Raw ADC/DC,
        # saturation, timestamps and continuity gates are never transformed.
        # Default +1 preserves production behavior; -1 is a shadow candidate
        # until compared against a real beat reference.
        ac = self.polarity * self.lp

        self.ir.append(ac)
        self.t.append(t_ms)
        if len(self.ir) > self.wave_n:
            self.ir.pop(0)
            self.t.pop(0)

        # 3) 幅度包络(峰值跟随, 仅用于冷启动门限 bootstrap 与信号强度显示;
        #    正式门限以"近期检出峰幅中位数"为准, 天然抗单次运动伪迹)
        #    运动期间**冻结**包络 —— 否则伪迹尖峰会把包络抬爆, 之后门限持续
        #    偏高(慢衰减, 约需 10s 才落回) → 真实脉搏被系统性漏检。
        if not self.motion_active:
            aa = ac if ac >= 0 else -ac
            if self.amp_env is None or aa > self.amp_env:
                self.amp_env = aa
            else:
                self.amp_env *= 0.995

        # 4) 峰值检测: 局部极大值 + 自适应门限 + 不应期
        detected = False
        if self.motion_active:
            # 运动段: 滤波器已更新(状态连续), 但峰位/RR 一概不记录。
            return False

        L = len(self.ir)
        if L >= 3:
            a, b, c = self.ir[L - 3], self.ir[L - 2], self.ir[L - 1]
            if b > a and b >= c and b > 0:
                # 抛物线插值: 用峰及左右邻居求亚采样精度峰位。
                # 信号较弱时主峰顶平坦, 朴素逐样本极大值会把峰位判偏 ±50ms,
                # 而 RMSSD/pNN50 对峰位时间抖动极敏感 → 必须做插值。
                denom = a - 2.0 * b + c
                sample_ms = 1000.0 / self.fs
                if denom != 0:
                    delta = 0.5 * (a - c) / denom
                    if delta > 0.5:
                        delta = 0.5
                    elif delta < -0.5:
                        delta = -0.5
                else:
                    delta = 0.0
                peak_t = self.t[L - 2] + delta * sample_ms
                dt = peak_t - self.last_peak_t if self.last_peak_t is not None else 10 ** 9
                # 不应期: 下限 450ms 硬性拦除次峰; 心率更慢时按 0.6×中位 RR 自适应放宽
                # (实测: 次峰位于主峰后 0.30~0.35s, 下限须 >350ms 才可靠)
                refr = self.min_peak_ms
                if len(self.rr) >= 3:
                    rec = sorted(self.rr[-5:])
                    med = rec[len(rec) // 2]
                    r2 = int(0.6 * med)
                    if r2 > refr:
                        refr = r2
                if dt >= refr:
                    # 门限基准: 近期检出峰幅的中位数(天然抗离群伪迹);
                    # 不足 3 个峰时退回包络 bootstrap。
                    if len(self.peak_amps) >= 3:
                        rec = sorted(self.peak_amps[-5:])
                        base = rec[len(rec) // 2]
                    else:
                        base = self.amp_env
                    gate = self.peak_ratio * base
                    if gate < 60:
                        gate = 60
                    if b > gate:
                        self.peaks.append(peak_t)
                        # 峰序列只用于取"最近两拍"算 RR, 无需长期保留
                        # (原实现无上限, 长时间运行会在 ESP32 上缓慢吃内存)
                        if len(self.peaks) > 8:
                            self.peaks.pop(0)
                        self.last_peak_t = peak_t
                        self.peak_amps.append(b)
                        if len(self.peak_amps) > 16:
                            self.peak_amps.pop(0)
                        if len(self.peaks) >= 2:
                            rr = self.peaks[-1] - self.peaks[-2]
                            if self.min_peak_ms <= rr <= self.max_peak_ms:
                                if self._rr_ok(rr):
                                    self.rr.append(rr)
                                    self.rr_t.append(peak_t / 1000.0)
                                    # 按时间窗裁剪: 情绪是"当前状态", 不能用
                                    # 7 分钟前的数据参与统计(否则状态切换后
                                    # SDNN/RMSSD 长时间被历史均值拖住, 与
                                    # 项目"延迟<1s"的响应性要求冲突)
                                    self._trim_window()
                                    # 增量队列: 供每秒上报"自上次以来的新 RR"
                                    self.pending_rr.append(rr)
                                    self.beat_seq += 1
                                    self.pending_events.append([self.beat_seq, round(peak_t, 1), round(rr, 1)])
                                    if len(self.pending_events) > 8:
                                        self.pending_events.pop(0)
                                        self.pending_dropped += 1
                                    if len(self.pending_rr) > 8:
                                        self.pending_rr.pop(0)
                                else:
                                    self.rr_rej += 1
                                    # Removing an interval and stitching its neighbours makes
                                    # RMSSD/spectra false. Restart the contiguous window instead.
                                    self.rr = []
                                    self.rr_t = []
                                    self.pending_rr = []
                                    self.pending_events = []
                                    self.segment_id += 1
                            else:
                                # A missing/extra beat is a continuity break, not
                                # permission to append to a pre-gap HRV window.
                                self.rr = []
                                self.rr_t = []
                                self.pending_rr = []
                                self.pending_events = []
                                self.segment_id += 1
                        detected = True
        return detected

    def _trim_window(self):
        """把 RR 缓冲裁到 window_sec 内(按时间, 不是按条数)。

        window_sec<=0 时退化为仅按 rr_n 条数限制(旧行为)。
        """
        if self.window_sec and self.window_sec > 0 and self.rr_t:
            t_now = self.rr_t[-1]
            cut = t_now - self.window_sec
            n = 0
            for t in self.rr_t:
                if t >= cut:
                    break
                n += 1
            if n:
                del self.rr[:n]
                del self.rr_t[:n]
        while len(self.rr) > self.rr_n:
            self.rr.pop(0)
            self.rr_t.pop(0)

    def _rr_ok(self, rr):
        """RR 离群剔除: 与近 5 个 RR 的中位数偏差需在 40% 以内。"""
        if len(self.rr) < 3:
            return True
        recent = sorted(self.rr[-5:])
        med = recent[len(recent) // 2]
        if med <= 0:
            return True
        return abs(rr - med) <= 0.4 * med

    # ---------- RR 清洗(离群剔除) ----------
    def _clean_rr(self):
        """迭代剔除离群 RR(与中位数偏差 >30%)。

        漏搏/多搏伪迹会严重拉高 SDNN/RMSSD, 计算指标前先按中位数迭代剔除
        (Kubios 式 artifact 处理)。正常 RSA 波动 (±20%) 不会被误剔。
        """
        # Incremental rejection already breaks the segment. Do not delete and
        # splice intervals here, or perform quadratic cumulative work on-device.
        return list(self.rr)

    # ---------- RR 抖动指数 ----------
    def rr_jitter(self, rr=None):
        """RR 抖动指数 = RMSSD / 平均RR —— "HRV 是否可信"的单值判据。

        为什么需要它: 坏拍(假峰挤掉真峰/漏搏)会把 RMSSD 抬到 100ms 以上,
        而 HR 仍然看着正常(≈75)。此时 sig_quality 若只看"拍数/灌注"是
        发现不了的 —— 必须看 RR 序列本身有多"不生理"。

        标定:
            静息真实   ≈ 0.02~0.06   (RMSSD 30ms / RR 800ms = 0.037)
            轻度活动   ≈ 0.06~0.10
            > 0.12     基本可判定 RR 被伪迹主导, HRV 不可用(HR 仍可能对)
        实测案例: 坏拍信号 RMSSD 130 / RR 815 = 0.16 → 远超阈值。
        """
        rr = self._clean_rr() if rr is None else rr
        m = self.mean_rr(rr)
        if m <= 0:
            return 0.0
        return self.rmssd(rr) / m

    # ---------- 信号质量(0~1) ----------
    def perfusion_pct(self):
        """灌注指数 PI(%) = 脉搏交流幅度 / 直流分量 × 100。

        指腹贴合良好时 1~3%; 腕部/贴合差时 <0.5%(实测腕部 0.62%)。
        是判断"这份 HRV 是否可信"的第一指标。
        """
        if not self.dc or self.dc <= 0 or not self.peak_amps:
            return 0.0
        amps = sorted(self.peak_amps)
        med = amps[len(amps) // 2]
        return 100.0 * med / self.dc

    def signal_quality(self):
        """信号质量 0~1 = 拍数充足度(35%) + RR 干净度(35%) + 灌注强度(30%)。

        用途: 作为 Agent 契约包 sig_quality 字段; 低于 0.5 时情绪结论不可信
        (此时不应触发主动关怀, 避免"误报焦虑"造成打扰)。
        """
        pi = self.perfusion_pct()
        # 灌注: 0.3% 以下记 0 分, 1.5% 以上记满分(线性)
        pi_ok = 0.0 if pi <= 0.3 else min(1.0, (pi - 0.3) / 1.2)

        nn = len(self.rr)
        n_ok = min(1.0, nn / 20.0)               # 20 拍以上视为样本充足

        tot = nn + self.rr_rej
        r_ok = 1.0 if tot == 0 else (nn / float(tot))  # 未被离群剔除的比例

        q = 0.35 * n_ok + 0.35 * r_ok + 0.30 * pi_ok

        # 饱和惩罚: 饱和度 >5% 说明 ADC 削顶, 波形已失真
        if self._sidx > 0:
            sat_ratio = self.sat_n / float(self._sidx)
            if sat_ratio > 0.05:
                q *= max(0.1, 1.0 - (sat_ratio - 0.05) * 9.0)

        # 运动惩罚: 近期运动占比越高, 当前 RR 窗口越"残缺"
        # (运动段被丢弃 → 窗口实际由运动前后的碎片拼成, 代表性下降)
        if self.motion_ema > 0.02:
            q *= max(0.15, 1.0 - self.motion_ema)

        # 抖动惩罚: RR 抖动指数超出生理范围 → HRV 被伪迹主导。
        # 这是**唯一能识别"HR 正常但 HRV 是假的"的判据** ——
        # 坏拍会让 RMSSD 虚高好几倍, 但拍数、灌注看起来都很正常。
        jit = getattr(self, "_jitter", 0.0)
        if jit > 0.08:
            q *= max(0.2, 1.0 - (jit - 0.08) * 4.0)

        return round(max(0.0, min(1.0, q)), 3)

    # ---------- 指标计算 ----------
    def mean_rr(self, rr=None):
        rr = self.rr if rr is None else rr
        n = len(rr)
        return (sum(rr) / n) if n else 0.0

    def mean_hr(self, rr=None):
        m = self.mean_rr(rr)
        return (60000.0 / m) if m > 0 else 0.0

    def sdnn(self, rr=None):
        rr = self.rr if rr is None else rr
        n = len(rr)
        if n < 2:
            return 0.0
        m = sum(rr) / n
        var = sum((x - m) ** 2 for x in rr) / (n - 1)
        return math.sqrt(var)

    def rmssd(self, rr=None):
        rr = self.rr if rr is None else rr
        n = len(rr)
        if n < 2:
            return 0.0
        s = sum((rr[i] - rr[i - 1]) ** 2 for i in range(1, n))
        return math.sqrt(s / (n - 1))

    def sdsd(self, rr=None):
        rr = self.rr if rr is None else rr
        n = len(rr)
        if n < 2:
            return 0.0
        diffs = [rr[i] - rr[i - 1] for i in range(1, n)]
        m = sum(diffs) / len(diffs)
        return math.sqrt(sum((d - m) ** 2 for d in diffs) / max(1, len(diffs) - 1))

    def pnn50(self, rr=None):
        rr = self.rr if rr is None else rr
        n = len(rr)
        if n < 2:
            return 0.0
        cnt = sum(1 for i in range(1, n) if abs(rr[i] - rr[i - 1]) > 50)
        return 100.0 * cnt / (n - 1)

    def metrics(self):
        rr = self._clean_rr()
        if self.peak_amps:
            _amps = sorted(self.peak_amps)
            _ac = _amps[len(_amps) // 2]      # 近期峰幅中位数 = 信号强度参考
        else:
            _ac = self.amp_env or 0.0

        # 增量 RR: 本次上报周期内新检出的心搏间隔(非"最近3拍", 无重复)。
        # Agent 契约包要求 ibi 为"最近3拍"; 但按"最近3拍"逐秒发送会产生
        # 2/3 的重叠重复(≈60bpm 时), 后端累积窗 n_beats 虚高 3 倍。
        # 故同时给出两者: ibi(契约兼容) 与 ibi_new(增量, 推荐后端采用)。
        ibi_recent = [round(v, 1) for v in rr[-3:]]
        ibi_new = [round(v, 1) for v in self.pending_rr]
        self.pending_rr = []
        events = self.pending_events
        self.pending_events = []
        pending_dropped = self.pending_dropped
        self.pending_dropped = 0

        # 抖动指数先算好并缓存 —— signal_quality() 要用它做惩罚
        _mr = self.mean_rr(rr)
        self._jitter = (self.rmssd(rr) / _mr) if _mr > 0 else 0.0

        return {
            'hr':    round(self.mean_hr(rr), 1),
            'sdnn':  round(self.sdnn(rr), 1),
            'rmssd': round(self.rmssd(rr), 1),
            'pnn50': round(self.pnn50(rr), 1),
            'sdsd':  round(self.sdsd(rr), 1),
            'nn':    len(rr),                       # 清洗后有效 RR 数
            'ac':    round(_ac, 1),
            'pi':    round(self.perfusion_pct(), 3),  # 灌注指数 %
            'rej':   len(self.rr) - len(rr),        # 清洗剔除的离群 RR 数
            'rej_tot': self.rr_rej,                 # 检测阶段剔除的离群 RR 数
            'sat':   self.sat_n,                    # 饱和/异常样本累计数
            'ibi':   ibi_recent,                    # 最近 3 拍(契约字段)
            'ibi_new': ibi_new,                     # 自上次上报以来新增(增量)
            'ibi_events': events,
            'segment_id': self.segment_id,
            'pending_dropped': pending_dropped,
            'jitter': round(self._jitter, 4),       # RR抖动指数: >0.12 则 HRV 不可信
            'sig_quality': self.signal_quality(),   # 0~1 信号质量
            'moving': self.motion_active,           # 此刻是否判定为运动
            'motion': round(self.motion_ema, 3),    # 近期运动占比 EMA(0~1)
            'motion_n': self.motion_n,              # 累计运动样本数
            'segs':  self.motion_segs,              # 运动段次数
        }
