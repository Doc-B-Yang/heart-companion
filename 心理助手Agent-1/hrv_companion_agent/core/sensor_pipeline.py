"""One sensor contract for USB, BLE and Web. Fail closed; no fabricated HRV.

IBI events are [unique beat id, segment-clock end ms, interval ms]. Equal
interval values may be real distinct beats: deduplicate identity, never values.
Intervals are not spliced across read errors, missing packets or rejected peaks.
GSR remains voltage features until the analogue module is physically calibrated.
"""
from __future__ import annotations
import math
from collections import deque
from statistics import mean, stdev

RAW_FEATURE_NAMES = ['hr_bpm', 'sdnn_ms', 'rmssd_ms', 'pnn50_pct',
                 'gsr_mean_v', 'gsr_std_v', 'gsr_slope_v_s',
                 'gsr_step_rms_v', 'gsr_range_v', 'motion_fraction']
FEATURE_NAMES = RAW_FEATURE_NAMES + ['hr_relative', 'rmssd_relative', 'gsr_relative']
FEATURE_SCHEMA = 'ppg-gsr-imu-baseline-voltage-v2'


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def packet_channel_diagnostics(packet):
    """Current-packet checks only; never features or a qualified window."""
    errors = packet.get('errors')
    errors = errors if isinstance(errors, dict) else {}
    now = packet.get('device_ms')
    eda_reasons, values, stamps = [], [], []
    if packet.get('ads') is not True or not finite(errors.get('ads')) or errors['ads'] != 0:
        eda_reasons.append('ADS离线、错误或状态未知')
    samples = packet.get('eda_samples')
    if not isinstance(samples, list) or not 1 <= len(samples) <= 24:
        eda_reasons.append('GSR样本不足或格式错误')
    else:
        for sample in samples:
            if (not isinstance(sample, (list, tuple)) or len(sample) != 2
                    or not all(finite(x) for x in sample)):
                eda_reasons.append('GSR样本格式错误')
                continue
            t, v = sample
            values.append(v)
            if not finite(now) or not max(0, now - 2500) <= t <= now:
                eda_reasons.append('GSR样本时间无效')
            if not .02 <= v <= 3.28:
                eda_reasons.append('GSR异常电压或削顶')
            if stamps and not 20 <= t - stamps[-1] <= 150:
                eda_reasons.append('GSR采样不连续')
            stamps.append(t)
    if stamps and finite(now) and now - stamps[-1] > 150:
        eda_reasons.append('GSR末样本已过期')
    bounded = len(values)>1 and all(.02 <= v <= 3.28 for v in values)
    step = math.sqrt(mean([(b-a)**2 for a,b in zip(values, values[1:])])) if bounded else None
    spread = stdev(values) if bounded else None
    if (step is not None and step > .03) or (spread is not None and spread > .15):
        eda_reasons.append('GSR快速波动超工程筛查门槛')
    imu_reasons = []
    if (packet.get('imu') is not True or packet.get('motion_known') is not True
            or not finite(errors.get('imu')) or errors['imu'] != 0):
        imu_reasons.append('运动状态未知或IMU读取错误')
    if packet.get('moving') is not False:
        imu_reasons.append('检测到运动或运动标志未知')
    if not finite(packet.get('motion')) or not 0 <= packet['motion'] <= 1:
        imu_reasons.append('运动比例无效或缺失')
    return {
        'scope': 'current_packet_only', 'qualified_window': False,
        'gsr': {'packet_ok': not eda_reasons, 'reasons': list(dict.fromkeys(eda_reasons)),
                'samples': len(values), 'min_v': min(values) if values else None,
                'max_v': max(values) if values else None, 'step_rms_v': step},
        'imu': {'packet_ok': not imu_reasons, 'reasons': imu_reasons}}


class SensorPipeline:
    def __init__(self, window_sec=60, min_span_sec=55, min_beats=30):
        self.window_ms = window_sec * 1000
        self.min_span_sec, self.min_beats = min_span_sec, min_beats
        self.key = None
        self.retired = deque(maxlen=16)
        self.seq = self.device_ms = self.segment = None
        self.beats = deque(maxlen=512)
        self.eda = deque(maxlen=1500)
        self.motion = deque(maxlen=120)
        self.last_beat_id = None
        self.last_eda_ms = None
        self.break_ms = 0
        self.duplicates = self.gaps = 0
        self.channel_diagnostics = {}

    def clear(self, clock_ms=0):
        self.beats.clear()
        self.eda.clear()
        self.motion.clear()
        self.last_beat_id = self.last_eda_ms = None
        self.break_ms = clock_ms

    def process(self, packet):
        self.channel_diagnostics = {}
        if not isinstance(packet, dict) or packet.get('schema') != 2:
            self.clear()
            return self._result(False, ['需要schema=2固件，旧包只可用于演示'], {}, False)
        key = (packet.get('device_id'), packet.get('boot_id'))
        seq, now, segment = packet.get('seq'), packet.get('device_ms'), packet.get('segment_id')
        if (not all(isinstance(x, str) and 1 <= len(x) <= 64 for x in key)
                or not isinstance(seq, int) or isinstance(seq, bool) or seq < 1
                or not finite(now) or now < 0 or not isinstance(segment, int)
                or isinstance(segment, bool) or segment < 0):
            self.clear()
            return self._result(False, ['缺少或无效的会话/序号/时间/片段'], {}, False)
        if key != self.key:
            if key in self.retired:
                return self._result(False, ['已结束会话的迟到包'], {}, False, ignored=True)
            if self.key:
                self.retired.append(self.key)
            self.key, self.seq, self.device_ms, self.segment = key, None, None, None
            self.clear()
        if self.seq is not None and (seq <= self.seq or now <= self.device_ms):
            self.duplicates += 1
            return self._result(False, ['重复或乱序包'], {}, False, ignored=True)
        reasons = []
        previous_ms = self.device_ms
        if self.seq is not None and (seq != self.seq + 1 or now - self.device_ms > 2500):
            self.gaps += 1
            reasons.append('缺包或采集暂停，重新积累连续窗口')
            self.clear(now)
        elif self.segment is not None and segment != self.segment:
            self.clear(previous_ms or 0)
        self.seq, self.device_ms, self.segment = seq, now, segment
        self.channel_diagnostics = packet_channel_diagnostics(packet)
        errors = packet.get('errors')
        if not isinstance(errors, dict) or any(not finite(errors.get(k)) or errors[k] < 0
                                               for k in ('ppg', 'imu', 'ads', 'fifo', 'clock')):
            reasons.append('错误计数缺失')
        elif any(errors[k] for k in ('ppg', 'imu', 'fifo', 'clock')):
            reasons.append('PPG/IMU读取或时间连续性错误')
        if packet.get('continuity') is not True or packet.get('ppg') is not True:
            reasons.append('PPG片段不连续或未在线')
        if packet.get('imu') is not True or packet.get('motion_known') is not True:
            reasons.append('运动状态未知')
        if packet.get('moving') is not False:
            reasons.append('运动期间不进行生理情绪分类')
        motion = packet.get('motion')
        if not finite(motion) or not 0 <= motion <= 1:
            reasons.append('运动比例无效或缺失')
        q = packet.get('sig_quality')
        if not finite(q) or not 0.5 <= q <= 1:
            reasons.append('PPG质量不足或未知')
        pi = packet.get('pi')
        if not finite(pi) or pi < .3:
            reasons.append('PPG灌注幅度不足或未知')
        rate, expected = packet.get('sample_rate'), packet.get('expected_rate')
        if not finite(rate) or not finite(expected) or expected <= 0 or abs(rate / expected - 1) > 0.10:
            reasons.append('采样速率异常或未知')
        self.channel_diagnostics['ppg'] = {'packet_ok': not reasons, 'reasons': list(reasons)}
        if reasons:
            self.clear(now)
            return self._result(True, reasons, {}, False)
        events = packet.get('ibi_events')
        if not isinstance(events, list) or len(events) > 8:
            self.clear(now)
            return self._result(True, ['间期事件格式错误或缺失'], {}, False)
        for event in events:
            if (not isinstance(event, (list, tuple)) or len(event) != 3
                    or not all(finite(x) for x in event)):
                self.clear(now)
                return self._result(True, ['无效间期事件'], {}, False)
            beat_id, end, rr = event
            if int(beat_id) != beat_id or beat_id < 1 or not 450 <= rr <= 2000 or end > now + 30:
                self.clear(now)
                return self._result(True, ['间期身份、范围或时间无效'], {}, False)
            if not self.beats and end - rr < self.break_ms - 30:
                # Discard a crossing interval without moving the break forward.
                continue
            if (int(beat_id) != beat_id or beat_id < 1 or not 450 <= rr <= 2000
                    or end > now + 30 or end - rr < self.break_ms - 30
                    or (self.last_beat_id is not None and beat_id != self.last_beat_id + 1)):
                self.clear(now)
                return self._result(True, ['间期身份、范围或时间不连续'], {}, False)
            if self.beats and (end <= self.beats[-1][0]
                               or abs(end - self.beats[-1][0] - rr) > 30):
                self.clear(now)
                return self._result(True, ['间期跨越缺失时间，禁止拼接'], {}, False)
            self.beats.append((end, float(rr)))
            self.last_beat_id = int(beat_id)
        while self.beats and self.beats[0][0] < now - self.window_ms:
            self.beats.popleft()
        self.motion.append((now, float(motion)))
        while self.motion and self.motion[0][0] < now - self.window_ms:
            self.motion.popleft()
        eda_reasons = []
        if packet.get('ads') is not True or errors['ads']:
            eda_reasons.append('ADS离线或读取错误')
        eda_samples = packet.get('eda_samples')
        if not isinstance(eda_samples, list) or not 1 <= len(eda_samples) <= 24:
            eda_reasons.append('GSR样本不足')
        else:
            for sample in eda_samples:
                if (not isinstance(sample, (list, tuple)) or len(sample) != 2
                        or not all(finite(x) for x in sample)):
                    eda_reasons.append('GSR样本格式错误')
                    break
                t, v = sample
                if (t > now or t < max(0, now - 2500) or not 0.02 <= v <= 3.28
                        or (self.last_eda_ms is not None and not 20 <= t - self.last_eda_ms <= 150)):
                    eda_reasons.append('GSR削顶、异常电压或采样不连续')
                    break
                self.eda.append((t, float(v)))
                self.last_eda_ms = t
        if eda_reasons:
            self.eda.clear()
            self.last_eda_ms = None
        elif self.eda and now - self.eda[-1][0] > 150:
            eda_reasons.append('GSR末样本已过期')
            self.eda.clear()
            self.last_eda_ms = None
        while self.eda and self.eda[0][0] < now - self.window_ms:
            self.eda.popleft()
        rr = [v for _, v in self.beats]
        feats = {}
        span = (self.beats[-1][0] - self.beats[0][0]) / 1000 if len(rr) > 1 else 0
        if rr:
            mr = mean(rr)
            diffs = [rr[i] - rr[i-1] for i in range(1, len(rr))]
            rmssd = math.sqrt(mean([d*d for d in diffs])) if diffs else 0
            feats.update(hr_bpm=60000 / mr, sdnn_ms=stdev(rr) if len(rr)>1 else 0,
                         rmssd_ms=rmssd, pnn50_pct=100 * mean([abs(d)>50 for d in diffs]) if diffs else 0,
                         jitter=rmssd/mr, n_beats=len(rr), window_span_sec=span,
                         motion_fraction=mean([v for _, v in self.motion]))
            if feats['jitter'] > 0.12:
                reasons.append('逐拍抖动超工程门槛，需检查波形/参考信号')
        if len(rr) < self.min_beats or span < self.min_span_sec:
            reasons.append('连续心搏窗口积累中')
        elif now - self.beats[-1][0] > 2500:
            reasons.append('没有新心搏，旧窗口已失效')
        eda_span = (self.eda[-1][0] - self.eda[0][0]) / 1000 if len(self.eda)>1 else 0
        if len(self.eda) < int(self.min_span_sec * 20 * 0.90) or eda_span < self.min_span_sec:
            eda_reasons.append('GSR连续窗口积累中')
        if self.eda:
            values = [v for _, v in self.eda]
            t = [(t-self.eda[0][0])/1000 for t, _ in self.eda]
            mv, mt = mean(values), mean(t)
            variance_t = sum((x-mt)**2 for x in t)
            slope = sum((x-mt)*(v-mv) for x,v in zip(t,values))/variance_t if variance_t else 0
            step = math.sqrt(mean([(values[i]-values[i-1])**2 for i in range(1,len(values))])) if len(values)>1 else 0
            feats.update(gsr_mean_v=mv, gsr_std_v=stdev(values) if len(values)>1 else 0,
                         gsr_slope_v_s=slope, gsr_step_rms_v=step,
                         gsr_range_v=max(values)-min(values))
            # Voltage engineering screens, not calibrated physiological thresholds.
            if step > 0.03 or feats['gsr_std_v'] > 0.15:
                eda_reasons.append('GSR快速波动超工程筛查门槛')
        all_reasons = reasons + eda_reasons
        return self._result(True, all_reasons, feats, not all_reasons,
                            ppg_ok=not reasons, eda_ok=not eda_reasons)

    def _result(self, accepted, reasons, feats, quality_ok, ppg_ok=False, eda_ok=False, ignored=False):
        start = max(self.beats[0][0] - self.beats[0][1] if self.beats else 0,
                    self.eda[0][0] if self.eda else 0)
        end = min(self.beats[-1][0] if self.beats else 0, self.eda[-1][0] if self.eda else 0)
        return {'accepted': accepted, 'ignored': ignored, 'features': feats,
                'window': {'start_ms': start, 'end_ms': end},
                'quality': {'quality_ok': quality_ok, 'ppg_ok': ppg_ok, 'eda_ok': eda_ok,
                            'channels': self.channel_diagnostics,
                            'quality_level': '合格' if quality_ok else '不可用',
                            'reason': '；'.join(reasons), 'reasons': reasons,
                            'jitter': feats.get('jitter', 0),
                            'n_beats': feats.get('n_beats', 0),
                            'window_span_sec': feats.get('window_span_sec', 0)},
                'session': {'device_id': self.key[0], 'boot_id': self.key[1],
                            'seq': self.seq, 'segment_id': self.segment} if self.key else {},
                'duplicates': self.duplicates, 'gaps': self.gaps}
