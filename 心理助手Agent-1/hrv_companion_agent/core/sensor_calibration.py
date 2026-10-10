"""Per-wearing-session calibration. Voltage does not prove electrode contact.

The wearer confirms contact and rest; quality screens cannot replace those
observations. Baselines are never silently learned from emotional predictions.
"""
from .. import config
from .sensor_pipeline import RAW_FEATURE_NAMES, finite


class SensorCalibration:
    def __init__(self):
        self.reset()

    def reset(self):
        self.session = None
        self.contact_at = None
        self.baseline = None
        self.baseline_end_ms = None

    def bind(self, session):
        if session != self.session:
            self.reset()
            self.session = session

    def status(self, now_ms):
        remaining = max(0, config.GSR_SETTLE_SEC - (now_ms - self.contact_at) / 1000) if self.contact_at is not None else None
        return {'contact_confirmed': self.contact_at is not None,
                'settle_remaining_sec': round(remaining, 1) if remaining is not None else None,
                'contact_ready': remaining == 0,
                'baseline_ready': self.baseline is not None}

    def confirm_contact(self, now_ms):
        self.contact_at = now_ms
        self.baseline = self.baseline_end_ms = None

    def capture(self, flow, now_ms):
        if not self.status(now_ms)['contact_ready']:
            raise ValueError('请确认电极佩戴并稳定等待120秒')
        if not flow.get('accepted') or not flow.get('quality', {}).get('quality_ok'):
            raise ValueError('静息基线需要约60秒连续合格的PPG和GSR窗口')
        feats = flow['features']
        if any(not finite(feats.get(k)) for k in RAW_FEATURE_NAMES):
            raise ValueError('基线通道不完整')
        # The whole baseline window must follow contact settling.
        if flow['window']['start_ms'] < self.contact_at + config.GSR_SETTLE_SEC * 1000:
            raise ValueError('请继续静息采集，基线窗口仍包含电极稳定期')
        self.baseline = {k: feats[k] for k in RAW_FEATURE_NAMES}
        self.baseline_end_ms = flow['window']['end_ms']

    def transform(self, feats):
        out = dict(feats)
        if self.baseline and all(finite(feats.get(k)) for k in RAW_FEATURE_NAMES):
            for target, raw, floor in [('hr_relative', 'hr_bpm', 1.0),
                                       ('rmssd_relative', 'rmssd_ms', 5.0),
                                       ('gsr_relative', 'gsr_mean_v', 0.02)]:
                base = self.baseline[raw]
                out[target] = (feats[raw] - base) / max(abs(base), floor)
        return out
