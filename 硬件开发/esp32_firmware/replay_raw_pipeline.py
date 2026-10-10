"""Offline reconstruction for the unchanged backend gates, never live input.

Only complete, contiguous captures whose errors are fully accounted for by
recorded FIFO events are supported. Report sequence/session and channel status
are derived audit metadata, not recorded production packets. No labels,
baseline, deployment, reference accuracy or classification are authorized.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / '心理助手Agent-1/hrv_companion_agent(1)/hrv_companion_agent'
sys.path.insert(0, str(BACKEND))
from core.sensor_pipeline import SensorPipeline


def evaluate(source):
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    if not any(r.get('type') == 'complete' for r in rows):
        raise ValueError('Incomplete capture')
    shadow = json.loads(source.with_suffix('.polarity.json').read_text(encoding='utf-8'))
    stages = {}
    for name, pieces in shadow['stages'].items():
        raw = [r for r in rows if r.get('type') == 'raw' and r['stage'] == name]
        result = next(r for r in rows if r.get('type') == 'result' and r['stage'] == name)
        fifo = [e for r in raw for e in r['fifo_events']]
        errors = result['errors']
        if (errors != dict(ppg=len(fifo), fifo=len(fifo), imu=0, ads=0, clock=0)
                or result['fifo_recovery_errors'] or result['usb_errors']
                or [r['seq'] for r in raw] != list(range(1, len(raw)+1))):
            raise ValueError('Unaccounted errors or capture gaps; status reconstruction refused')
        if len(pieces) != 1:
            raise ValueError('This audit supports one unblocked raw piece only')
        eda = [e for r in raw for e in r['eda']]
        imu = [e for r in raw for e in r['imu']]
        ppg = [e for r in raw for e in r['ppg']]
        if not imu or any(e[4] for e in imu) or any(e[4] for e in ppg):
            raise ValueError('Motion/unknown-state capture requires a separate audit')
        candidates = {}
        for label, candidate in pieces[0].items():
            pipeline = SensorPipeline()
            output, previous = [], 0
            for seq, metric in enumerate(candidate['metrics'], 1):
                now = metric['time_ms']
                f = sum(previous < e[0] <= now for e in fifo)
                imu_recent = any(now-250 <= e[0] <= now for e in imu)
                n = sum(previous < e[1] <= now for e in ppg)
                packet = dict(schema=2, device_id='offline-raw-audit', boot_id='offline-only',
                    seq=seq, device_ms=now, segment_id=metric['segment_id'],
                    errors=dict(ppg=f, fifo=f, imu=0, ads=0, clock=0),
                    continuity=not f, ppg=bool(n), imu=imu_recent,
                    motion_known=imu_recent, moving=False, motion=0,
                    ads=True, sig_quality=metric['sig_quality'], pi=metric['pi'],
                    sample_rate=n*1000/(now-previous), expected_rate=100,
                    ibi_events=metric['ibi_events'],
                    eda_samples=[[e[0], e[2]] for e in eda if previous < e[0] <= now])
                assessed = pipeline.process(packet)
                output.append(dict(time_ms=now, **assessed))
                previous = now
            passing = [r for r in output if r['quality']['quality_ok']]
            candidates[label] = dict(
                reconstructed_backend_passing_reports=len(passing),
                passing_times_ms=[r['time_ms'] for r in passing],
                max_retained_beat_span_sec=max(r['quality']['window_span_sec'] for r in output),
                reasons=dict(Counter(reason for r in output for reason in r['quality']['reasons'])),
                reports=output)
        stages[name] = candidates
    return dict(source=str(source.resolve()), classification_allowed=False,
        accuracy_validated=False, baseline_saved=False, deployed=False,
        scope='offline reconstructed packets only; unchanged backend engineering gates',
        derived_fields=['session', 'report_sequence', 'channel_status', 'interval_error_counts',
                        'sample_rate', 'stationary_motion_fraction'],
        limitations='Nominal FIFO clock; stationary error-accounted single-piece captures only. '
                    'Passing here is neither live production acceptance nor independent physiology validation.',
        stages=stages)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    source = parser.parse_args().source
    audit = evaluate(source)
    dest = source.with_suffix('.pipeline-audit.json')
    with dest.open('x', encoding='utf-8') as out:
        json.dump(audit, out, ensure_ascii=False, indent=2, allow_nan=False)
    for stage, candidates in audit['stages'].items():
        print(stage, {k: {f: v[f] for f in ('reconstructed_backend_passing_reports',
                    'passing_times_ms', 'max_retained_beat_span_sec')} for k,v in candidates.items()})
    print(dest)
