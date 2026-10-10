"""Plot controlled GSR electrode comparisons from actual saved captures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent / 'diagnostics' / 'acceptance-2026-10-03'
names = ['gsr-electrodes-off-220350.txt', 'gsr-worn-two-minutes-221429.txt',
         'gsr-electrodes-off-after-worn-221959.txt']
data = []
for name in names:
    rows = []
    for line in (root/name).read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    if name.startswith('gsr-worn'):
        codes = [v for r in rows for v in r.get('adc_a0_codes', [])]
        duration = [r['duration_s'] for r in rows if r.get('type') == 'result'][-1]
    else:
        rows = [r for r in rows if r.get('mode') == 'continuous_128sps']
        codes = [v for r in rows for v in r.get('codes', [])]
        duration = [r['duration_s'] for r in rows if r.get('type') == 'adc_capture_result'][-1]
    volts = np.asarray(codes, dtype=float)*4.096/32768
    data.append((volts, duration))
stats = [{'file':name, 'reads':int(v.size), 'duration_s':duration,
          'mean_v':float(v.mean()), 'std_v':float(v.std()),
          'min_v':float(v.min()), 'max_v':float(v.max())}
         for name,(v,duration) in zip(names,data)]
fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
labels = ['Before wear: electrodes separated', 'Worn: two-minute contact record',
          'After removal: electrodes separated']
for ax, (volts,duration), label in zip(axes, data, labels):
    ax.plot(np.arange(volts.size)/max(1,volts.size)*duration, volts, lw=0.7)
    ax.set(title=label, xlabel='Approximate read time (s)', ylabel='ADS A0 voltage (V)', ylim=(0.5,2.0))
    ax.axhline(stats[0]['mean_v'], ls='--', color='tab:red', lw=0.8)
fig.suptitle('Actual GSR output: contact response and unresolved instability')
fig.savefig(root/'GSR-contact-comparison.png', dpi=160)
plt.close(fig)
(root/'GSR-contact-comparison.json').write_text(json.dumps({'captures':stats,
    'scope':'Electrode contact and analog stability only. Time within each capture reconstructed from read count; not calibrated conductance or emotional inference.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats, indent=2))
