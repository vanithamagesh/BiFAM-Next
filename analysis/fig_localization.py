import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"], "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e0'
A = pd.read_csv('results/ablation/localization_ablation.csv')
for c in ['bifam_hit', 'ag_hit', 'cam_hit']: A[c] = A[c].map({True: 1, False: 0, 'True': 1, 'False': 0}).astype(float)
order = ['Full BiFAM-Next', 'DenseNet121 encoder', 'Transformer head -> pooling', 'Without channel attention', 'BiFAM -> concatenation',
         'Without pyramid pooling', 'Without attention-gated path', 'Without label smoothing', 'Without mask supervision']
lab = {'Full BiFAM-Next': 'Full BiFAM-Next', 'DenseNet121 encoder': 'DenseNet121 encoder', 'Transformer head -> pooling': 'Head → global pooling',
       'Without channel attention': 'w/o channel attention', 'BiFAM -> concatenation': 'BiFAM → concatenation', 'Without pyramid pooling': 'w/o pyramid pooling',
       'Without attention-gated path': 'w/o attention-gated path', 'Without label smoothing': 'w/o label smoothing', 'Without mask supervision': 'w/o mask supervision'}
col = lambda v: '#2a78d6' if v == 'Full BiFAM-Next' else '#eb6834' if v == 'Without mask supervision' else '#b8b6b0'
area = A.mask_frac.median()
fig, (a, b) = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.08))
data = [A[A.variant == v].bifam_in.values for v in order]
y = np.arange(len(order))[::-1]
bp = a.boxplot(data, positions=y, vert=False, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
for bx, v in zip(bp['boxes'], order): bx.set_facecolor(col(v)); bx.set_edgecolor(INK2)
for yy, d in zip(y, data): a.text(1.02, yy, f"{np.median(d) * 100:.1f}%", va='center', fontsize=10, color=INK)
a.axvline(area, color=INK2, ls='--', lw=1); a.text(area + 0.01, len(order) - 0.35, f'tumor area ({area * 100:.1f}%):\nlevel of a uniform map', fontsize=9, color=INK2, va='top')
a.set_yticks(y); a.set_yticklabels([lab[v] for v in order]); a.set_xlim(0, 1.12); a.set_xticks(np.linspace(0, 1, 6))
a.set_xlabel('Share of the BiFAM map (stride 4) inside the expert tumor mask'); a.grid(axis='x', color=GRID); a.set_axisbelow(True)
a.text(-0.32, 1.02, '(a)', transform=a.transAxes, fontsize=14, fontweight='bold')
h = 0.26
for k, (c, name, mk) in enumerate([('bifam_hit', 'BiFAM map', '#2a78d6'), ('ag_hit', 'attention gate α', '#1baf7a'), ('cam_hit', 'Grad-CAM', '#eda100')]):
    v = [A[A.variant == o][c].mean() * 100 for o in order]
    bars = b.barh(y + (1 - k) * h, v, h * 0.92, color=mk, label=name)
    for yy, vv in zip(y + (1 - k) * h, v):
        if not np.isnan(vv): b.text(vv + 1, yy, f"{vv:.0f}", va='center', fontsize=8.5, color=INK)
b.set_yticks(y); b.set_yticklabels([]); b.set_xlim(0, 108); b.set_xlabel('Slices whose map maximum lies inside the tumor (%)')
b.grid(axis='x', color=GRID); b.set_axisbelow(True); b.legend(frameon=False, loc='lower right', fontsize=10)
b.text(-0.03, 1.02, '(b)', transform=b.transAxes, fontsize=14, fontweight='bold')
fig.savefig('figures/fig_p1_localization.png', dpi=250, bbox_inches='tight', facecolor='white')
from PIL import Image; w, hh = Image.open('figures/fig_p1_localization.png').size; print(round(hh / w, 3))
