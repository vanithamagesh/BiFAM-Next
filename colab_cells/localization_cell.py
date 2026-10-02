# ===== Paper 1: how well the decoder attention finds the tumor, for the full model and every ablation variant. =====
# Patient-level test set (610 slices); uses only saved checkpoints, nothing is trained. GPU ~10-15 min. Outputs: MyDrive/BiFAM/paper1/
import os, sys, json, subprocess, numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
if 'PROJECT' not in globals():
    from google.colab import drive; drive.mount('/content/drive')
    PROJECT = '/content/drive/MyDrive/BiFAM'
sys.path.insert(0, PROJECT); import bifamnet
bifamnet.TIMM_NAMES.setdefault('efficientnet_b0', 'efficientnet_b0')
from torch import nn
if not hasattr(bifamnet, 'PyramidPool'):           # BiFAM-Next = BiFAMLite + pyramid pooling (same code as the jobs notebook)
    class PyramidPool(nn.Module):
        def __init__(self, c, bins=(1, 2, 3, 6)):
            super().__init__()
            self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(c, c // 4, 1, bias=False),
                                                       nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True)) for b in bins])
            self.fuse = nn.Sequential(nn.Conv2d(c + len(bins) * (c // 4), c, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        def forward(self, x):
            h, w = x.shape[-2:]
            return self.fuse(torch.cat([x] + [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False) for s in self.stages], 1))
    _Lite = bifamnet.BiFAMLite
    class BiFAMNext(_Lite):
        def __init__(self, **cfg):
            super().__init__(**cfg)
            if self.cfg.get('ppm', 'convnext' in str(self.cfg.get('backbone', ''))):
                self.bottleneck = nn.Sequential(self.bottleneck, PyramidPool(self.cfg['bottleneck_channels']))
    bifamnet.PyramidPool, bifamnet.BiFAMLite = PyramidPool, BiFAMNext
D = globals().get('DATA_DIR', '/content/data_v2_s20')
if not os.path.exists(f'{D}/splits/threeclass_image.json'):
    print('restoring prepared data from Drive ...')
    subprocess.run(['unzip', '-q', '-o', f'{PROJECT}/prepared_data_v2_s20.zip', '-d', '/content'], check=True)
OUT = f'{PROJECT}/paper1'; os.makedirs(OUT, exist_ok=True)
RUNS = f'{PROJECT}/runs/next/threeclass_patient'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device:', dev)
idx = bifamnet.load_index(D, 'threeclass').set_index('path', drop=False)
test = [p for p in json.load(open(f'{D}/splits/threeclass_patient.json'))['test'] if p in idx.index]
VARIANTS = [('bifamlite', 'Full BiFAM-Next'), ('abl_no_mask', 'Without mask supervision'), ('abl_no_ppm', 'Without pyramid pooling'),
            ('abl_concat', 'BiFAM -> concatenation'), ('abl_no_ag', 'Without attention-gated path'), ('abl_no_ca', 'Without channel attention'),
            ('abl_pool', 'Transformer head -> pooling'), ('abl_no_ls', 'Without label smoothing'), ('abl_densenet121', 'DenseNet121 encoder')]
def inputs(paths, size):
    return torch.stack([bifamnet._row_input(D, idx.loc[p], size, 1) for p in paths])
def mask(p, size):
    m = idx.loc[p, 'mask']
    return (bifamnet._png(D, m, size) > 127) if m else np.zeros((size, size), bool)
def share_in(h, m):                                    # share of the map's total activation that lies inside the tumor mask
    return float((h * m).sum() / (h.sum() + 1e-8))

def next_maps(model, paths, bs=8):
    """BiFAM-Next: Grad-CAM, BiFAM 1/4 and attention gate 1/4 (all in [0,1]) + single-view probabilities."""
    size = model.cfg['img_size']; res = []
    for s in range(0, len(paths), bs):
        x = inputs(paths[s:s + bs], size).to(dev)
        mp = bifamnet.compute_maps(model, x, level=2)
        with torch.no_grad():
            pr = bifamnet.to_prob(model(x)['logits'].float(), 3).cpu().numpy()
        for i in range(len(x)):
            res.append(dict(cam=mp['gradcam'][i].cpu().numpy(), bifam=mp['bifam'][i].cpu().numpy(),
                            ag=mp['ag'][i].cpu().numpy() if 'ag' in mp else None, prob=pr[i]))
    return res


def hit(h, m):                                         # pointing game: the map's maximum lies inside the tumor mask
    return bool(m.flat[int(np.argmax(h))]) if m.any() else np.nan

# the switch of each ablation run (same as the jobs notebook); PPM is read from the saved weights themselves
OVR = dict(abl_no_ppm=dict(ppm=False), abl_no_mask=dict(seg_head=False), abl_concat=dict(fusion='concat'), abl_no_ag=dict(use_ag=False),
           abl_no_ca=dict(use_ca=False), abl_pool=dict(head='pool'), abl_densenet121=dict(backbone='densenet121'))
def load_variant(path, run):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = {**ck['model_cfg'], **OVR.get(run, {}), 'pretrained': False}
    cfg['ppm'] = any(k.startswith('bottleneck.1.stages') for k in ck['model'])
    m = bifamnet.build_model(ck['model_name'], **cfg); m.load_state_dict(ck['model'])
    return m.to(dev).eval()

rows = []
for run, name in VARIANTS:
    d = f'{RUNS}/{run}'
    if not os.path.exists(f'{d}/seed11.pt'):
        print(f'skip {run}: no seed11.pt'); continue
    try:
        model = load_variant(f'{d}/seed11.pt', run); size = model.cfg['img_size']
    except Exception as e:
        print(f'skip {run}: could not load ({str(e)[:200]})'); continue
    pred = pd.read_csv(f'{d}/predictions_seed11.csv', keep_default_na=False).set_index('path')
    pc = [c for c in pred.columns if c.startswith('prob_')]
    for s in range(0, len(test), 64):
        chunk = test[s:s + 64]
        for p, r in zip(chunk, next_maps(model, chunk)):
            m = mask(p, size); y = int(idx.loc[p, 'label']); yh = int(pred.loc[p, pc].values.argmax())
            rows.append(dict(run=run, variant=name, path=p, group=idx.loc[p, 'group'], label=y, correct=yh == y, mask_frac=float(m.mean()),
                             bifam_in=share_in(r['bifam'], m), ag_in=share_in(r['ag'], m) if r['ag'] is not None else np.nan,
                             cam_in=share_in(r['cam'], m), bifam_hit=hit(r['bifam'], m),
                             ag_hit=hit(r['ag'], m) if r['ag'] is not None else np.nan, cam_hit=hit(r['cam'], m)))
    del model; torch.cuda.empty_cache()
    print(f'{name:32s} done')
A = pd.DataFrame(rows); A.to_csv(f'{OUT}/localization_ablation.csv', index=False)
S = A.groupby('variant', sort=False).agg(n=('path', 'size'), bifam_in=('bifam_in', 'median'), ag_in=('ag_in', 'median'), cam_in=('cam_in', 'median'),
                                       bifam_hit=('bifam_hit', 'mean'), ag_hit=('ag_hit', 'mean'), cam_hit=('cam_hit', 'mean')).round(3)
print(S.to_string()); S.to_csv(f'{OUT}/localization_summary.csv')

fig, axs = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
names = list(S.index)
for ax, k, t in [(axs[0], 'bifam_in', 'BiFAM map, stride 4'), (axs[1], 'cam_in', 'Grad-CAM, final decoder block')]:
    data = [A[A.variant == v][k].values for v in names]
    bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False, medianprops=dict(color='black'))
    for b, v in zip(bp['boxes'], names):
        b.set_facecolor('#2a78d6' if v == 'Full BiFAM-Next' else '#eb6834' if v == 'Without mask supervision' else '#cfcfcf')
    ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, rotation=35, ha='right', fontsize=9); ax.set_title(t)
    ax.set_ylim(0, 1); ax.grid(axis='y', color='#e5e5e5'); ax.set_axisbelow(True)
axs[0].set_ylabel('Share of activation inside the tumor mask')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_p1_localization.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close(fig)
print('saved to', OUT, sorted(os.listdir(OUT)))
