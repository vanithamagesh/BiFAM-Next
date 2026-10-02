# ===== Gallery of attention maps from the saved runs (like Figure 11), for Paper 1 and Paper 2. GPU ~5-10 min. =====
# Nothing is trained. Outputs: MyDrive/BiFAM/figures_gallery/
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
OUT = f'{PROJECT}/figures_gallery'; os.makedirs(OUT, exist_ok=True)
RUNS = f'{PROJECT}/runs/next'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device:', dev)
IDX = {'fourclass': bifamnet.load_index(D, 'fourclass').set_index('path', drop=False),
       'threeclass': bifamnet.load_index(D, 'threeclass').set_index('path', drop=False)}
NAMES = {3: ['glioma', 'meningioma', 'pituitary'], 4: ['glioma', 'meningioma', 'pituitary', 'no tumor']}
OVR = dict(abl_no_ppm=dict(ppm=False), abl_no_mask=dict(seg_head=False), abl_concat=dict(fusion='concat'), abl_no_ag=dict(use_ag=False),
           abl_no_ca=dict(use_ca=False), abl_pool=dict(head='pool'), abl_densenet121=dict(backbone='densenet121'))
def load_variant(path, run):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = {**ck['model_cfg'], **OVR.get(run, {}), 'pretrained': False}
    cfg['ppm'] = any(k.startswith('bottleneck.1.stages') for k in ck['model'])
    m = bifamnet.build_model(ck['model_name'], **cfg); m.load_state_dict(ck['model'])
    return m.to(dev).eval()


def get(task, run='bifamlite'):
    """model, reported predictions (flip-averaged, as in the paper) and the index of a run; None if missing."""
    d = f'{RUNS}/{task}/{run}'
    if not os.path.exists(f'{d}/seed11.pt'): print(f'skip {task}/{run}: no seed11.pt'); return None
    m = load_variant(f'{d}/seed11.pt', run)
    p = pd.read_csv(f'{d}/predictions_seed11.csv', keep_default_na=False).set_index('path')
    pc = sorted(c for c in p.columns if c.startswith('prob_'))
    p['pred'] = p[pc].values.argmax(1); p['p'] = p[pc].values.max(1); p['ok'] = p.pred == p.label
    return m, p, IDX['fourclass' if task.startswith('four') else 'threeclass']

def maps(model, idx, paths, bs=8):
    """unthresholded BiFAM (stride 4) and Grad-CAM maps in [0, 1] at the model input size"""
    size = model.cfg['img_size']; out = []
    for s in range(0, len(paths), bs):
        x = torch.stack([bifamnet._row_input(D, idx.loc[q], size, 1) for q in paths[s:s + bs]]).to(dev)
        mp = bifamnet.compute_maps(model, x, level=2)
        out += [dict(bifam=mp['bifam'][i].cpu().numpy(), cam=mp['gradcam'][i].cpu().numpy()) for i in range(len(x))]
    return out

def slice_img(idx, p, size):
    return bifamnet._png(D, idx.loc[p, 'path'], size) / 255.0

def gallery(fname, idx, paths, titles, rows, size, title_colors=None):
    """rows: list of (row label, list of maps or None for the MRI row)"""
    n = len(paths); fig, axs = plt.subplots(len(rows), n, figsize=(2.05 * n + 0.9, 2.05 * len(rows) + 0.5), squeeze=False)
    for j, p in enumerate(paths):
        im = slice_img(idx, p, size)
        for i, (lab, mm) in enumerate(rows):
            ax = axs[i, j]; ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(im, cmap='gray')
            if mm is not None: hm = ax.imshow(mm[j], cmap='jet', alpha=0.45, vmin=0, vmax=1)
            if i == 0: ax.set_title(titles[j], fontsize=9.5, color=(title_colors or ['black'] * n)[j])
            if j == 0: ax.set_ylabel(lab, fontsize=9.5)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(0, 1)), cax=cax); cb.set_label('attention weight', fontsize=9.5)
    fig.subplots_adjust(left=0.05, right=0.9, wspace=0.04, hspace=0.12)
    fig.savefig(f'{OUT}/{fname}', dpi=250, bbox_inches='tight', facecolor='white'); plt.close(fig); print('saved', fname)

def pick(p, idx, k_per_class, ok=True, seed=0, conf=0.9):
    """k correct (or wrong) test slices per class, each from a different patient where possible"""
    q = p[p.ok == ok].copy(); q['group'] = [idx.loc[x, 'group'] if x in idx.index else x for x in q.index]
    if ok: q = q[q.p >= conf]
    out = []
    for c in sorted(q.label.unique()):
        r = q[q.label == c].sample(frac=1, random_state=seed).drop_duplicates('group')
        out += list(r.index[:k_per_class])
    return out

lbl = lambda p, K, i: f"{NAMES[K][int(p.loc[i, 'label'])]}\np = {100 * p.loc[i, 'p']:.1f}%"

# 1) four-class benchmark: two correctly classified test slices per class
g = get('fourclass_image')
if g:
    m, p, idx = g; K = len(NAMES[4]) if p.label.max() == 3 else 3; ps = pick(p, idx, 2); mp = maps(m, idx, ps)
    gallery('gallery_fourclass.png', idx, ps, [lbl(p, K, i) for i in ps],
            [('MRI slice', None), ('BiFAM attention', [x['bifam'] for x in mp]), ('Grad-CAM', [x['cam'] for x in mp])], m.cfg['img_size'])
    del m; torch.cuda.empty_cache()
# 2) three-class, new patients (patient-level split): two correct slices per class, from different patients
g = get('threeclass_patient')
if g:
    m, p, idx = g; ps = pick(p, idx, 2); mp = maps(m, idx, ps); size = m.cfg['img_size']
    gallery('gallery_patient_level.png', idx, ps, [lbl(p, 3, i) for i in ps],
            [('MRI slice', None), ('BiFAM attention', [x['bifam'] for x in mp]), ('Grad-CAM', [x['cam'] for x in mp])], size)
    # 3) the most confident misclassified slices of the patient-level split, one per patient
    w = p[~p.ok].copy(); w['group'] = [idx.loc[x, 'group'] for x in w.index]
    w = w.sort_values('p', ascending=False).drop_duplicates('group').head(6); ws = list(w.index); wm = maps(m, idx, ws)
    gallery('gallery_patient_level_errors.png', idx, ws,
            [f"{NAMES[3][int(p.loc[i, 'label'])]} → {NAMES[3][int(p.loc[i, 'pred'])]}\np = {100 * p.loc[i, 'p']:.1f}%" for i in ws],
            [('MRI slice', None), ('BiFAM attention', [x['bifam'] for x in wm]), ('Grad-CAM', [x['cam'] for x in wm])], size, ['#c00000'] * len(ws))
    del m; torch.cuda.empty_cache()
    # 4) the same slices with and without mask supervision
    g2 = get('threeclass_patient', 'abl_no_mask')
    if g2:
        m2, p2, _ = g2; m1 = get('threeclass_patient')[0]
        mp1 = maps(m1, idx, ps); mp2 = maps(m2, idx, ps)
        gallery('gallery_mask_supervision.png', idx, ps, [lbl(p, 3, i) for i in ps],
                [('MRI slice', None), ('with mask\nsupervision', [x['bifam'] for x in mp1]), ('without mask\nsupervision', [x['bifam'] for x in mp2])], size)
        del m1, m2; torch.cuda.empty_cache()
# 5) three-class image-level benchmark: two correct slices per class
g = get('threeclass_image')
if g:
    m, p, idx = g; ps = pick(p, idx, 2); mp = maps(m, idx, ps)
    gallery('gallery_threeclass_image.png', idx, ps, [lbl(p, 3, i) for i in ps],
            [('MRI slice', None), ('BiFAM attention', [x['bifam'] for x in mp]), ('Grad-CAM', [x['cam'] for x in mp])], m.cfg['img_size'])
print('saved to', OUT, sorted(os.listdir(OUT)))
