# ==== BiFAM-Next: every remaining experiment of the manuscript, one JOB per Colab session =========================
# Needs: bifamnet and run_local imported. All jobs use the 'next' protocol: 1 seed (11), 384 px, <= 30 epochs,
# early stopping with patience 10, encoder learning-rate multiplier 0.3. Results: <runs>/next/<task>/<run name>/
import json, sys, time
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

CNX = 'convnextv2_nano.fcmae_ft_in22k_in1k'
run_local.PRESETS['next'] = dict(seeds=[11], img_size=384, slices_per_case=20, epochs=30, patience=10, batch=16, n_images=40)
bifamnet.DEFAULT_TRAIN['encoder_lr_mult'] = 0.3

# --- BiFAM-Next pieces: pyramid pooling (switchable with model cfg 'ppm') and label smoothing ---
if not hasattr(bifamnet, 'PyramidPool'):
    class PyramidPool(nn.Module):
        """Multi-scale context: average-pool the bottleneck to 1x1, 2x2, 3x3 and 6x6, project, upsample, concatenate, fuse."""
        def __init__(self, c, bins=(1, 2, 3, 6)):
            super().__init__()
            self.stages = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(b), nn.Conv2d(c, c // 4, 1, bias=False),
                                                       nn.BatchNorm2d(c // 4), nn.ReLU(inplace=True)) for b in bins])
            self.fuse = nn.Sequential(nn.Conv2d(c + len(bins) * (c // 4), c, 1, bias=False), nn.BatchNorm2d(c),
                                      nn.ReLU(inplace=True))

        def forward(self, x):
            h, w = x.shape[-2:]
            return self.fuse(torch.cat([x] + [F.interpolate(s(x), size=(h, w), mode='bilinear', align_corners=False)
                                              for s in self.stages], 1))

    _Lite = bifamnet.BiFAMLite

    class BiFAMNext(_Lite):
        def __init__(self, **cfg):
            super().__init__(**cfg)
            if self.cfg.get('ppm', 'convnext' in str(self.cfg.get('backbone', ''))):
                self.bottleneck = nn.Sequential(self.bottleneck, PyramidPool(self.cfg['bottleneck_channels']))
    bifamnet.PyramidPool, bifamnet.BiFAMLite = PyramidPool, BiFAMNext

if not hasattr(bifamnet, '_plain_loss'):
    bifamnet._plain_loss = bifamnet.loss_fn
_plain_loss = bifamnet._plain_loss


def smooth_loss(logits, y, num_outputs, eps=0.1):
    if num_outputs == 1:
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float() * (1 - eps) + 0.5 * eps)
    return F.cross_entropy(logits, y.long(), label_smoothing=eps)


def _m(**kw):
    """run_training arguments; every switch goes through the model configuration (works with every bifamnet.py)."""
    return dict(model_over={'backbone': CNX, **kw})


NEXT = _m()                                                                   # full BiFAM-Next
LITE = _m(backbone='densenet121', ppm=False)                                  # BiFAM-Lite (earlier model)
P3 = 'threeclass_patient'
# run name -> (manuscript item, task, run_training arguments, label smoothing)
RUNS = {
    'bifamlite':       ('BiFAM-Next, full model', None, NEXT, True),
    'abl_no_ppm':      ('Ablation: without pyramid pooling', P3, _m(ppm=False), True),
    'abl_no_mask':     ('Ablation: without mask supervision', P3, _m(seg_head=False), True),
    'abl_concat':      ('Ablation: BiFAM -> concatenation', P3, _m(fusion='concat'), True),
    'abl_no_ag':       ('Ablation: attention-gated pathway removed', P3, _m(use_ag=False), True),
    'abl_no_ca':       ('Ablation: channel attention removed', P3, _m(use_ca=False), True),
    'abl_pool':        ('Ablation: transformer head -> global pooling', P3, _m(head='pool'), True),
    'abl_no_ls':       ('Ablation: without label smoothing', P3, NEXT, False),
    'abl_densenet121': ('Ablation: DenseNet121 encoder instead of ConvNeXt-V2', P3, _m(backbone='densenet121', ppm=True), True),
    'lite':            ('BiFAM-Lite (DenseNet121, no PPM, no label smoothing)', None, LITE, False),
    # final image-level model: 512 px, three seeds, averaged with the 384 px seed-11 run (ensemble fixed in advance)
    'hr512_s22':       ('BiFAM-Next 512 px, seed 22 (ensemble member)', None, dict(NEXT, img_size=512, seeds=[22]), True),
    'hr512_s33':       ('BiFAM-Next 512 px, seed 33 (ensemble member)', None, dict(NEXT, img_size=512, seeds=[33]), True),
    'hr512_s44':       ('BiFAM-Next 512 px, seed 44 (ensemble member)', None, dict(NEXT, img_size=512, seeds=[44]), True),
    # round 2 (towards a higher image-level result): other encoders, longer training, members for a larger ensemble
    'dn512_s22':       ('DenseNet121 encoder, 512 px, seed 22', None, dict(_m(backbone='densenet121', ppm=True), img_size=512, seeds=[22], epochs=40), True),
    'dn512_s33':       ('DenseNet121 encoder, 512 px, seed 33', None, dict(_m(backbone='densenet121', ppm=True), img_size=512, seeds=[33], epochs=40), True),
    'cnxtiny_s11':     ('ConvNeXt-V2-Tiny encoder, 384 px, seed 11', None, dict(_m(backbone='convnextv2_tiny.fcmae_ft_in22k_in1k'), seeds=[11], epochs=40), True),
    'dn384_s44':       ('DenseNet121 encoder, 384 px, seed 44', None, dict(_m(backbone='densenet121', ppm=True), seeds=[44], epochs=40), True),
    'dn_s22':          ('DenseNet121 encoder, seed 22 (patient level)', P3, dict(_m(backbone='densenet121', ppm=True), seeds=[22]), True),
    'dn_s33':          ('DenseNet121 encoder, seed 33 (patient level)', P3, dict(_m(backbone='densenet121', ppm=True), seeds=[33]), True),
}
# JOB -> list of (task, run name); each job fits one 1-2 h Colab session on a T4
JOBS = {
    1: [(P3, 'bifamlite')],                                   # ~40 min + 3 min latency: Tables 2, 3, 4, 5 (patient level)
    2: [(P3, 'abl_no_ppm'), (P3, 'abl_no_mask')],             # ~75 min: Table 7
    3: [(P3, 'abl_concat'), (P3, 'abl_no_ag')],               # ~75 min: Table 7
    4: [(P3, 'abl_no_ca'), (P3, 'abl_pool')],                 # ~75 min: Table 7
    5: [(P3, 'abl_no_ls'), (P3, 'abl_densenet121')],          # ~75 min: Table 7
    6: [(P3, 'lite'), ('threeclass_image', 'lite')],          # ~75 min: Table 5 (BiFAM-Lite, three-class)
    7: [('fourclass_image', 'lite')],                         # ~45 min: Table 5 (BiFAM-Lite, four-class)
    8: [('threeclass_image', 'hr512_s22')],                   # ~65 min: image-level ensemble member 1
    9: [('threeclass_image', 'hr512_s33')],                   # ~65 min: image-level ensemble member 2
    10: [('threeclass_image', 'hr512_s44')],                  # ~65 min: image-level ensemble member 3
    12: [('threeclass_image', 'dn512_s22')],                  # ~70 min: DenseNet121, 512 px
    13: [('threeclass_image', 'dn512_s33')],                  # ~70 min: DenseNet121, 512 px
    14: [('threeclass_image', 'cnxtiny_s11')],                # ~60 min: larger ConvNeXt-V2 encoder
    15: [('threeclass_image', 'dn384_s44')],                  # ~45 min: DenseNet121, 384 px
    16: [(P3, 'dn_s22'), (P3, 'dn_s33')],                     # ~70 min: DenseNet121 on unseen patients, two more seeds
}


def measure_latency(out, size=384):
    """Table 2: parameters, GMAC and latency per slice of the three BiFAM networks (GPU, and CPU for BiFAM-Next)."""
    rows = []
    specs = [('BiFAM-Next', 'bifamlite', dict(backbone=CNX)), ('BiFAM-Lite', 'bifamlite', dict(backbone='densenet121', ppm=False)),
             ('BiFAM-Net', 'bifamnet', {})]
    devs = [torch.device('cuda')] if torch.cuda.is_available() else []
    for name, model, cfg in specs:
        for dev in devs + [torch.device('cpu')]:
            if dev.type == 'cpu' and name != 'BiFAM-Next':
                continue
            m = bifamnet.build_model(model, num_outputs=4, img_size=size, pretrained=False, **cfg)
            r = bifamnet.profile(m, dev, size, warmup=20 if dev.type == 'cuda' else 3, runs=200 if dev.type == 'cuda' else 10)
            r = dict(model=name, device=dev.type, **r); rows.append(r); print('[latency]', r)
            del m
    pd.DataFrame(rows).to_csv(out, index=False)


def make_attention(data_root, task, out, device=None, n=40, max_errors=12):
    """Attention maps (BiFAM and attention gate at three levels, Grad-CAM, overlay) for about n test slices spread over
    the classes plus up to max_errors misclassified ones; saved in <run>/attention/ with a contact sheet."""
    out = Path(out)
    if (out / 'attention').exists():
        return
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    df = bifamnet.load_index(data_root, bifamnet.TASKS[task][0]).set_index('path', drop=False)
    test = df.loc[bifamnet.load_split(data_root, task)['test']]
    for ck in sorted(out.glob('seed*.pt')):
        seed = int(ck.stem[4:].split('_')[0])
        model, _ = bifamnet.load_checkpoint(ck, dev)
        rows = [bifamnet._pick(test, n, seed)]
        pf = out / f'predictions_seed{seed}.csv'
        if pf.exists():
            pr = pd.read_csv(pf, keep_default_na=False)
            prob = pr[[c for c in pr.columns if c.startswith('prob_')]].values
            wrong = [p for p in pr.path[prob.argmax(1) != pr.label.values] if p in test.index][:max_errors]
            rows.append(test.loc[wrong])
        rows = pd.concat(rows).drop_duplicates('path')
        bifamnet.attention_gallery(data_root, model, rows, bifamnet.TASKS[task][1], out / 'attention', task)
        del model


def run_job(job, settings):
    P = run_local.PRESETS['next']
    data = settings['data'] + (f"_s{P['slices_per_case']}" if P['slices_per_case'] != bifamnet.SLICES_PER_CASE else '')
    runs = Path(settings['runs']) / 'next'
    todo = JOBS[job]
    # prepare the data if this machine does not have it yet (no training here)
    missing = sorted({t for t, _ in todo if t != 'kaggle4_image' and not (Path(data) / 'splits' / f'{t}.json').exists()})
    data_k4 = settings.get('data_k4', settings['data'] + '_k4')
    if any(t == 'kaggle4_image' for t, _ in todo) and not k4_prepare(data_k4, settings.get('raw', 'raw') + '/kaggle4',
                                                                      settings.get('local_k4', 'kaggle4')):
        return
    if missing:
        run_local.SETTINGS.update(settings, backbone=CNX, run_main=False, run_figures=False, run_baselines=False,
                                  run_ablation=False, run_fusion=False, run_analysis=False)
        sys.argv = ['run_local.py', '--tasks', *missing, '--preset', 'next', '--skip-doctor']
        run_local.main()
    import inspect
    _params = inspect.signature(bifamnet.run_training).parameters
    train_over = dict(epochs=P['epochs'], batch=P['batch'], workers=settings.get('workers', 2), tta=1, patience=P['patience'])
    t_start = time.time()
    for task, name in todo:
        item, _, kw, ls = RUNS[name]
        out = runs / task / name
        if (out / 'summary.json').exists():
            print(f'[job {job}] {task}/{name} already finished - skipping'); continue
        print(f'\n[job {job}] ===== {task} / {name}: {item} (label smoothing {"on" if ls else "off"}) =====')
        bifamnet.loss_fn = smooth_loss if ls else _plain_loss
        try:
            opt = {k: v for k, v in dict(keep_checkpoints=True, channels='t1ce').items() if k in _params}   # kept for the attention maps
            kw = dict(kw); size = kw.pop('img_size', P['img_size']); seeds = kw.pop('seeds', P['seeds']); ep = kw.pop('epochs', None)
            tov = dict(train_over, batch=min(train_over['batch'], 12)) if size > 400 else dict(train_over)   # 512 px fits a T4 with batch 12
            if ep:
                tov['epochs'] = ep
            bifamnet.run_training(data_k4 if task == 'kaggle4_image' else data, task, out, model='bifamlite', seeds=seeds, train_over=tov,
                                  img_size=size, device=settings.get('device'), **opt, **kw)
        finally:
            bifamnet.loss_fn = smooth_loss
        try:
            make_attention(data_k4 if task == 'kaggle4_image' else data, task, out, settings.get('device'))
        except Exception:
            import traceback; print('[attention] failed:'); traceback.print_exc()
        print(f'[job {job}] {name} done after {(time.time() - t_start) / 60:.0f} min in this session')
    if job == 1 and not (runs / 'latency.csv').exists():
        measure_latency(runs / 'latency.csv')
    bifamnet.collect_results(runs)
    print('\n================ RESULTS SO FAR (macro F1 / accuracy, %) ================')
    for s in sorted(runs.glob('*/*/summary.json')):
        m = json.load(open(s)).get('mean', {})
        if m.get('macro_f1') is not None:
            print(f"{s.parent.parent.name:20s} {s.parent.name:18s} macro F1 {100 * m['macro_f1']:6.2f}   accuracy {100 * m['accuracy']:6.2f}")
    print(f'\n[job {job}] finished. Send me the results zip (next step).')
