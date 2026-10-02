"""BiFAM-Net - complete single-file implementation for a local GPU workstation (Linux or Windows).

"Channel Aligned Multiplicative Feature Fusion in a Dense U-Net with Vision Transformer for Brain MRI
Tumor Classification" (Vanitha et al.)

Contents: DenseNet201 U-Net encoder, channel attention + attention-gated dual skips, BiFAM (Algorithm 1),
six-layer ViT head on 16x16 patches of the restored-resolution decoder output (Algorithm 2); the eight fusion
operators; thirteen baselines; figshare / Br35H (MD5 + pHash de-duplication) / BraTS 2015 preparation;
image-level, patient-level and case-level (5-fold CV) partitions; training with inner-validation selection;
per-class metrics, Wilson and bootstrap intervals, paired bootstrap, exact McNemar, seed t-test; localization
(BiFAM / attention gate / Grad-CAM vs. whole-tumor masks); t-SNE; source probe; cost profiling; figures.

Easiest: edit the settings at the top of run_local.py and run `python run_local.py`.
Or use the commands directly (python bifamnet.py <command> --help for all options):

  python bifamnet.py doctor                                   # GPU / CUDA / packages / disk / batch-size check
  python bifamnet.py selftest --out selftest --seeds 11       # whole pipeline on tiny synthetic data
  python bifamnet.py prepare --raw data/raw --data data/processed --brats D:/BRATS2015_Training
  python bifamnet.py train --task brats_cv --out runs/brats_cv/bifamnet            # 4 tasks: brats_cv,
  python bifamnet.py train --task threeclass_patient --out runs/threeclass_patient/bifamnet   # threeclass_patient,
  python bifamnet.py train --task fourclass_image --out runs/fourclass_image/bifamnet         # threeclass_image,
  python bifamnet.py all --runs runs                          # every experiment        # fourclass_image
  python bifamnet.py visualize --runs runs --raw data/raw --n 200
  other commands: compare, tune, localize, features, profile

Datasets: figshare downloads automatically; Br35H through kagglehub (needs a Kaggle API token in
~/.kaggle/kaggle.json, or unzip it yourself into data/raw/br35h); BraTS 2015 needs registration
(https://www.smir.ch/BRATS/Start2015) - point --brats at the folder that contains HGG/ and LGG/.

Runs are resumable: `all` skips every run whose summary.json exists. Only the main BiFAM-Net checkpoints
are kept (~390 MB each); use --keep-all-checkpoints to keep all.

Settings the manuscript defers to Supplementary Table S1 (optimizer, schedule, batch, seeds, augmentation
ranges) are placeholders in SEEDS / DEFAULT_TRAIN below; replace them with the supplementary values.
The decoder/transformer widths are not given either; the default (dec_channels 256/128/64, dim 768) has
95.97 M parameters (manuscript: 96.8 M). The hybrid baselines are compact re-implementations of the cited
design families, not the original authors' code. Run `doctor` to find the batch size that fits your GPU.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def _ensure(*pkgs):
    """pip-install missing packages (Colab lacks timm, SimpleITK, h5py or kagglehub on some images)."""
    import importlib
    missing = [p for p in pkgs if importlib.util.find_spec(p.split("==")[0].replace("-", "_")) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure("timm", "SimpleITK", "h5py")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import cohen_kappa_score, confusion_matrix, matthews_corrcoef, roc_auc_score  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


# =====================================================================================================
# Fusion operators
# Skip-feature refinement and the eight fusion operators compared in Section 3.6.
#
# Every operator receives the channel-attended map F_CA and the attention-gated map F_A of the same
# skip feature S (both with C_s channels) and returns a map with C_T = C_s channels, so the decoder is
# identical for all operators.
# =====================================================================================================
class SE(nn.Module):
    """Squeeze-and-excitation gains: sigma(W2 ReLU(W1 GAP(x) + b1) + b2), shape (B, C, 1, 1)."""

    def __init__(self, c, r=16):
        super().__init__()
        h = max(c // r, 4)
        self.fc = nn.Sequential(nn.Linear(c, h), nn.ReLU(inplace=True), nn.Linear(h, c), nn.Sigmoid())

    def forward(self, x):
        return self.fc(x.mean((2, 3)))[..., None, None]


class ChannelAttention(nn.Module):
    """Eq. (2): F_CA = sigma(W2 delta(W1 z)) (.) S with z = GAP(S)."""

    def __init__(self, c, r=16):
        super().__init__()
        self.se = SE(c, r)

    def forward(self, s):
        return s * self.se(s)


class AttentionGate(nn.Module):
    """Eq. (3): alpha = sigma(psi(ReLU(W_s S + W_g g + b_g)) + b_psi); F_A = alpha (.) S.

    The gating signal g (the deeper decoder feature) is resampled to the spatial size of S.
    Returns (F_A, alpha).
    """

    def __init__(self, cs, cg, ci=None):
        super().__init__()
        ci = ci or max(cs // 2, 16)
        self.ws = nn.Conv2d(cs, ci, 1, bias=False)
        self.wg = nn.Conv2d(cg, ci, 1, bias=True)
        self.psi = nn.Conv2d(ci, 1, 1, bias=True)

    def forward(self, s, g):
        g = F.interpolate(g, size=s.shape[-2:], mode="bilinear", align_corners=False)
        alpha = torch.sigmoid(self.psi(F.relu(self.ws(s) + self.wg(g))))
        return s * alpha, alpha


# ------------------------------------------------------------------ fusion operators
class BiFAM(nn.Module):
    """Bilinear Feature Aggregation Module (Algorithm 1, Eqs. 5-12).

    F'_CA = W_CA * F_CA + b_CA,  F'_A = W_A * F_A + b_A        channel alignment (1x1 convs)
    F_AG  = F'_CA (.) F'_A                                     Hadamard interaction
    gamma = sigma(W2 ReLU(W1 GAP(F_AG) + b1) + b2)             recalibration, r = 16
    F_out = gamma (.) F_AG
    """

    def __init__(self, c_ca, c_a, c_t, r=16, recalibrate=True):
        super().__init__()
        self.align_ca = nn.Conv2d(c_ca, c_t, 1)
        self.align_a = nn.Conv2d(c_a, c_t, 1)
        self.se = SE(c_t, r) if recalibrate else None

    def forward(self, f_ca, f_a):
        f = self.align_ca(f_ca) * self.align_a(f_a)
        return f * self.se(f) if self.se is not None else f


class AddFusion(nn.Module):
    def __init__(self, c_ca, c_a, c_t, r=16):
        super().__init__()
        self.proj = nn.Identity() if c_ca == c_a == c_t else nn.Conv2d(c_ca, c_t, 1)

    def forward(self, f_ca, f_a):
        return self.proj(f_ca + f_a)


class ConcatFusion(nn.Module):
    """Concatenation followed by a 1x1 projection to C_T (optionally SE-recalibrated)."""

    def __init__(self, c_ca, c_a, c_t, r=16, recalibrate=False):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_ca + c_a, c_t, 1, bias=False), nn.BatchNorm2d(c_t), nn.ReLU(inplace=True))
        self.se = SE(c_t, r) if recalibrate else None

    def forward(self, f_ca, f_a):
        f = self.proj(torch.cat([f_ca, f_a], 1))
        return f * self.se(f) if self.se is not None else f


class CBAMFusion(nn.Module):
    """Concatenation + projection followed by CBAM channel then spatial attention (Woo et al., 2018)."""

    def __init__(self, c_ca, c_a, c_t, r=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_ca + c_a, c_t, 1, bias=False), nn.BatchNorm2d(c_t), nn.ReLU(inplace=True))
        h = max(c_t // r, 4)
        self.mlp = nn.Sequential(nn.Conv2d(c_t, h, 1), nn.ReLU(inplace=True), nn.Conv2d(h, c_t, 1))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, f_ca, f_a):
        f = self.proj(torch.cat([f_ca, f_a], 1))
        f = f * torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(f, 1)) + self.mlp(F.adaptive_max_pool2d(f, 1)))
        return f * torch.sigmoid(self.spatial(torch.cat([f.mean(1, keepdim=True), f.amax(1, keepdim=True)], 1)))


class AFFFusion(nn.Module):
    """Attentional feature fusion (Dai et al., 2021): Z = M(X+Y) (.) X + (1 - M(X+Y)) (.) Y, M = MS-CAM."""

    def __init__(self, c_ca, c_a, c_t, r=16):
        super().__init__()
        self.ax, self.ay = nn.Conv2d(c_ca, c_t, 1), nn.Conv2d(c_a, c_t, 1)
        h = max(c_t // r, 4)
        mk = lambda: nn.Sequential(nn.Conv2d(c_t, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU(inplace=True),
                                   nn.Conv2d(h, c_t, 1, bias=False), nn.BatchNorm2d(c_t))
        self.local, self.glob = mk(), mk()

    def forward(self, f_ca, f_a):
        x, y = self.ax(f_ca), self.ay(f_a)
        s = x + y
        m = torch.sigmoid(self.local(s) + self.glob(F.adaptive_avg_pool2d(s, 1)))
        return m * x + (1 - m) * y


class LowRankBilinear(nn.Module):
    """Hadamard low-rank bilinear pooling (Kim et al., 2017): P^T (tanh(U^T x) (.) tanh(V^T y))."""

    def __init__(self, c_ca, c_a, c_t, r=16, rank=None):
        super().__init__()
        rank = rank or c_t
        self.u, self.v = nn.Conv2d(c_ca, rank, 1), nn.Conv2d(c_a, rank, 1)
        self.p = nn.Conv2d(rank, c_t, 1)

    def forward(self, f_ca, f_a):
        return self.p(torch.tanh(self.u(f_ca)) * torch.tanh(self.v(f_a)))


FUSIONS = {
    "add": AddFusion,
    "product": lambda a, b, c, r=16: BiFAM(a, b, c, r, recalibrate=False),   # product, no recalibration
    "concat": ConcatFusion,
    "concat_se": lambda a, b, c, r=16: ConcatFusion(a, b, c, r, recalibrate=True),  # recalibration over concatenation
    "cbam": CBAMFusion,
    "aff": AFFFusion,
    "lowrank": LowRankBilinear,
    "bifam": BiFAM,
}


def make_fusion(name, c_ca, c_a, c_t, r=16):
    if name not in FUSIONS:
        raise ValueError(f"unknown fusion '{name}', choose from {sorted(FUSIONS)}")
    return FUSIONS[name](c_ca, c_a, c_t, r)


# =====================================================================================================
# Network
# Attention-gated Dense U-Net with a BiFAM decoder and a vision-transformer classification head.
#
# Algorithm 2 of the manuscript:
#   (S1, S2, S3, Z) <- DenseNet201(x)                 skips at 128^2, 64^2, 32^2 (256, 512, 1792 ch); bottleneck 16^2 (1920 ch)
#   for l = 3 .. 1:  F_CA <- CA(S_l); F_A <- AG(S_l, D); F_out <- BiFAM(F_CA, F_A); D <- Conv(Conv([F_out, Up(D)]))
#   D <- Upsample(D) to 512 x 512
#   z0 <- [x_cls; Patch16(D) E] + E_pos                1,024 tokens + class token
#   z6 <- 6 pre-norm transformer layers (MSA + ReLU FFN)
#   y  <- softmax / sigmoid of W_h LN(z6[cls])
#
# Ablation switches (Table 6): fusion='concat' | use_ag=False | use_ca=False | head='pool' | encoder='unet'.
# =====================================================================================================
DEFAULT_MODEL = dict(num_outputs=4, img_size=512, encoder="densenet201", pretrained=True, fusion="bifam",
                     use_ca=True, use_ag=True, head="vit", dec_channels=(256, 128, 64), dim=768, depth=6, heads=12,
                     mlp_ratio=4.0, patch=16, dropout=0.1, reduction=16)


def conv_bn_relu(ci, co, k=3):
    return nn.Sequential(nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


# ------------------------------------------------------------------ encoders
class DenseNet201Encoder(nn.Module):
    """torchvision DenseNet201 split into skips S1..S3 and bottleneck Z (Section 2.4).

    The three RGB kernels of the ImageNet stem are averaged into one kernel for single-channel input.
    """
    channels = (256, 512, 1792, 1920)

    def __init__(self, pretrained=True, in_channels=1):
        super().__init__()
        from torchvision.models import DenseNet201_Weights, densenet201
        try:
            f = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None).features
        except Exception as e:                                                  # offline: no ImageNet weights
            print(f"[model] could not load ImageNet DenseNet201 weights ({e}); using random initialisation")
            f = densenet201(weights=None).features
        w = f.conv0.weight.data
        f.conv0 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        f.conv0.weight.data = w.mean(1, keepdim=True).repeat(1, in_channels, 1, 1)
        self.stem = nn.Sequential(f.conv0, f.norm0, f.relu0, f.pool0)            # 1/4
        self.b1, self.t1 = f.denseblock1, f.transition1                         # S1: 256 ch, 1/4
        self.b2, self.t2 = f.denseblock2, f.transition2                         # S2: 512 ch, 1/8
        self.b3, self.t3 = f.denseblock3, f.transition3                         # S3: 1792 ch, 1/16
        self.b4, self.n5 = f.denseblock4, f.norm5                               # Z: 1920 ch, 1/32

    def forward(self, x):
        s1 = self.b1(self.stem(x))
        s2 = self.b2(self.t1(s1))
        s3 = self.b3(self.t2(s2))
        z = F.relu(self.n5(self.b4(self.t3(s3))))
        return [s1, s2, s3], z


class PlainUNetEncoder(nn.Module):
    """Randomly initialised plain U-Net encoder (ablation): double 3x3 conv blocks, max pooling.

    A strided stem keeps the skip resolutions of DenseNet201 (1/4, 1/8, 1/16) and the bottleneck at 1/32.
    """
    channels = (128, 256, 512, 1024)

    def __init__(self, pretrained=False, in_channels=1):
        super().__init__()
        c = self.channels
        self.stem = nn.Sequential(conv_bn_relu(in_channels, 64), nn.MaxPool2d(2), conv_bn_relu(64, 64), nn.MaxPool2d(2))
        self.blocks = nn.ModuleList([nn.Sequential(conv_bn_relu(ci, co), conv_bn_relu(co, co))
                                     for ci, co in zip((64,) + c[:-1], c)])

    def forward(self, x):
        x = self.stem(x); skips = []
        for i, b in enumerate(self.blocks):
            x = b(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        return skips[:3], skips[3]


# ------------------------------------------------------------------ decoder
class DecoderStage(nn.Module):
    """Figure 4 / Eq. (4): D_l = Conv(Conv([BiFAM(CA(S), AG(S, D_{l+1})), Up(D_{l+1})]))."""

    def __init__(self, cs, cd, co, fusion="bifam", use_ca=True, use_ag=True, r=16):
        super().__init__()
        self.ca = ChannelAttention(cs, r) if use_ca else None
        self.ag = AttentionGate(cs, cd) if use_ag else None
        self.fuse = make_fusion(fusion, cs, cs, cs, r) if use_ag else None
        self.conv = nn.Sequential(conv_bn_relu(cs + cd, co), conv_bn_relu(co, co))

    def forward(self, s, d):
        f_ca = self.ca(s) if self.ca is not None else s
        alpha = None
        if self.ag is not None:
            f_a, alpha = self.ag(s, d)
            f_out = self.fuse(f_ca, f_a)
        else:                                   # attention-gated skip pathway removed: single skip F_CA
            f_out = f_ca
        up = F.interpolate(d, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([f_out, up], 1)), f_out, alpha


# ------------------------------------------------------------------ transformer head
class ViTHead(nn.Module):
    """Eqs. (13)-(15): 16x16 patch embedding of the restored-resolution map, class token, learned positions,
    L pre-norm layers (multi-head self-attention + ReLU FFN), LayerNorm and a linear classifier."""

    def __init__(self, c_in, img_size=512, patch=16, dim=768, depth=6, heads=12, mlp_ratio=4.0, dropout=0.1, num_outputs=4):
        super().__init__()
        n = (img_size // patch) ** 2
        self.embed = nn.Conv2d(c_in, dim, patch, stride=patch)                  # equivalent to flatten + E
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, int(dim * mlp_ratio), dropout, activation="relu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, num_outputs)
        self.drop = nn.Dropout(dropout)

    def forward(self, d):
        t = self.embed(d).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        if t.shape[1] != self.pos.shape[1]:                                     # other input sizes: interpolate positions
            g = int((self.pos.shape[1] - 1) ** 0.5); h = w = int((t.shape[1] - 1) ** 0.5)
            p = F.interpolate(self.pos[:, 1:].reshape(1, g, g, -1).permute(0, 3, 1, 2), size=(h, w), mode="bicubic")
            pos = torch.cat([self.pos[:, :1], p.flatten(2).transpose(1, 2)], 1)
        else:
            pos = self.pos
        feat = self.norm(self.blocks(self.drop(t + pos)))[:, 0]
        return self.fc(feat), feat


class PoolHead(nn.Module):
    """Ablation: global average pooling and a linear layer in place of the transformer."""

    def __init__(self, c_in, num_outputs=4, dropout=0.1, **_):
        super().__init__()
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(c_in, num_outputs)

    def forward(self, d):
        feat = d.mean((2, 3))
        return self.fc(self.drop(feat)), feat


# ------------------------------------------------------------------ full network
class BiFAMNet(nn.Module):
    """DenseNet201 U-Net with channel attention, attention-gated dual skips, BiFAM and a ViT head.

    forward(x) returns a dict with
      logits  (B, K) - softmax logits (K = 4 or 3) or a single HGG logit (K = 1)
      feat    (B, dim) - class-token feature (t-SNE, source probe)
      and, when return_maps=True, 'fout' / 'alpha' lists (decoder order 32^2, 64^2, 128^2) and 'dec' (last decoder map).
    """

    def __init__(self, **cfg):
        super().__init__()
        c = {**DEFAULT_MODEL, **cfg}; self.cfg = c
        enc = {"densenet201": DenseNet201Encoder, "unet": PlainUNetEncoder}[c["encoder"]]
        self.encoder = enc(pretrained=c["pretrained"] and c["encoder"] == "densenet201", in_channels=c.get("in_channels", 1))
        cs = enc.channels
        dc = list(c["dec_channels"])
        cin = [cs[3], dc[0], dc[1]]
        self.decoder = nn.ModuleList([DecoderStage(cs[2 - i], cin[i], dc[i], c["fusion"], c["use_ca"], c["use_ag"], c["reduction"])
                                      for i in range(3)])
        if c["head"] == "vit":
            self.head = ViTHead(dc[-1], c["img_size"], c["patch"], c["dim"], c["depth"], c["heads"], c["mlp_ratio"],
                                c["dropout"], c["num_outputs"])
        else:
            self.head = PoolHead(dc[-1], c["num_outputs"], c["dropout"])

    def encoder_parameters(self):
        return self.encoder.parameters()

    def forward(self, x, return_maps=False):
        skips, d = self.encoder(x)
        fouts, alphas = [], []
        for i, stage in enumerate(self.decoder):
            d, f_out, alpha = stage(skips[2 - i], d)
            fouts.append(f_out); alphas.append(alpha)
        dec = d
        d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        logits, feat = self.head(d)
        out = {"logits": logits, "feat": feat}
        if return_maps:
            out.update(fout=fouts, alpha=alphas, dec=dec)
        return out


# ------------------------------------------------------------------ BiFAM-Lite
DEFAULT_LITE = dict(num_outputs=4, img_size=512, in_channels=1, backbone="densenet121", pretrained=True, fusion="bifam",
                    use_ca=True, use_ag=True, head="vit", skip_channels=(64, 96, 128), bottleneck_channels=256,
                    dec_channels=(128, 96, 64), dim=256, depth=6, heads=8, mlp_ratio=4.0, patch=4, dropout=0.1,
                    reduction=16, seg_head=True)


class TimmEncoder(nn.Module):
    """Any timm backbone; returns the skips at strides 4, 8, 16 and the stride-32 bottleneck."""

    def __init__(self, name="densenet121", pretrained=True, in_channels=1):
        super().__init__()
        import timm
        try:
            self.net = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_channels)
        except Exception as e:                                                  # offline: no ImageNet weights
            if not pretrained:
                raise
            print(f"[model] could not load ImageNet weights for {name} ({e}); using random initialisation")
            self.net = timm.create_model(name, pretrained=False, features_only=True, in_chans=in_channels)
        red, ch = self.net.feature_info.reduction(), self.net.feature_info.channels()
        self.idx = [max(i for i, r in enumerate(red) if r == s) for s in (4, 8, 16, 32)]
        self.channels = tuple(ch[i] for i in self.idx)

    def forward(self, x):
        f = self.net(x)
        s = [f[i] for i in self.idx]
        return s[:3], s[3]


class BiFAMLite(nn.Module):
    """Light BiFAM network: timm encoder (DenseNet121 by default), 1x1-reduced skips, the same channel-attention /
    attention-gate / BiFAM decoder, a transformer reading 4x4 patches of the 128^2 decoder map directly (1,024 tokens
    for a 512^2 input, no upsampling), and an auxiliary segmentation head used for mask supervision.

    forward(x) -> {'logits', 'feat', 'seg'} (+ 'fout', 'alpha', 'dec' when return_maps=True).
    """

    def __init__(self, **cfg):
        super().__init__()
        c = {**DEFAULT_LITE, **cfg}; self.cfg = c
        self.encoder = TimmEncoder(c["backbone"], c["pretrained"], c["in_channels"])
        cs, sc, dc = self.encoder.channels, list(c["skip_channels"]), list(c["dec_channels"])
        self.reduce = nn.ModuleList([conv_bn_relu(cs[i], sc[i], 1) for i in range(3)])
        self.bottleneck = conv_bn_relu(cs[3], c["bottleneck_channels"], 1)
        cin = [c["bottleneck_channels"], dc[0], dc[1]]
        self.decoder = nn.ModuleList([DecoderStage(sc[2 - i], cin[i], dc[i], c["fusion"], c["use_ca"], c["use_ag"], c["reduction"])
                                      for i in range(3)])
        dec_size = c["img_size"] // 4
        if c["head"] == "vit":
            self.head = ViTHead(dc[-1], dec_size, c["patch"], c["dim"], c["depth"], c["heads"], c["mlp_ratio"],
                                c["dropout"], c["num_outputs"])
        else:
            self.head = PoolHead(dc[-1], c["num_outputs"], c["dropout"])
        self.seg = nn.Conv2d(dc[-1], 1, 1) if c["seg_head"] else None

    def encoder_parameters(self):
        return self.encoder.parameters()

    def forward(self, x, return_maps=False):
        skips, z = self.encoder(x)
        skips = [r(s) for r, s in zip(self.reduce, skips)]
        d = self.bottleneck(z)
        fouts, alphas = [], []
        for i, stage in enumerate(self.decoder):
            d, f_out, alpha = stage(skips[2 - i], d)
            fouts.append(f_out); alphas.append(alpha)
        logits, feat = self.head(d)
        out = {"logits": logits, "feat": feat}
        if self.seg is not None:
            out["seg"] = F.interpolate(self.seg(d), size=x.shape[-2:], mode="bilinear", align_corners=False)
        if return_maps:
            out.update(fout=fouts, alpha=alphas, dec=d)
        return out


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_model(name="bifamnet", **cfg):
    """'bifamnet', 'bifamlite' or any baseline name from baselines.BASELINES."""
    if name == "bifamnet":
        return BiFAMNet(**cfg)
    if name == "bifamlite":
        return BiFAMLite(**cfg)
    return build_baseline(name, cfg.get("num_outputs", 4), cfg.get("img_size", 512), cfg.get("pretrained", True),
                          cfg.get("in_channels", 1))


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, checkpoint dict with 'cfg', 'task', 'classes', ...)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["model_name"], **{**ck["model_cfg"], "pretrained": False})
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


# =====================================================================================================
# Baselines
# The thirteen baselines of Table 5, all adapted to a single-channel 512 x 512 input.
#
# CNN and transformer baselines come from timm / torchvision with ImageNet weights. The three hybrid
# baselines are compact re-implementations of the design families cited in the manuscript
# (cross-attention between a CNN and a ViT stream [27], parallel-backbone fusion in the head [28],
# and a CNN encoder followed by selective state-space blocks [32]); they are not the original authors' code.
# =====================================================================================================
TIMM_NAMES = {
    "inceptionv3": "inception_v3", "xception": "legacy_xception", "resnet18": "resnet18", "resnet50": "resnet50",
    "resnet101": "resnet101", "densenet201": "densenet201", "mobilenetv2": "mobilenetv2_100",
    "vit": "vit_base_patch16_224", "swin": "swin_tiny_patch4_window7_224",
}
BASELINES = list(TIMM_NAMES) + ["shufflenet", "cross_attention_hybrid", "parallel_fusion", "state_space_hybrid"]


def _timm(name, num_outputs, img_size, pretrained, in_chans=1, **kw):
    import timm
    extra = {"img_size": img_size} if name.startswith(("vit", "swin")) else {}
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_outputs, in_chans=in_chans, **extra, **kw)
    except Exception:                                                        # offline: fall back to random init
        if not pretrained:
            raise
        print(f"[baselines] could not load pretrained weights for {name}; using random initialisation")
        return timm.create_model(name, pretrained=False, num_classes=num_outputs, in_chans=in_chans, **extra, **kw)


class _Wrap(nn.Module):
    """Uniform interface: forward(x) -> {'logits', 'feat'}."""

    def __init__(self, net, img_size=512):
        super().__init__()
        self.net, self.cfg = net, dict(img_size=img_size)

    def forward(self, x, return_maps=False):
        if hasattr(self.net, "forward_features") and hasattr(self.net, "forward_head"):
            f = self.net.forward_features(x)
            return {"logits": self.net.forward_head(f), "feat": self.net.forward_head(f, pre_logits=True)}
        y = self.net(x)
        return {"logits": y, "feat": y}


class ShuffleNet(nn.Module):
    def __init__(self, num_outputs, pretrained=True):
        super().__init__()
        from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0
        try:
            self.net = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None)
        except Exception:
            self.net = shufflenet_v2_x1_0(weights=None)
        w = self.net.conv1[0].weight.data
        self.net.conv1[0] = nn.Conv2d(1, 24, 3, 2, 1, bias=False); self.net.conv1[0].weight.data = w.mean(1, keepdim=True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_outputs)

    def forward(self, x, return_maps=False):
        n = self.net
        f = n.conv5(n.stage4(n.stage3(n.stage2(n.maxpool(n.conv1(x)))))).mean((2, 3))
        return {"logits": n.fc(f), "feat": f}


class _CNNTokens(nn.Module):
    """ResNet50 up to layer3 (1/16 resolution, 1024 channels) projected to `dim`-wide tokens."""

    def __init__(self, dim, pretrained):
        super().__init__()
        self.cnn = _timm("resnet50", 0, None, pretrained, features_only=True, out_indices=(3,))
        self.proj = nn.Conv2d(1024, dim, 1)

    def forward(self, x):
        return self.proj(self.cnn(x)[0]).flatten(2).transpose(1, 2)


class CrossAttentionHybrid(nn.Module):
    """Parallel CNN and ViT streams exchanging information through bidirectional cross-attention."""

    def __init__(self, num_outputs, img_size=512, pretrained=True, dim=384, heads=6):
        super().__init__()
        self.cnn = _CNNTokens(dim, pretrained)
        self.vit = _timm("vit_small_patch16_224", 0, img_size, pretrained)
        self.c2v = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.v2c = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.fc = nn.Linear(2 * dim, num_outputs)

    def forward(self, x, return_maps=False):
        c = self.cnn(x)
        v = self.vit.forward_features(x)
        v2 = self.n1(v + self.c2v(v, c, c, need_weights=False)[0])
        c2 = self.n2(c + self.v2c(c, v, v, need_weights=False)[0])
        f = torch.cat([v2[:, 0], c2.mean(1)], 1)
        return {"logits": self.fc(f), "feat": f}


class ParallelFusion(nn.Module):
    """Two parallel CNN backbones (EfficientNet-B0, ResNet50) and a ViT-Small stream, fused in the head."""

    def __init__(self, num_outputs, img_size=512, pretrained=True):
        super().__init__()
        self.a = _timm("efficientnet_b0", 0, None, pretrained)
        self.b = _timm("resnet50", 0, None, pretrained)
        self.c = _timm("vit_small_patch16_224", 0, img_size, pretrained)
        d = self.a.num_features + self.b.num_features + self.c.num_features
        self.fc = nn.Sequential(nn.Linear(d, 512), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(512, num_outputs))

    def forward(self, x, return_maps=False):
        f = torch.cat([self.a(x), self.b(x), self.c(x)], 1)
        return {"logits": self.fc(f), "feat": f}


class SelectiveSSM(nn.Module):
    """Selective state-space block with a scalar input-dependent decay per head (Mamba-2 / SSD dual form).

    h_t = a_t h_{t-1} + B_t x_t,  y_t = C_t h_t, evaluated exactly as a causal decay-masked product,
    run in both scan directions and followed by a gated output projection.
    """

    def __init__(self, dim, heads=8, state=16, expand=2):
        super().__init__()
        self.h, self.n, self.e = heads, state, dim * expand
        self.norm = nn.LayerNorm(dim)
        self.inp = nn.Linear(dim, 2 * self.e)
        self.conv = nn.Conv1d(self.e, self.e, 4, padding=3, groups=self.e)
        self.bc = nn.Linear(self.e, 2 * heads * state)
        self.dt = nn.Linear(self.e, heads)
        self.a_log = nn.Parameter(torch.zeros(heads))
        self.out = nn.Linear(self.e, dim)

    def _scan(self, u):
        Bsz, L, _ = u.shape
        b, c = self.bc(u).view(Bsz, L, 2, self.h, self.n).unbind(2)             # (B, L, H, N)
        loga = -F.softplus(self.dt(u)) * torch.exp(self.a_log)                  # log a_t <= 0, (B, L, H)
        cum = loga.cumsum(1).transpose(1, 2)                                    # (B, H, L)
        seg = cum[..., :, None] - cum[..., None, :]                             # log prod a over (s, t]
        mask = torch.ones(L, L, dtype=torch.bool, device=u.device).tril()
        decay = torch.exp(seg.masked_fill(~mask, float("-inf")))
        scores = torch.einsum("bthn,bshn->bhts", c, b) * decay                  # (B, H, L, L)
        x = u.view(Bsz, L, self.h, -1).transpose(1, 2)                          # (B, H, L, E/H)
        return (scores @ x).transpose(1, 2).reshape(Bsz, L, self.e)

    def forward(self, x):
        u, z = self.inp(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))
        y = self._scan(u) + self._scan(u.flip(1)).flip(1)
        return x + self.out(y * F.silu(z))


class StateSpaceHybrid(nn.Module):
    """ResNet50 (to layer3) tokens followed by eight bidirectional selective state-space blocks."""

    def __init__(self, num_outputs, img_size=512, pretrained=True, dim=768, depth=8):
        super().__init__()
        self.cnn = _CNNTokens(dim, pretrained)
        self.blocks = nn.Sequential(*[SelectiveSSM(dim) for _ in range(depth)])
        self.norm, self.fc = nn.LayerNorm(dim), nn.Linear(dim, num_outputs)

    def forward(self, x, return_maps=False):
        f = self.norm(self.blocks(self.cnn(x))).mean(1)
        return {"logits": self.fc(f), "feat": f}


def build_baseline(name, num_outputs=4, img_size=512, pretrained=True, in_channels=1):
    if name in TIMM_NAMES:
        m = _Wrap(_timm(TIMM_NAMES[name], num_outputs, img_size, pretrained, in_channels), img_size)
        m.cfg["in_channels"] = in_channels
        return m
    builders = dict(shufflenet=lambda: ShuffleNet(num_outputs, pretrained),
                    cross_attention_hybrid=lambda: CrossAttentionHybrid(num_outputs, img_size, pretrained),
                    parallel_fusion=lambda: ParallelFusion(num_outputs, img_size, pretrained),
                    state_space_hybrid=lambda: StateSpaceHybrid(num_outputs, img_size, pretrained))
    if name not in builders:
        raise ValueError(f"unknown baseline '{name}', choose from {BASELINES}")
    m = builders[name]()
    if in_channels != 1:                                   # multi-sequence input: learned 1x1 projection to one channel
        m = _ChannelAdapter(m, in_channels)
    m.cfg = dict(img_size=img_size, in_channels=in_channels)
    return m


class _ChannelAdapter(nn.Module):
    def __init__(self, net, in_channels):
        super().__init__()
        self.proj, self.net = nn.Conv2d(in_channels, 1, 1), net

    def forward(self, x, return_maps=False):
        return self.net(self.proj(x))


# =====================================================================================================
# Data
# Datasets, preprocessing, duplicate removal and partitions (Sections 2.1-2.2, Table 1).
#
# Collections
#   figshare  3,064 CE-T1 slices (.mat v7.3) from 233 patients: glioma 1,426, meningioma 708, pituitary 930 [58, 59]
#   Br35H     'no' folder of the Kaggle Br35H set: no-tumor class, 1,400 images after duplicate removal [60]
#   BraTS2015 training set, 220 HGG + 54 LGG cases; 50 axial T1ce slices per case, whole-tumor masks (OT) [41]
#
# prepare_*() convert the raw downloads into PNG slices plus an index CSV (one row per slice) with columns
#   path, label, group, source[, mask, case, slice]
# where `group` is the patient identifier (figshare), the case identifier (BraTS) or a near-duplicate
# cluster identifier (Br35H, which has no patient identifiers).
#
# Class indices follow Figure 9: 0 glioma, 1 meningioma, 2 pituitary tumor, 3 no tumor. HGG = 1, LGG = 0.
# =====================================================================================================
FOUR_CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
THREE_CLASSES = FOUR_CLASSES[:3]
BINARY_CLASSES = ["LGG", "HGG"]
FIGSHARE_LABEL = {1: 1, 2: 0, 3: 2}                  # figshare cjdata.label (1 men, 2 gli, 3 pit) -> our index
IMG_SIZE = 512
SLICES_PER_CASE = 50
SEQUENCES = ("T1", "T1c", "T2", "Flair")              # channel order of the multi-sequence BraTS input
DISPLAY_CHANNEL = 1                                    # T1ce is shown in figures


# ------------------------------------------------------------------ helpers
def to_uint8(img, lo=0.5, hi=99.5):
    """Per-image percentile clipping and scaling to [0, 255]; uses only the statistics of the slice itself."""
    img = np.asarray(img, np.float32)
    nz = img[img > 0] if (img > 0).any() else img.ravel()
    a, b = np.percentile(nz, [lo, hi])
    return (np.clip((img - a) / max(b - a, 1e-6), 0, 1) * 255).round().astype(np.uint8)


def resize(img, size=IMG_SIZE, nearest=False):
    im = Image.fromarray(img)
    if im.size != (size, size):
        im = im.resize((size, size), Image.NEAREST if nearest else Image.BILINEAR)
    return np.asarray(im)


def to_gray(im: Image.Image):
    return np.asarray(im.convert("L"))


# ------------------------------------------------------------------ duplicate detection (Supplementary S7)
def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def phash64(img):
    """64-bit perceptual hash: 32x32 greyscale, 2-D DCT, top-left 8x8 block (without DC) against its median."""
    from scipy.fft import dctn
    a = np.asarray(Image.fromarray(img).convert("L").resize((32, 32), Image.LANCZOS), np.float32)
    d = dctn(a, norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return int("".join("1" if b else "0" for b in bits), 2)


def _thumb(p, size=64):
    """size x size greyscale thumbnail scaled to [0, 1] (for the pixel-level near-duplicate check)."""
    a = np.asarray(Image.fromarray(to_gray(Image.open(p))).convert("L").resize((size, size), Image.BILINEAR), np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-6)


def duplicate_clusters(paths, max_hamming=4, max_pixel_diff=0.01):
    """Clusters of exact duplicates (identical MD5) and near duplicates (the same image re-saved or resized).

    Two images are near duplicates only if their 64-bit pHash differs in <= max_hamming bits AND their 64 x 64
    thumbnails differ by <= max_pixel_diff on average. Near duplicates are matched against the first image of each
    cluster only (no chaining), because MRI slices of different patients can have similar hashes and chaining
    would merge long runs of distinct images into one cluster.
    """
    n = len(paths)
    label = np.full(n, -1)
    first = {}
    for i, p in enumerate(paths):                                   # exact duplicates
        h = md5(p)
        if h in first:
            label[i] = label[first[h]]
        else:
            first[h] = i; label[i] = i
    hashes = np.array([phash64(to_gray(Image.open(p))) for p in paths], dtype=np.uint64)
    bits = np.unpackbits(hashes.view(np.uint8).reshape(n, 8), axis=1).astype(np.uint8)
    thumbs = {}
    reps = []                                                       # indices of cluster representatives
    for i in range(n):
        if label[i] != i:                                           # exact duplicate of an earlier image
            continue
        match = None
        if reps:
            r = np.array(reps)
            for j in r[(bits[r] != bits[i]).sum(1) <= max_hamming]:
                if j not in thumbs:
                    thumbs[j] = _thumb(paths[j])
                if i not in thumbs:
                    thumbs[i] = _thumb(paths[i])
                if np.abs(thumbs[i] - thumbs[j]).mean() <= max_pixel_diff:
                    match = int(j); break
        if match is None:
            reps.append(i)
        else:
            label[i] = label[match]
    for i in range(n):                                              # exact copies follow their original
        label[i] = label[label[i]]
    return label


# ------------------------------------------------------------------ figshare
def _mat_string(ds):
    return "".join(chr(int(c)) for c in np.asarray(ds).ravel())


def prepare_figshare(raw_dir, out_dir):
    """raw_dir: folder containing the figshare .mat files (1.mat ... 3064.mat), possibly in sub-folders."""
    import h5py
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    (out_dir / "figshare").mkdir(parents=True, exist_ok=True)
    files = sorted(raw_dir.rglob("*.mat"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    files = [f for f in files if f.stem.isdigit()]
    rows = []
    for f in files:
        with h5py.File(f, "r") as h:
            cj = h["cjdata"]
            label = FIGSHARE_LABEL[int(np.asarray(cj["label"]).ravel()[0])]
            pid = _mat_string(cj["PID"]).strip()
            img = np.asarray(cj["image"]).T
            mask = np.asarray(cj["tumorMask"]).T if "tumorMask" in cj else None
        rel = f"figshare/{int(f.stem):04d}.png"
        Image.fromarray(resize(to_uint8(img))).save(out_dir / rel)
        row = dict(path=rel, label=label, group=f"fs_{pid}", source="figshare", mask="")
        if mask is not None:
            mrel = f"figshare/{int(f.stem):04d}_mask.png"
            Image.fromarray(resize((mask > 0).astype(np.uint8) * 255, nearest=True)).save(out_dir / mrel)
            row["mask"] = mrel
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "figshare.csv", index=False)
    return df


# ------------------------------------------------------------------ Br35H
def prepare_br35h(raw_dir, out_dir, n_images=1400, max_hamming=4, seed=0):
    """raw_dir: the Br35H download; its 'no' folder provides the no-tumor class.

    Exact and near duplicates are grouped into clusters; one image per cluster is kept, so no duplicate
    can fall on both sides of a split. `n_images` images are then drawn (1,400 in the manuscript).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    (out_dir / "br35h").mkdir(parents=True, exist_ok=True)
    cands = [d for d in raw_dir.rglob("*") if d.is_dir() and d.name.lower() == "no"]
    if not cands:
        raise FileNotFoundError(f"no 'no' folder under {raw_dir}")
    paths = sorted(p for p in cands[0].iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    cl = duplicate_clusters(paths, max_hamming)
    keep = sorted({c: i for i, c in reversed(list(enumerate(cl)))}.values())   # first image of every cluster
    removed = len(paths) - len(keep)
    rng = np.random.default_rng(seed)
    if n_images and len(keep) > n_images:
        keep = sorted(rng.choice(keep, n_images, replace=False).tolist())
    rows = []
    for k, i in enumerate(keep):
        rel = f"br35h/{k:04d}.png"
        Image.fromarray(resize(to_uint8(to_gray(Image.open(paths[i])), 0, 100))).save(out_dir / rel)
        rows.append(dict(path=rel, label=3, group=f"br35h_{cl[i]}", source="br35h", mask="", original=paths[i].name))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "br35h.csv", index=False)
    json.dump(dict(candidates=len(paths), duplicates_removed=int(removed), kept=len(df), max_hamming=max_hamming),
              open(out_dir / "br35h_dedup.json", "w"), indent=2)
    if removed > 0.2 * len(paths):
        print(f"[br35h] WARNING: {removed} of {len(paths)} images were flagged as duplicates - check the duplicate filter")
    return df


def cross_source_duplicates(df, root, max_hamming=4):
    """Report near-duplicate pairs across collections of a combined index (diagnostic)."""
    cl = duplicate_clusters([Path(root) / p for p in df.path], max_hamming)
    g = pd.Series(cl).groupby(cl).size()
    return df.assign(dup_cluster=cl)[pd.Series(cl).map(g).values > 1]


# ------------------------------------------------------------------ BraTS 2015
def _read_mha(path):
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))          # (z, y, x) = (155, 240, 240)


def select_slices(vol, k=SLICES_PER_CASE):
    """Image-only slice selection: the k axial slices with the largest brain (non-zero) area, in axial order."""
    area = (vol > 0).reshape(len(vol), -1).sum(1)
    return np.sort(np.argsort(-area, kind="stable")[:k])


def _find_sequence(case, seq):
    pat = re.compile(rf"MR_{seq}\.", re.I)
    return next((p for p in case.rglob("*.mha") if pat.search(p.name)), None)


def prepare_brats2015(raw_dir, out_dir, k=SLICES_PER_CASE, multiseq=True):
    """raw_dir: BRATS2015_Training with HGG/ and LGG/ case folders (VSD.Brain.XX.O.MR_T1c.*, VSD.Brain_*.OT.*).

    Writes the T1ce slice as PNG (column `path`) and, when all four sequences exist and multiseq=True, the
    co-registered T1 / T1ce / T2 / FLAIR slice as a compressed uint8 array (column `path4`, channel order SEQUENCES).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    rows = []
    for grade, lab in (("HGG", 1), ("LGG", 0)):
        cases = sorted(d for d in (raw_dir / grade).iterdir() if d.is_dir()) if (raw_dir / grade).exists() else []
        for case in cases:
            t1c = next(case.rglob("*T1c*.mha"), None)
            ot = next(case.rglob("*OT*.mha"), None)
            if t1c is None:
                print(f"[brats] skip {case}: no T1c volume"); continue
            vol = _read_mha(t1c).astype(np.float32)
            seg = _read_mha(ot) if ot is not None else np.zeros_like(vol, np.uint8)
            seqs = None
            if multiseq:
                files = [_find_sequence(case, q) for q in SEQUENCES]
                if all(files):
                    seqs = [_read_mha(f).astype(np.float32) for f in files]
            cid = f"{grade}_{case.name}"
            (out_dir / "brats2015" / cid).mkdir(parents=True, exist_ok=True)
            for z in select_slices(vol, k):
                rel = f"brats2015/{cid}/{z:03d}.png"; mrel = f"brats2015/{cid}/{z:03d}_mask.png"
                Image.fromarray(resize(to_uint8(vol[z]))).save(out_dir / rel)
                Image.fromarray(resize((seg[z] > 0).astype(np.uint8) * 255, nearest=True)).save(out_dir / mrel)
                p4 = ""
                if seqs is not None:
                    p4 = f"brats2015/{cid}/{z:03d}_4seq.npz"
                    np.savez_compressed(out_dir / p4, x=np.stack([to_uint8(v[z]) if v[z].any() else np.zeros(v[z].shape, np.uint8)
                                                                  for v in seqs]))
                rows.append(dict(path=rel, label=lab, group=cid, source="brats2015", mask=mrel, case=cid, slice=int(z),
                                 mask_pixels=int((seg[z] > 0).sum()), path4=p4))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "brats2015.csv", index=False)
    return df


# ------------------------------------------------------------------ partitions
def _inner_val(train, seed, frac, by_group):
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split
    if by_group:
        k = max(2, int(round(1 / frac)))
        tr, va = next(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(train, train.label, train.group))
        return train.iloc[tr], train.iloc[va]
    tr, va = train_test_split(np.arange(len(train)), test_size=frac, stratify=train.label, random_state=seed)
    return train.iloc[tr], train.iloc[va]


def split_image_level(df, test_size=0.2, seed=0, val_frac=0.1):
    """Stratified 80:20 image-level split without reference to the patient identifier (benchmark protocol)."""
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(df)), test_size=test_size, stratify=df.label, random_state=seed)
    train, test = df.iloc[tr], df.iloc[te]
    train, val = _inner_val(train, seed, val_frac, by_group=False)
    return dict(train=train.path.tolist(), val=val.path.tolist(), test=test.path.tolist())


def split_patient_level(df, n_test_patients=46, seed=0, val_frac=0.1):
    """Patient-level repartition of the three figshare classes (187 training / 46 test patients)."""
    from sklearn.model_selection import train_test_split
    pat = df.groupby("group").label.first()
    tr_p, te_p = train_test_split(pat.index.values, test_size=n_test_patients, stratify=pat.values, random_state=seed)
    train, test = df[df.group.isin(tr_p)], df[df.group.isin(te_p)]
    train, val = _inner_val(train, seed, val_frac, by_group=True)
    return dict(train=train.path.tolist(), val=val.path.tolist(), test=test.path.tolist())


def split_case_cv(df, n_folds=5, seed=0, val_frac=0.2):
    """Stratified case-level k-fold CV; all slices of a case share a fold. Inner validation split by case."""
    from sklearn.model_selection import StratifiedKFold
    cases = df.groupby("group").label.first()
    folds = []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(cases.index, cases.values):
        train = df[df.group.isin(cases.index[tr])]
        test = df[df.group.isin(cases.index[te])]
        train, val = _inner_val(train, seed, val_frac, by_group=True)
        folds.append(dict(train=train.path.tolist(), val=val.path.tolist(), test=test.path.tolist()))
    return folds


def load_index(root, task):
    """Combined index for a task: 'fourclass', 'threeclass' (figshare only) or 'brats'."""
    root = Path(root)
    if task == "brats":
        return pd.read_csv(root / "brats2015.csv", keep_default_na=False)
    fs = pd.read_csv(root / "figshare.csv", keep_default_na=False)
    if task == "threeclass":
        return fs
    return pd.concat([fs, pd.read_csv(root / "br35h.csv", keep_default_na=False)], ignore_index=True)


def make_splits(root, seed=0):
    """Writes the partition files used by all experiments into <root>/splits/."""
    root = Path(root); (root / "splits").mkdir(exist_ok=True)
    out = {}
    if (root / "figshare.csv").exists() and (root / "br35h.csv").exists():
        out["fourclass_image"] = split_image_level(load_index(root, "fourclass"), seed=seed)
    if (root / "figshare.csv").exists():
        fs = load_index(root, "threeclass")
        out["threeclass_image"] = split_image_level(fs, seed=seed)
        out["threeclass_patient"] = split_patient_level(fs, seed=seed)
    if (root / "brats2015.csv").exists():
        out["brats_cv"] = split_case_cv(load_index(root, "brats"), seed=seed)
    for k, v in out.items():
        json.dump(v, open(root / "splits" / f"{k}.json", "w"))
    return out


def load_split(root, name, fold=None):
    s = json.load(open(Path(root) / "splits" / f"{name}.json"))
    return s[fold] if fold is not None else s


TASKS = {  # split name -> (index task, class names, number of outputs, evaluation unit)
    "fourclass_image": ("fourclass", FOUR_CLASSES, 4, "image"),
    "threeclass_image": ("threeclass", THREE_CLASSES, 3, "image"),
    "threeclass_patient": ("threeclass", THREE_CLASSES, 3, "patient"),
    "brats_cv": ("brats", BINARY_CLASSES, 1, "case"),
}


# ------------------------------------------------------------------ torch dataset
def load_slice(root, r, size=IMG_SIZE, channels=1):
    """(C, size, size) float tensor in [0, 1]. channels=4 uses the multi-sequence array when the row has one;
    single-sequence rows are repeated so that one model can read mixed collections."""
    root = Path(root)
    p4 = r.get("path4", "") if hasattr(r, "get") else ""
    if channels > 1 and isinstance(p4, str) and p4:
        a = np.load(root / p4)["x"]
        a = np.stack([resize(c, size) for c in a])
    else:
        a = resize(np.asarray(Image.open(root / r.path).convert("L")), size)[None]
        if channels > 1:
            a = np.repeat(a, channels, 0)
    return torch.from_numpy(np.ascontiguousarray(a)).float() / 255


def has_multiseq(root):
    p = Path(root) / "brats2015.csv"
    if not p.exists():
        return False
    df = pd.read_csv(p, keep_default_na=False)
    return "path4" in df and (df.path4 != "").all()


class SliceDataset:
    """Loads PNG slices, applies training-only augmentation and per-image z-score normalisation.

    Augmentation: horizontal flip, rotation (+-10 deg), scaling (0.9-1.1), brightness/contrast/gamma jitter.
    Vertical flipping is excluded because it inverts the anatomical orientation.
    """

    def __init__(self, root, df, train=False, size=IMG_SIZE, with_mask=False, channels=1):
        self.root, self.df, self.train, self.size, self.with_mask = Path(root), df.reset_index(drop=True), train, size, with_mask
        self.channels = channels

    def __len__(self):
        return len(self.df)

    def _augment(self, x, m, rng):
        from torchvision.transforms.v2 import functional as TF
        if rng.random() < 0.5:
            x = TF.horizontal_flip(x); m = TF.horizontal_flip(m) if m is not None else None
        ang, sc = float(rng.uniform(-10, 10)), float(rng.uniform(0.9, 1.1))
        x = TF.affine(x, angle=ang, translate=[0, 0], scale=sc, shear=[0.0], interpolation=TF.InterpolationMode.BILINEAR)
        if m is not None:
            m = TF.affine(m, angle=ang, translate=[0, 0], scale=sc, shear=[0.0], interpolation=TF.InterpolationMode.NEAREST)
        x = x.clamp(0, 1) ** float(rng.uniform(0.8, 1.25))
        x = (x - 0.5) * float(rng.uniform(0.9, 1.1)) + 0.5 + float(rng.uniform(-0.05, 0.05))
        return x, m

    def __getitem__(self, i):
        r = self.df.iloc[i]
        x = load_slice(self.root, r, self.size, self.channels)
        m = None
        if self.with_mask:
            m = torch.zeros_like(x) if not r.get("mask", "") else \
                torch.from_numpy(resize(np.asarray(Image.open(self.root / r["mask"]).convert("L")), self.size, True)).float()[None] / 255
        if self.train:
            x, m = self._augment(x, m, np.random.default_rng())
        mu = x.mean((1, 2), keepdim=True); sd = x.std((1, 2), keepdim=True)
        x = (x - mu) / (sd + 1e-6)                                          # per-image (per-channel) z-score
        out = dict(x=x, y=torch.tensor(float(r.label)), idx=i)
        if m is not None:
            out["mask"] = (m > 0.5).float()
            # masks are known for figshare (tumorMask), BraTS (OT) and Br35H no-tumor slices (empty)
            out["has_mask"] = torch.tensor(float(bool(r.get("mask", "")) or r.get("source", "") == "br35h"))
        return out


# =====================================================================================================
# Metrics and statistics
# Metrics and statistics (Section 2.10).
#
# * per-class precision, recall, specificity, F1; macro F1 (primary metric, Eq. 18); accuracy; balanced
#   accuracy; Cohen's kappa; Matthews correlation; macro one-vs-rest AUC (four/three classes) or binary AUC
# * case-level aggregation for HGG/LGG (Eq. 19): majority vote of slice labels, mean HGG probability as score
# * Wilson score intervals; percentile bootstrap (2,000 replicates) resampling the unit of evaluation;
#   paired bootstrap of the macro-F1 difference; exact McNemar test; seed-level paired t-test
# =====================================================================================================
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def macro_f1(y, pred, k):
    cm = confusion_matrix(y, pred, labels=range(k))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return float(np.mean(np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)))


def report(y, pred, prob=None, classes=None):
    """Full classification report. `prob`: (N, K) class probabilities, or (N,) positive-class probability."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    k = len(classes) if classes else int(max(y.max(), pred.max()) + 1)
    classes = classes or [str(i) for i in range(k)]
    cm = confusion_matrix(y, pred, labels=range(k))
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp; tn = cm.sum() - tp - fp - fn
    per = {}
    for i, c in enumerate(classes):
        prec = tp[i] / max(tp[i] + fp[i], 1); rec = tp[i] / max(tp[i] + fn[i], 1)
        per[c] = dict(n=int(cm[i].sum()), precision=prec, precision_ci=wilson(tp[i], tp[i] + fp[i]),
                      recall=rec, recall_ci=wilson(tp[i], tp[i] + fn[i]), specificity=tn[i] / max(tn[i] + fp[i], 1),
                      f1=2 * tp[i] / max(2 * tp[i] + fp[i] + fn[i], 1))
    out = dict(n=len(y), accuracy=float((y == pred).mean()), accuracy_ci=wilson(int((y == pred).sum()), len(y)),
               macro_precision=float(np.mean([v["precision"] for v in per.values()])),
               macro_recall=float(np.mean([v["recall"] for v in per.values()])),
               macro_specificity=float(np.mean([v["specificity"] for v in per.values()])),
               macro_f1=float(np.mean([v["f1"] for v in per.values()])),
               balanced_accuracy=float(np.mean([v["recall"] for v in per.values()])),
               kappa=float(cohen_kappa_score(y, pred)), mcc=float(matthews_corrcoef(y, pred)),
               confusion=cm.tolist(), per_class=per)
    if prob is not None:
        prob = np.asarray(prob)
        try:
            if prob.ndim == 1:
                out["auc"] = float(roc_auc_score(y, prob))
            else:
                out["auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro", labels=range(k)))
                for i, c in enumerate(classes):
                    per[c]["auc"] = float(roc_auc_score(y == i, prob[:, i]))
        except ValueError:
            out["auc"] = float("nan")
    return out


def aggregate_cases(groups, slice_prob, threshold=0.5):
    """Eq. (19): case label = majority vote of slice labels (ties -> mean probability >= threshold);
    case score = mean slice HGG probability. Returns (case ids, case predictions, case scores)."""
    groups = np.asarray(groups); p = np.asarray(slice_prob, float)
    ids = np.unique(groups)
    pred, score = [], []
    for g in ids:
        q = p[groups == g]; votes = (q >= threshold).mean()
        score.append(q.mean())
        pred.append(int(votes > 0.5 or (votes == 0.5 and q.mean() >= threshold)))
    return ids, np.array(pred), np.array(score)


def bootstrap_ci(y, pred, k, groups=None, n_boot=2000, seed=0, fn=None):
    """Percentile 95% CI of macro F1 (or `fn(y, pred)`), resampling units (rows, or whole groups)."""
    fn = fn or (lambda a, b: macro_f1(a, b, k))
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    units, idx = _units(y, groups)
    vals = [fn(y[i], pred[i]) for i in (_draw(units, idx, rng) for _ in range(n_boot))]
    return tuple(np.percentile(vals, [2.5, 97.5]))


def paired_bootstrap(y, pred_a, pred_b, k, groups=None, n_boot=2000, seed=0):
    """Paired bootstrap of macro F1(a) - macro F1(b); both models' predictions are resampled together."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    units, idx = _units(y, groups)
    d = []
    for _ in range(n_boot):
        i = _draw(units, idx, rng)
        d.append(macro_f1(y[i], a[i], k) - macro_f1(y[i], b[i], k))
    return dict(diff=macro_f1(y, a, k) - macro_f1(y, b, k), ci=tuple(np.percentile(d, [2.5, 97.5])),
                p_boot=float(2 * min(np.mean(np.array(d) <= 0), np.mean(np.array(d) >= 0))))


def _units(y, groups):
    if groups is None:
        return np.arange(len(y)), None
    groups = np.asarray(groups); u = np.unique(groups)
    return u, {g: np.nonzero(groups == g)[0] for g in u}


def _draw(units, idx, rng):
    s = rng.choice(units, len(units), replace=True)
    return s if idx is None else np.concatenate([idx[g] for g in s])


def mcnemar_exact(y, pred_a, pred_b):
    """Exact (binomial) McNemar test on paired correctness."""
    y, a, b = map(np.asarray, (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if n01 + n10 == 0 else stats.binomtest(n01, n01 + n10, 0.5).pvalue
    return dict(a_only_correct=n01, b_only_correct=n10, p=float(p))


def seed_ttest(scores_a, scores_b):
    """Seed-level paired t-test (secondary information only)."""
    t = stats.ttest_rel(scores_a, scores_b)
    return dict(mean_a=float(np.mean(scores_a)), sd_a=float(np.std(scores_a, ddof=1)),
                mean_b=float(np.mean(scores_b)), sd_b=float(np.std(scores_b, ddof=1)), t=float(t.statistic), p=float(t.pvalue))


# =====================================================================================================
# Training and evaluation
# Training with inner-validation model selection, inference and task-level evaluation (Sections 2.9-2.10).
# =====================================================================================================
# Five predefined seeds and the default training configuration. Replace these values with the ones in
# Supplementary Table S1 when reproducing the manuscript exactly.
SEEDS = (11, 22, 33, 44, 55)
DEFAULT_TRAIN = dict(epochs=40, batch=8, lr=1e-4, encoder_lr_mult=0.1, weight_decay=0.05, warmup_epochs=2,
                     patience=8, grad_clip=1.0, amp=True, workers=4, seed=SEEDS[0], eval_batch=16,
                     seg_weight=0.5, att_weight=0.1, tta=0)
# seg_weight / att_weight: mask supervision, used only by models with a segmentation head (BiFAM-Lite)
# tta: 1 = average the prediction of the slice and its horizontal flip at inference


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _img_size(model):
    return getattr(model, "cfg", {}).get("img_size", 512)


def _channels(model):
    return getattr(model, "cfg", {}).get("in_channels", 1)


def _mask_supervised(model):
    return bool(getattr(model, "cfg", {}).get("seg_head", False))


def _loader(root, df, train, cfg, with_mask=False, size=512, channels=1):
    return DataLoader(SliceDataset(root, df, train=train, size=size, with_mask=with_mask, channels=channels),
                      batch_size=cfg["batch"] if train else cfg["eval_batch"],
                      shuffle=train, drop_last=train and len(df) > cfg["batch"], num_workers=cfg["workers"],
                      pin_memory=torch.cuda.is_available(), persistent_workers=cfg["workers"] > 0)


def loss_fn(logits, y, num_outputs):
    """Eq. (16) cross-entropy for K classes; Eq. (17) binary cross-entropy for HGG/LGG. No class weighting."""
    if num_outputs == 1:
        return F.binary_cross_entropy_with_logits(logits.squeeze(1), y.float())
    return F.cross_entropy(logits, y.long())


def to_prob(logits, num_outputs):
    return torch.sigmoid(logits.squeeze(1)) if num_outputs == 1 else torch.softmax(logits, 1)


def _soft_dice(p, m, eps=1.0):
    inter = (p * m).flatten(1).sum(1)
    return 1 - (2 * inter + eps) / (p.flatten(1).sum(1) + m.flatten(1).sum(1) + eps)


def mask_losses(out, mask, has_mask):
    """Mask supervision: (1) BCE + soft Dice of the segmentation head; (2) attention consistency - soft Dice between
    the per-slice min-max normalised BiFAM map of the last decoder level and the tumor mask (slices with a tumor)."""
    sel = has_mask > 0.5
    zero = out["logits"].sum() * 0
    if not sel.any():
        return zero, zero
    m = mask[sel]
    seg = out["seg"][sel].float()
    l_seg = F.binary_cross_entropy_with_logits(seg, m) + _soft_dice(torch.sigmoid(seg), m).mean()
    l_att = zero
    pos = m.flatten(1).sum(1) > 0
    if "fout" in out and pos.any():
        a = out["fout"][-1][sel][pos].float().mean(1, keepdim=True)
        a = F.interpolate(a, size=m.shape[-2:], mode="bilinear", align_corners=False)
        lo = a.flatten(1).min(1)[0][:, None, None, None]; hi = a.flatten(1).max(1)[0][:, None, None, None]
        a = (a - lo) / (hi - lo).clamp_min(1e-6)
        l_att = _soft_dice(a, m[pos]).mean()
    return l_seg, l_att


@torch.inference_mode()
def predict(model, root, df, device, cfg=None, features=False):
    """Returns class probabilities ((N, K), or (N,) HGG probability) and optionally class-token features."""
    cfg = {**DEFAULT_TRAIN, **(cfg or {})}
    model.eval(); probs, feats = [], []
    for b in _loader(root, df, False, cfg, size=_img_size(model), channels=_channels(model)):
        x = b["x"].to(device, non_blocking=True)
        views = [x, torch.flip(x, (-1,))] if cfg.get("tta") else [x]      # horizontal flip only (anatomy-preserving)
        pv, fv = [], []
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"] and device.type == "cuda"):
                out = model(v)
            lg = out["logits"].float()
            pv.append(to_prob(lg, lg.shape[1])); fv.append(out["feat"].float())
        probs.append(torch.stack(pv).mean(0).cpu())
        if features:
            feats.append(torch.stack(fv).mean(0).cpu())
    p = torch.cat(probs).numpy()
    return (p, torch.cat(feats).numpy()) if features else p


def evaluate(df, prob, classes, unit="image", n_boot=2000, seed=0):
    """Evaluate at the level of the partition.

    unit='image'   : every slice is a sample; bootstrap resamples slices (descriptive intervals)
    unit='patient' : every slice is a sample; bootstrap resamples patients (df.group)
    unit='case'    : slices are aggregated per case (Eq. 19); every case counts once
    """
    y = df.label.values.astype(int)
    if unit == "case":
        ids, pred, score = aggregate_cases(df.group.values, prob)
        yc = df.groupby("group").label.first().loc[ids].values.astype(int)
        r = report(yc, pred, score, classes)
        r["macro_f1_ci"] = bootstrap_ci(yc, pred, 2, None, n_boot, seed)
        r["cases"] = dict(ids=ids.tolist(), y=yc.tolist(), pred=pred.tolist(), score=score.tolist())
        return r
    pred = prob.argmax(1) if prob.ndim == 2 else (prob >= 0.5).astype(int)
    r = report(y, pred, prob, classes)
    r["macro_f1_ci"] = bootstrap_ci(y, pred, len(classes), df.group.values if unit == "patient" else None, n_boot, seed)
    return r


def _val_score(model, root, val_df, device, cfg, num_outputs, unit):
    p = predict(model, root, val_df, device, cfg)
    if unit == "case":
        ids, pred, _ = aggregate_cases(val_df.group.values, p)
        return macro_f1(val_df.groupby("group").label.first().loc[ids].values.astype(int), pred, 2)
    pred = p.argmax(1) if p.ndim == 2 else (p >= 0.5).astype(int)
    return macro_f1(val_df.label.values.astype(int), pred, max(num_outputs, 2))


def train(root, train_df, val_df, model_name="bifamnet", model_cfg=None, train_cfg=None, unit="image", out=None,
          device=None, log=print):
    """Train end to end; select the checkpoint with the best inner-validation macro F1 (early stopping).

    Returns (best model in eval mode, history dict). Saves {out}.pt if `out` is given.
    """
    cfg = {**DEFAULT_TRAIN, **(train_cfg or {})}; mcfg = dict(model_cfg or {})
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(cfg["seed"])
    model = build_model(model_name, **mcfg).to(device)
    k = mcfg.get("num_outputs", 4)
    enc = set(map(id, model.encoder_parameters())) if hasattr(model, "encoder_parameters") else set()
    groups = [dict(params=[p for p in model.parameters() if id(p) in enc], lr=cfg["lr"] * cfg["encoder_lr_mult"]),
              dict(params=[p for p in model.parameters() if id(p) not in enc], lr=cfg["lr"])]
    groups = [g for g in groups if g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])
    masked = _mask_supervised(model)
    dl = _loader(root, train_df, True, cfg, with_mask=masked, size=_img_size(model), channels=_channels(model))
    total, warm = cfg["epochs"] * len(dl), cfg["warmup_epochs"] * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")
    best, best_state, bad, hist = -1.0, None, 0, []
    log(f"[train] {model_name} params={count_parameters(model) / 1e6:.2f}M train={len(train_df)} val={len(val_df)} "
        f"device={device} cfg={json.dumps({k_: v for k_, v in cfg.items() if k_ != 'workers'})}")
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time(); run = 0.0
        for b in dl:
            x, y = b["x"].to(device, non_blocking=True), b["y"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                o = model(x, return_maps=masked and cfg["att_weight"] > 0)
                loss = loss_fn(o["logits"].float(), y, k)
                if masked:
                    l_seg, l_att = mask_losses(o, b["mask"].to(device), b["has_mask"].to(device))
                    loss = loss + cfg["seg_weight"] * l_seg + cfg["att_weight"] * l_att
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
        f1 = _val_score(model, root, val_df, device, cfg, k, unit)
        hist.append(dict(epoch=ep + 1, loss=run / max(len(dl), 1), val_macro_f1=f1, sec=time.time() - t0))
        log(f"[train] epoch {ep + 1:3d} loss {hist[-1]['loss']:.4f} val macro F1 {f1:.4f} ({hist[-1]['sec']:.0f}s)")
        if f1 > best:
            best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"[train] early stop at epoch {ep + 1}"); break
    model.load_state_dict(best_state); model.eval()
    info = dict(best_val_macro_f1=best, history=hist)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{out}.pt.tmp"                    # write, then rename: a disconnect never leaves a half-written checkpoint
        torch.save(dict(model=model.state_dict(), model_name=model_name, model_cfg=mcfg, train_cfg=cfg, **info), tmp)
        os.replace(tmp, f"{out}.pt")
        to_json(info, f"{out}_history.json")
    return model, info


def to_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=conv)


# =====================================================================================================
# Analyses
# Localization against expert masks, t-SNE, source-separability probe and cost profiling (Sections 3.6-3.7).
#
# Localization (BraTS 2015, Supplementary S6)
#   maps     BiFAM  : channel mean of F_out at the third decoder level (128 x 128)
#            AG     : attention-gate coefficients alpha at the same level
#            GradCAM: gradient-weighted class activation of the final decoder block for the predicted class
#   each map is resized to 512 x 512 and min-max normalised per slice, then thresholded at the 90th percentile
#   of the activation values of the *training* cases of the same fold (no test case sets its own threshold).
#   scores   Dice, IoU and coverage on slices with a non-empty mask; pointing game (arg-max inside the mask).
#   nulls    uniform map (the whole brain region predicted) and a uniformly random point inside the brain.
# =====================================================================================================
MAPS = ("bifam", "ag", "gradcam")


def _norm(m):
    lo = m.flatten(1).min(1)[0][:, None, None]; hi = m.flatten(1).max(1)[0][:, None, None]
    return (m - lo) / (hi - lo).clamp_min(1e-8)


def compute_maps(model, x, level=2, size=None):
    """Returns {'bifam', 'ag', 'gradcam'} tensors of shape (B, size, size) in [0, 1]."""
    model.eval()
    with torch.enable_grad():
        out = model(x, return_maps=True)
        dec = out["dec"]; dec.retain_grad()
        lg = out["logits"]
        target = (torch.where(lg[:, 0] >= 0, lg[:, 0], -lg[:, 0]) if lg.shape[1] == 1
                  else lg.gather(1, lg.argmax(1, keepdim=True))[:, 0])
        target.sum().backward()
    size = size or x.shape[-1]
    w = dec.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * dec).sum(1))
    up = lambda m: F.interpolate(m[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    maps = dict(bifam=up(out["fout"][level].mean(1)), gradcam=up(cam))
    if out["alpha"][level] is not None:
        maps["ag"] = up(out["alpha"][level][:, 0])
    return {k: _norm(v.detach()) for k, v in maps.items()}


def _batches(root, df, batch, workers, size=512, channels=1):
    return DataLoader(SliceDataset(root, df, train=False, size=size, with_mask=True, channels=channels),
                      batch_size=batch, num_workers=workers)


def fold_thresholds(model, root, train_df, device, q=90, max_slices=2000, batch=8, workers=4, level=2, seed=0):
    """90th percentile of pooled map activations over (a random subset of) the fold's training slices."""
    sub = train_df.sample(min(max_slices, len(train_df)), random_state=seed)
    vals = {k: [] for k in MAPS}
    rng = np.random.default_rng(seed)
    for b in _batches(root, sub, batch, workers, model.cfg["img_size"], model.cfg.get("in_channels", 1)):
        for k, m in compute_maps(model, b["x"].to(device), level).items():
            v = m.flatten().cpu().numpy()
            vals[k].append(rng.choice(v, min(len(v), 20000), replace=False))
    return {k: float(np.percentile(np.concatenate(v), q)) for k, v in vals.items() if v}


def localization_scores(model, root, test_df, thresholds, device, batch=8, workers=4, level=2):
    """Per-slice scores for every map and the two null references."""
    rows = []
    for b in _batches(root, test_df, batch, workers, model.cfg["img_size"], model.cfg.get("in_channels", 1)):
        x = b["x"].to(device); mask = b["mask"][:, 0].bool().to(device)
        brain = x[:, 0] > x[:, 0].flatten(1).min(1)[0][:, None, None] + 1e-3
        maps = compute_maps(model, x, level)
        for i in range(len(x)):
            m = mask[i]; n = int(m.sum()); r = dict(idx=int(b["idx"][i]), mask_pixels=n)
            for k, v in maps.items():
                pr = v[i] >= thresholds[k]
                inter = int((pr & m).sum()); s = int(pr.sum())
                r[f"{k}_dice"] = 2 * inter / (s + n) if n else np.nan
                r[f"{k}_iou"] = inter / (s + n - inter) if n else np.nan
                r[f"{k}_coverage"] = inter / n if n else np.nan
                r[f"{k}_pointing"] = float(m.flatten()[v[i].flatten().argmax()]) if n else np.nan
            br = brain[i]; inter = int((br & m).sum()); s = int(br.sum())
            r["uniform_dice"] = 2 * inter / (s + n) if n else np.nan
            r["uniform_iou"] = inter / (s + n - inter) if n else np.nan
            r["random_pointing"] = inter / max(s, 1) if n else np.nan
            rows.append(r)
    return rows


def summarize_localization(rows):
    import pandas as pd
    d = pd.DataFrame(rows); d = d[d.mask_pixels > 0]
    cols = [c for c in d.columns if c.endswith(("_dice", "_iou", "_coverage", "_pointing"))]
    return dict(n_slices=len(d), **{c: float(d[c].mean()) for c in cols})


# ------------------------------------------------------------------ feature space and source probe
def tsne(features, perplexity=30, iters=1000, seed=0):
    from sklearn.manifold import TSNE
    try:
        t = TSNE(2, perplexity=perplexity, max_iter=iters, init="pca", random_state=seed)
    except TypeError:                                                  # scikit-learn < 1.5
        t = TSNE(2, perplexity=perplexity, n_iter=iters, init="pca", random_state=seed)
    return t.fit_transform(features)


def plot_tsne(emb, y, pred, classes, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, lab, title in ((ax[0], y, "True label"), (ax[1], pred, "Predicted label")):
        for i, c in enumerate(classes):
            s = lab == i
            a.scatter(emb[s, 0], emb[s, 1], s=6, label=f"{i}, {c}")
        a.set_title(title); a.set_xticks([]); a.set_yticks([])
    ax[1].legend(markerscale=3, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def source_probe(features, source, folds=5, seed=0):
    """Cross-validated logistic-regression probe predicting the source collection from frozen features.

    Because class and source coincide in the four-class set, a high accuracy shows separability only;
    it does not measure the size of an acquisition effect.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    acc = cross_val_score(clf, features, source, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
    return dict(accuracy=float(acc.mean()), sd=float(acc.std()), folds=acc.tolist())


# ------------------------------------------------------------------ cost
def profile(model, device, size=512, warmup=100, runs=1000):
    """Parameters, GFLOPs (multiply-accumulate counted as two FLOPs / 2 = MACs reported as GFLOPs) and latency."""
    from torch.utils.flop_counter import FlopCounterMode
    model.eval().to(device); x = torch.randn(1, 1, size, size, device=device)
    with FlopCounterMode(display=False) as fc, torch.inference_mode():
        model(x)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / runs * 1000
    return dict(params_m=sum(p.numel() for p in model.parameters()) / 1e6, gflops=fc.get_total_flops() / 2e9,
                ms_per_slice=ms, peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None)


# =====================================================================================================
# MIL, calibration, conformal prediction, ensembles
# Case-level attention MIL, calibration / conformal prediction / selective accuracy, and seed ensembles.
#
# MIL (BraTS HGG/LGG)
#   The slice model of each fold embeds every slice of a case (class-token feature). A gated-attention MIL head
#   (Ilse et al., 2018) learns which slices matter and outputs one case probability, replacing the majority vote.
#   It is trained on the fold's training cases (features taken under training augmentation to limit the optimism
#   of in-sample features), selected on the fold's inner-validation cases and applied once to the test cases.
#
# Calibration
#   Temperature scaling fitted on the inner-validation predictions; expected calibration error (15 bins);
#   split-conformal prediction sets (LAC score) with guaranteed marginal coverage 1 - alpha; accuracy at
#   fixed coverage (selective prediction: the least confident cases are referred to a radiologist).
# =====================================================================================================
# ------------------------------------------------------------------ MIL
class GatedAttentionMIL(nn.Module):
    """a_i = softmax(w^T (tanh(V h_i) * sigmoid(U h_i))); z = sum a_i h_i; logit = W z."""

    def __init__(self, in_dim, hid=128, dropout=0.25):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hid), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.v, self.u, self.w = nn.Linear(hid, hid), nn.Linear(hid, hid), nn.Linear(hid, 1)
        self.fc = nn.Linear(hid, 1)

    def forward(self, bag):                              # bag: (N, D) slices of one case
        h = self.proj(bag)
        a = torch.softmax(self.w(torch.tanh(self.v(h)) * torch.sigmoid(self.u(h))).squeeze(-1), 0)
        return self.fc((a[:, None] * h).sum(0)).squeeze(-1), a


@torch.no_grad()
def case_features(model, root, df, device, augment=False, passes=1, batch=16, workers=2):
    """{case id: (list of slice paths, (passes, N, D) features)} using the model's class-token feature."""
    model.eval(); out = {}
    df = df.reset_index(drop=True)
    feats = []
    for _ in range(passes):
        dl = DataLoader(SliceDataset(root, df, train=augment, size=_img_size(model), channels=_channels(model)),
                        batch_size=batch, num_workers=workers)
        f = []
        for b in dl:
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                f.append(model(b["x"].to(device))["feat"].float().cpu())
        feats.append(torch.cat(f))
    feats = torch.stack(feats)                            # (passes, N_slices, D)
    for g, idx in df.groupby("group").indices.items():
        out[g] = (df.path.values[idx].tolist(), feats[:, idx])
    return out


def train_mil(train_bags, val_bags, labels, seed=0, epochs=150, lr=1e-3, wd=1e-2, patience=25, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    d = next(iter(train_bags.values()))[1].shape[-1]
    mil = GatedAttentionMIL(d).to(device)
    opt = torch.optim.AdamW(mil.parameters(), lr=lr, weight_decay=wd)
    keys = list(train_bags); best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        mil.train(); np.random.shuffle(keys)
        for g in keys:
            f = train_bags[g][1]; f = f[np.random.randint(len(f))].to(device)        # one augmentation pass
            keep = torch.rand(len(f)) > 0.2                                            # slice dropout
            f = f[keep.to(f.device)] if keep.any() else f
            logit, _ = mil(f)
            loss = F.binary_cross_entropy_with_logits(logit, torch.tensor(float(labels[g]), device=device))
            opt.zero_grad(); loss.backward(); opt.step()
        p = predict_mil(mil, val_bags, device)
        yv = np.array([labels[g] for g in p]); pv = np.array(list(p.values()))
        f1 = report(yv, (pv >= 0.5).astype(int), pv, ["LGG", "HGG"])["macro_f1"] if len(set(yv)) > 1 else float((yv == (pv >= .5)).mean())
        score = f1 - 1e-3 * F.binary_cross_entropy(torch.tensor(pv).float().clamp(1e-6, 1 - 1e-6), torch.tensor(yv).float()).item()
        if score > best:
            best, bad, best_state = score, 0, {k: v.clone() for k, v in mil.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    mil.load_state_dict(best_state); mil.eval()
    return mil


@torch.no_grad()
def predict_mil(mil, bags, device="cpu", return_attention=False):
    mil.eval(); probs, att = {}, {}
    for g, (paths, f) in bags.items():
        logit, a = mil(f.mean(0).to(device))
        probs[g] = float(torch.sigmoid(logit)); att[g] = (paths, a.cpu().numpy())
    return (probs, att) if return_attention else probs


def run_mil(data, run, out, seed=11, device=None, workers=2):
    """MIL case aggregation for every fold of a BraTS run; pooled out-of-fold evaluation vs. majority vote."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = load_index(data, "brats").set_index("path", drop=False)
    labels = df.groupby("group").label.first().to_dict()
    case_rows, att_rows = [], []
    for f, sp in enumerate(load_split(data, "brats_cv")):
        ck = Path(run) / f"seed{seed}_fold{f}.pt"
        if not ck.exists():
            print(f"[mil] missing {ck}; skipped"); continue
        model, _ = load_checkpoint(ck, dev)
        tr = case_features(model, data, df.loc[sp["train"]], dev, augment=True, passes=2, workers=workers)
        va = case_features(model, data, df.loc[sp["val"]], dev, workers=workers)
        te = case_features(model, data, df.loc[sp["test"]], dev, workers=workers)
        mil = train_mil(tr, va, labels, seed=seed, device=dev)
        probs, att = predict_mil(mil, te, dev, return_attention=True)
        for g, p in probs.items():
            case_rows.append(dict(case=g, fold=f, label=labels[g], mil_prob=p))
            for path, w in zip(*att[g]):
                att_rows.append(dict(case=g, fold=f, path=path, attention=float(w)))
        print(f"[mil] fold {f}: {len(probs)} test cases")
    cases = pd.DataFrame(case_rows)
    cases.to_csv(out / f"mil_cases_seed{seed}.csv", index=False)
    pd.DataFrame(att_rows).to_csv(out / f"mil_slice_attention_seed{seed}.csv", index=False)
    y = cases.label.values.astype(int); pm = (cases.mil_prob.values >= 0.5).astype(int)
    r = report(y, pm, cases.mil_prob.values, ["LGG", "HGG"]); r["macro_f1_ci"] = bootstrap_ci(y, pm, 2)
    res = dict(mil=r)
    pf = Path(run) / f"predictions_seed{seed}.csv"
    if pf.exists():                                         # paired comparison with the majority vote
        p = pd.read_csv(pf, keep_default_na=False)
        ids, pv, _ = aggregate_cases(p.group.values, p.prob.values)
        vote = dict(zip(ids, pv)); common = [g for g in cases.case if g in vote]
        c = cases.set_index("case").loc[common]
        yv = c.label.values.astype(int); a = (c.mil_prob.values >= 0.5).astype(int); b = np.array([vote[g] for g in common])
        res["majority_vote"] = report(yv, b, None, ["LGG", "HGG"])
        res["mil_vs_vote"] = dict(paired_bootstrap=paired_bootstrap(yv, a, b, 2), mcnemar=mcnemar_exact(yv, a, b))
    to_json(res, out / f"mil_seed{seed}.json")
    msg = f"[mil] MIL: accuracy {100 * r['accuracy']:.2f} macro F1 {100 * r['macro_f1']:.2f} AUC {100 * r.get('auc', float('nan')):.2f}"
    if "majority_vote" in res:
        msg += f" | majority vote: accuracy {100 * res['majority_vote']['accuracy']:.2f} macro F1 {100 * res['majority_vote']['macro_f1']:.2f}"
    print(msg)
    return res


# ------------------------------------------------------------------ calibration and conformal prediction
def _probs(df):
    pc = [c for c in df.columns if c.startswith("prob_")]
    if pc:
        return df[pc].values.astype(float)
    p = df["prob"].values.astype(float)
    return np.stack([1 - p, p], 1)


def _case_level(df):
    g = df.groupby("group")
    p = g.prob.mean().values
    return pd.DataFrame(dict(group=g.prob.mean().index, label=g.label.first().values, prob=p))


def fit_temperature(p, y):
    logit = torch.log(torch.tensor(p).clamp_min(1e-8)).float(); yt = torch.tensor(y).long()
    t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logit / torch.exp(t), yt); loss.backward(); return loss
    opt.step(closure)
    return float(torch.exp(t))


def apply_temperature(p, T):
    return torch.softmax(torch.log(torch.tensor(p).clamp_min(1e-8)) / T, 1).numpy()


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1); acc = pred == y; e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            e += s.mean() * abs(acc[s].mean() - conf[s].mean())
    return float(e)


def conformal(p_cal, y_cal, p_test, y_test, alpha=0.05):
    """Split conformal prediction with the LAC score s = 1 - p(true class)."""
    n = len(y_cal); s = 1 - p_cal[np.arange(n), y_cal]
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = p_test >= 1 - q
    cover = sets[np.arange(len(y_test)), y_test]
    size = sets.sum(1); single = size == 1
    return dict(alpha=alpha, qhat=float(q), coverage=float(cover.mean()), mean_set_size=float(size.mean()),
                singleton_fraction=float(single.mean()),
                singleton_accuracy=float((p_test[single].argmax(1) == y_test[single]).mean()) if single.any() else None)


def selective(p, y, coverages=(1.0, 0.95, 0.9, 0.8)):
    order = np.argsort(-p.max(1)); correct = (p.argmax(1) == y)[order]
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        out[f"accuracy_at_{int(c * 100)}pct_coverage"] = float(correct[:k].mean())
    curve = np.cumsum(correct) / np.arange(1, len(y) + 1)
    return out, curve


def run_calibration(run, task, seed=11, alpha=0.05, out=None):
    """Temperature scaling, ECE, conformal sets and selective accuracy from the saved val/test predictions."""
    run = Path(run); out = Path(out or run / "calibration"); out.mkdir(parents=True, exist_ok=True)
    unit = TASKS[task][3]
    te = pd.read_csv(run / f"predictions_seed{seed}.csv", keep_default_na=False)
    va = pd.read_csv(run / f"val_predictions_seed{seed}.csv", keep_default_na=False)
    if unit == "case":                                     # case-level: mean slice probability per case (per fold)
        te = pd.concat([_case_level(d) for _, d in te.groupby("fold")]); va = pd.concat([_case_level(d) for _, d in va.groupby("fold")])
    pv, yv, pt, yt = _probs(va), va.label.values.astype(int), _probs(te), te.label.values.astype(int)
    T = fit_temperature(pv, yv); pt_T = apply_temperature(pt, T); pv_T = apply_temperature(pv, T)
    sel, curve = selective(pt_T, yt)
    res = dict(task=task, unit=unit, n_cal=len(yv), n_test=len(yt), temperature=T,
               ece_before=ece(pt, yt), ece_after=ece(pt_T, yt), selective=sel,
               conformal={str(a): conformal(pv_T, yv, pt_T, yt, a) for a in sorted({alpha, 0.1, 0.05, 0.01})})
    to_json(res, out / f"calibration_seed{seed}.json")
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(np.arange(1, len(curve) + 1) / len(curve), curve); ax[0].set_xlabel("coverage (most confident first)")
    ax[0].set_ylabel("accuracy"); ax[0].set_title("accuracy vs. coverage"); ax[0].set_ylim(min(curve.min(), 0.9) - 0.01, 1.005)
    conf, acc = pt_T.max(1), pt_T.argmax(1) == yt; edges = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            mids.append(conf[s].mean()); accs.append(acc[s].mean())
    ax[1].plot([0, 1], [0, 1], "k--", lw=0.8); ax[1].plot(mids, accs, "o-")
    ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title(f"reliability (T = {T:.2f})")
    fig.tight_layout(); fig.savefig(out / f"calibration_seed{seed}.png", dpi=150); plt.close(fig)
    c = res["conformal"][str(alpha)]
    print(f"[calibrate] {task}: T={T:.2f} ECE {res['ece_before']:.3f}->{res['ece_after']:.3f}; "
          f"{int((1 - alpha) * 100)}% conformal coverage {c['coverage']:.3f}, mean set size {c['mean_set_size']:.2f}; "
          + ", ".join(f"{k} {v:.4f}" for k, v in sel.items()))
    return res


# ------------------------------------------------------------------ seed ensembles
def run_ensemble(run, task, seeds, out=None):
    """Average the probabilities of all seeds (the partitions are identical across seeds) and evaluate."""
    run = Path(run); classes, unit = TASKS[task][1], TASKS[task][3]
    frames = [pd.read_csv(run / f"predictions_seed{s}.csv", keep_default_na=False) for s in seeds
              if (run / f"predictions_seed{s}.csv").exists()]
    if len(frames) < 2:
        print(f"[ensemble] {run}: fewer than two seeds - skipped"); return None
    pc = [c for c in frames[0].columns if c.startswith("prob")]
    ens = frames[0][["path", "group", "label"]].copy()
    for c in pc:
        ens[c] = np.mean([f.set_index("path").loc[ens.path, c].values for f in frames], 0)
    prob = ens["prob"].values if pc == ["prob"] else ens[pc].values
    r = evaluate(ens, prob, classes, unit)
    ens.to_csv(run / "predictions_ensemble.csv", index=False)
    to_json(dict(seeds=len(frames), **r), Path(out or run / "metrics_ensemble.json"))
    print(f"[ensemble] {task}: {len(frames)} seeds -> accuracy {100 * r['accuracy']:.2f} macro F1 {100 * r['macro_f1']:.2f}")
    return r


# =====================================================================================================
# Figures: sample slices, preprocessing stages, augmentation, attention maps, errors, curves, fusion plane
# =====================================================================================================
def _plt():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _png(root, rel, size=None):
    a = np.asarray(Image.open(Path(root) / rel).convert("L"))
    return resize(a, size) if size else a


def _class_names(task):
    return TASKS[task][1] if task in TASKS else FOUR_CLASSES


def raw_slice(raw, row):
    """Original (pre-processing) image of an index row, looked up in the raw download; None if unavailable."""
    if raw is None:
        return None
    raw = Path(raw)
    try:
        if row.source == "figshare":
            import h5py
            n = int(Path(row.path).stem.split("_")[0])
            f = next((p for p in (raw / "figshare").rglob(f"{n}.mat")), None)
            if f is None:
                return None
            with h5py.File(f, "r") as h:
                return np.asarray(h["cjdata"]["image"]).T.astype(np.float32)
        if row.source == "br35h" and row.get("original", ""):
            f = next((p for p in (raw / "br35h").rglob(row.original) if p.parent.name.lower() == "no"), None)
            return None if f is None else np.asarray(Image.open(f).convert("L"), np.float32)
        if row.source == "brats2015":
            grade, case = row.case.split("_", 1)
            f = next((raw / "brats2015" / grade / case).rglob("*T1c*.mha"), None)
            return None if f is None else _read_mha(f)[int(row.slice)].astype(np.float32)
    except Exception as e:
        print(f"[viz] raw image unavailable for {row.path}: {e}")
    return None


def _model_input(root, rel, size):
    x = torch.from_numpy(_png(root, rel, size)).float()[None] / 255
    return (x - x.mean()) / (x.std() + 1e-6)


def _row_input(root, row, size, channels):
    x = load_slice(root, row, size, channels)
    return (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-6)


def fig_samples(data, out, n_per_class=8, seed=0):
    """Figure 2: sample slices of every class of every prepared collection."""
    plt = _plt(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for index_task, classes, name in (("fourclass", FOUR_CLASSES, "fourclass"), ("brats", BINARY_CLASSES, "brats2015")):
        try:
            df = load_index(data, index_task)
        except FileNotFoundError:
            continue
        labels = sorted(df.label.unique())
        fig, ax = plt.subplots(len(labels), n_per_class, figsize=(1.6 * n_per_class, 1.75 * len(labels)), squeeze=False)
        for r, lab in enumerate(labels):
            sub = df[df.label == lab].sample(min(n_per_class, (df.label == lab).sum()), random_state=seed)
            for c in range(n_per_class):
                a = ax[r, c]; a.axis("off")
                if c < len(sub):
                    a.imshow(_png(data, sub.path.iloc[c], 256), cmap="gray")
                    if index_task == "brats" and sub["mask"].iloc[c]:
                        a.contour(_png(data, sub["mask"].iloc[c], 256) > 127, levels=[0.5], colors="lime", linewidths=0.6)
            ax[r, 0].set_title(classes[lab], loc="left", fontsize=9)
        fig.suptitle(f"Sample slices - {name}" + (" (green: whole-tumor mask)" if index_task == "brats" else ""), fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"samples_{name}.png", dpi=150); plt.close(fig)


def fig_preprocessing(data, out, raw=None, per_class=2, size=512, seed=0):
    """Raw slice -> percentile clipping + resizing (stored PNG) -> training augmentation -> z-scored model input."""
    plt = _plt(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    frames = []
    for index_task in ("fourclass", "brats"):
        try:
            frames.append(load_index(data, index_task))
        except FileNotFoundError:
            pass
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    rows = pd.concat([g.sample(min(per_class, len(g)), random_state=seed) for _, g in df.groupby(["source", "label"])])
    ds = SliceDataset(data, rows, train=True, size=size)
    fig, ax = plt.subplots(len(rows), 5, figsize=(15, 3 * len(rows)), squeeze=False)
    for i in range(len(rows)):
        r = rows.iloc[i]; rw = raw_slice(raw, r); stored = _png(data, r.path)
        torch.manual_seed(seed + i); aug = ds[i]["x"][0].numpy()
        inp = _model_input(data, r.path, size)[0].numpy()
        name = (FOUR_CLASSES + BINARY_CLASSES)[r.label if r.source != "brats2015" else 4 + r.label]
        panels = [(rw, f"raw {'' if rw is None else rw.shape}"), (stored, f"clipped + resized {stored.shape}"),
                  (aug, "augmented (training only)"), (inp, "model input (z-score)")]
        for j, (img, title) in enumerate(panels):
            a = ax[i, j]; a.axis("off")
            if img is None:
                a.text(0.5, 0.5, "raw file not given\n(--raw)", ha="center", va="center", fontsize=9); continue
            im = a.imshow(img, cmap="gray"); a.set_title(title, fontsize=8)
            if j == 3:
                fig.colorbar(im, ax=a, fraction=0.046)
        ax[i, 0].set_title(f"{r.source}: {name}\nraw {'' if rw is None else rw.shape}", fontsize=8)
        a = ax[i, 4]
        if rw is not None:
            a.hist(rw[rw > 0].ravel() / max(rw.max(), 1), 60, alpha=0.5, label="raw (scaled)", density=True)
        a.hist(inp.ravel(), 60, alpha=0.5, label="model input", density=True)
        a.legend(fontsize=7); a.set_title("intensity histogram", fontsize=8); a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "preprocessing_stages.png", dpi=110); plt.close(fig)


def fig_augmentation(data, out, n_slices=4, n_aug=8, size=512, seed=0):
    """Several random training augmentations (no vertical flip) of a few slices."""
    plt = _plt(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    try:
        df = load_index(data, "fourclass")
    except FileNotFoundError:
        df = load_index(data, "brats")
    rows = df.groupby("label").sample(1, random_state=seed).head(n_slices)
    ds = SliceDataset(data, rows, train=True, size=size)
    fig, ax = plt.subplots(len(rows), n_aug + 1, figsize=(1.8 * (n_aug + 1), 1.9 * len(rows)), squeeze=False)
    for i in range(len(rows)):
        ax[i, 0].imshow(_model_input(data, rows.path.iloc[i], size)[0], cmap="gray"); ax[i, 0].set_title("original", fontsize=8)
        for j in range(n_aug):
            ax[i, j + 1].imshow(ds[i]["x"][0], cmap="gray"); ax[i, j + 1].set_title(f"aug {j + 1}", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(out / "augmentation_examples.png", dpi=120); plt.close(fig)


def all_maps(model, x):
    """BiFAM output (channel mean) and attention-gate maps at every decoder level plus Grad-CAM; (B, H, W) in [0, 1]."""
    size = x.shape[-1]
    with torch.no_grad():
        out = model(x, return_maps=True)
    up = lambda m: _norm(F.interpolate(m[:, None].float(), size=(size, size), mode="bilinear", align_corners=False)[:, 0])
    maps = {}
    for lvl, res in enumerate(("32", "64", "128")):
        maps[f"BiFAM {res}²"] = up(out["fout"][lvl].mean(1))
        if out["alpha"][lvl] is not None:
            maps[f"AG α {res}²"] = up(out["alpha"][lvl][:, 0])
    maps["Grad-CAM"] = compute_maps(model, x)["gradcam"]
    prob = to_prob(out["logits"].float(), out["logits"].shape[1])
    return {k: v.cpu().numpy() for k, v in maps.items()}, prob.cpu().numpy()


def _overlay(ax, img, heat, mask=None, alpha=0.45):
    ax.imshow(img, cmap="gray"); ax.imshow(heat, cmap="jet", alpha=alpha, vmin=0, vmax=1)
    if mask is not None and mask.any():
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
    ax.axis("off")


def attention_gallery(data, model, rows, classes, out, tag, batch=8, device=None):
    """One PNG per slice (input, BiFAM/AG maps at three levels, Grad-CAM, overlay, mask) + a contact sheet."""
    plt = _plt(); out = Path(out); (out / tag).mkdir(parents=True, exist_ok=True)
    dev = next(model.parameters()).device; size = model.cfg["img_size"]; ch = model.cfg.get("in_channels", 1)
    disp = DISPLAY_CHANNEL if ch > 1 else 0
    rows = rows.reset_index(drop=True); sheet = []
    for s in range(0, len(rows), batch):
        chunk = rows.iloc[s:s + batch]
        x = torch.stack([_row_input(data, chunk.iloc[j], size, ch) for j in range(len(chunk))]).to(dev)
        maps, prob = all_maps(model, x)
        for i in range(len(chunk)):
            r = chunk.iloc[i]; img = x[i, disp].cpu().numpy()
            mask = (_png(data, r["mask"], size) > 127) if r.get("mask", "") else None
            if prob.ndim == 1:
                pred = int(prob[i] >= 0.5); p = prob[i] if pred else 1 - prob[i]
            else:
                pred = int(prob[i].argmax()); p = prob[i, pred]
            names = list(maps)
            fig, ax = plt.subplots(1, len(names) + 2, figsize=(2.2 * (len(names) + 2), 2.6))
            ax[0].imshow(img, cmap="gray"); ax[0].axis("off")
            if mask is not None and mask.any():
                ax[0].contour(mask, levels=[0.5], colors="lime", linewidths=0.8)
            ax[0].set_title(f"true {classes[int(r.label)]}\npred {classes[pred]} (p={p:.2f})", fontsize=8,
                            color="black" if pred == int(r.label) else "red")
            for j, n in enumerate(names):
                ax[j + 1].imshow(maps[n][i], cmap="jet", vmin=0, vmax=1); ax[j + 1].set_title(n, fontsize=8); ax[j + 1].axis("off")
            _overlay(ax[-1], img, maps["BiFAM 128²"][i], mask); ax[-1].set_title("BiFAM 128² overlay", fontsize=8)
            fig.tight_layout()
            fname = f"{s + i:04d}_{classes[int(r.label)]}_pred-{classes[pred]}.png"
            fig.savefig(out / tag / fname, dpi=110); plt.close(fig)
            sheet.append((img, maps["BiFAM 128²"][i], maps["Grad-CAM"][i], mask, classes[int(r.label)], classes[pred], p))
    # contact sheet: BiFAM overlay (top) and Grad-CAM overlay (bottom) for every slice
    cols = 8; nrow = math.ceil(len(sheet) / cols)
    fig, ax = plt.subplots(2 * nrow, cols, figsize=(2 * cols, 4.1 * nrow), squeeze=False)
    for a in ax.ravel():
        a.axis("off")
    for k, (img, bif, cam, mask, t, pr, p) in enumerate(sheet):
        rr, cc = 2 * (k // cols), k % cols
        _overlay(ax[rr, cc], img, bif, mask); ax[rr, cc].set_title(f"{t}->{pr} {p:.2f}", fontsize=7, color="black" if t == pr else "red")
        _overlay(ax[rr + 1, cc], img, cam, mask)
    fig.suptitle(f"{tag}: BiFAM 128² (odd rows) and Grad-CAM (even rows); green = expert mask", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{tag}_contact_sheet.png", dpi=110); plt.close(fig)
    print(f"[viz] {len(sheet)} attention figures -> {out / tag}")


def _pick(df, n, seed):
    """Up to n slices spread evenly over the classes."""
    k = max(1, n // max(df.label.nunique(), 1))
    return pd.concat([g.sample(min(k, len(g)), random_state=seed) for _, g in df.groupby("label")])


def fig_attention(data, runs, out, n=48, seed=SEEDS[0], device=None, main="bifamlite"):
    """Attention figures for the four-class, three-class and BraTS models (each on its own test slices only)."""
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); runs = Path(runs)
    for task in ("fourclass_image", "threeclass_patient", "threeclass_image"):
        ck = runs / task / main / f"seed{seed}.pt"
        if not ck.exists():
            continue
        model, _ = load_checkpoint(ck, dev)
        df = load_index(data, TASKS[task][0]).set_index("path", drop=False)
        test = df.loc[load_split(data, task)["test"]]
        attention_gallery(data, model, _pick(test, n, seed), TASKS[task][1], out, task)
    run = runs / "brats_cv" / main
    if (run / f"seed{seed}_fold0.pt").exists():
        df = load_index(data, "brats").set_index("path", drop=False)
        folds = load_split(data, "brats_cv"); per = max(1, n // len(folds))
        for f, sp in enumerate(folds):
            ck = run / f"seed{seed}_fold{f}.pt"
            if not ck.exists():
                continue
            model, _ = load_checkpoint(ck, dev)
            test = df.loc[sp["test"]]
            test = test[test.mask_pixels > 0] if "mask_pixels" in test and (test.mask_pixels > 0).any() else test
            attention_gallery(data, model, _pick(test, per, seed), BINARY_CLASSES, out, f"brats_cv_fold{f}")


def fig_misclassified(data, runs, out, seed=SEEDS[0], max_n=24, main="bifamlite"):
    """Every (up to max_n) misclassified test slice of the image-level runs, with its predicted probability."""
    plt = _plt(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for task in ("fourclass_image", "threeclass_patient", "threeclass_image"):
        f = Path(runs) / task / main / f"predictions_seed{seed}.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f, keep_default_na=False); pc = [c for c in p.columns if c.startswith("prob_")]
        p["pred"] = p[pc].values.argmax(1); p["p"] = p[pc].values.max(1)
        wrong = p[p.pred != p.label].head(max_n); classes = TASKS[task][1]
        if wrong.empty:
            continue
        cols = 6; nrow = math.ceil(len(wrong) / cols)
        fig, ax = plt.subplots(nrow, cols, figsize=(2.2 * cols, 2.4 * nrow), squeeze=False)
        for a in ax.ravel():
            a.axis("off")
        for k, r in enumerate(wrong.itertuples()):
            a = ax[k // cols, k % cols]; a.imshow(_png(data, r.path, 256), cmap="gray")
            a.set_title(f"{classes[r.label]} -> {classes[r.pred]}\np={r.p:.2f}", fontsize=7, color="red")
        fig.suptitle(f"{task}: misclassified test slices ({(p.pred != p.label).sum()} of {len(p)})", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"misclassified_{task}.png", dpi=120); plt.close(fig)


def fig_training_curves(runs, out, main="bifamlite"):
    """Training loss and inner-validation macro F1 per epoch for every main BiFAM-Net run."""
    plt = _plt(); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for task_dir in sorted(Path(runs).iterdir()) if Path(runs).exists() else []:
        hs = sorted((task_dir / main).glob("*_history.json"))
        if not hs:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        for h in hs:
            hist = json.load(open(h))["history"]
            e = [r["epoch"] for r in hist]
            ax[0].plot(e, [r["loss"] for r in hist], marker="o", label=h.stem.replace("_history", ""))
            ax[1].plot(e, [r["val_macro_f1"] for r in hist], marker="o")
        ax[0].set_title("training loss"); ax[1].set_title("inner-validation macro F1"); ax[1].set_ylim(0, 1.02)
        for a in ax:
            a.set_xlabel("epoch")
        ax[0].legend(fontsize=6, ncol=2); fig.suptitle(task_dir.name); fig.tight_layout()
        fig.savefig(out / f"training_curves_{task_dir.name}.png", dpi=130); plt.close(fig)


def fig_fusion_plane(runs, out, main="bifamlite"):
    """Figure 8: macro F1 (mean over seeds) against GFLOPs per slice for every fusion operator."""
    plt = _plt(); runs = Path(runs); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if not (runs / "profile.csv").exists():
        return
    prof = pd.read_csv(runs / "profile.csv").set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4)); drawn = False
    for a, task in zip(ax, ("fourclass_image", "brats_cv")):
        for f in FUSIONS:
            run = runs / task / (main if f == "bifam" else f"fusion_{f}") / "summary.json"
            key = main if f == "bifam" else f"{main}[{f}]"
            if run.exists() and key in prof.index:
                y = json.load(open(run))["mean"]["macro_f1"]; x = prof.loc[key, "gflops"]
                a.scatter(x, y, s=40, color="crimson" if f == "bifam" else "steelblue")
                a.annotate(f, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points"); drawn = True
        a.set_xlabel("GFLOPs per slice"); a.set_ylabel("macro F1"); a.set_title(task)
    if drawn:
        fig.tight_layout(); fig.savefig(out / "fusion_cost_accuracy.png", dpi=150)
    plt.close(fig)


def run_visualize(data, runs, out=None, raw=None, n=48, seed=SEEDS[0], device=None, main="bifamlite"):
    out = Path(out or Path(runs) / "figures")
    steps = [("samples", lambda: fig_samples(data, out)),
             ("preprocessing", lambda: fig_preprocessing(data, out, raw)),
             ("augmentation", lambda: fig_augmentation(data, out)),
             ("attention", lambda: fig_attention(data, runs, out / "attention", n, seed, device, main)),
             ("misclassified", lambda: fig_misclassified(data, runs, out, seed, main=main)),
             ("training curves", lambda: fig_training_curves(runs, out, main)),
             ("fusion plane", lambda: fig_fusion_plane(runs, out, main))]
    for name, fn in steps:
        try:
            fn(); print(f"[viz] {name} done")
        except Exception:
            import traceback; print(f"[viz] {name} failed:"); traceback.print_exc()
    n_png = len(list(out.rglob("*.png")))
    print(f"[viz] {n_png} figures in {out}")


def cmd_visualize(a):
    run_visualize(a.data, a.runs, a.out, a.raw, a.n, a.seed, a.device, a.model)


# =====================================================================================================
# Colab helpers: dataset download
# =====================================================================================================
def _run(cmd):
    print("$", cmd); return subprocess.call(cmd, shell=True)


UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
FIGSHARE_ARTICLE = 1512427           # Cheng et al., doi:10.6084/m9.figshare.1512427


def _figshare_ready(d):
    return any(p.stem.isdigit() for p in Path(d).rglob("*.mat")) if Path(d).exists() else False


def _unpack_zips(dest, *srcs):
    """Extract every valid zip under srcs and dest into dest, once each (the figshare archive contains inner zips)."""
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    roots = [Path(p) for p in srcs if p and Path(p).exists()] + [dest]
    done = set()
    while True:
        todo = [z for r in roots for z in r.rglob("*.zip") if z.resolve() not in done]
        if not todo:
            return
        for z in todo:
            done.add(z.resolve())
            if zipfile.is_zipfile(z):
                print(f"  unpacking {z.name}")
                with zipfile.ZipFile(z) as f:
                    f.extractall(dest / z.stem if z.parent != dest else dest)
            else:
                print(f"  skipping {z.name}: not a valid zip file ({z.stat().st_size} bytes)")


def _fetch(url, path):
    """Download url to path with a browser User-Agent; True if the result is a real zip file."""
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    if zipfile.is_zipfile(path):
        return True
    head = open(path, "rb").read(200)
    print(f"  {Path(path).name}: the server returned a web page, not a zip ({head[:60]!r}...)")
    Path(path).unlink(missing_ok=True)
    return False


def download_figshare(raw, local=None, download=True):
    """figshare brain tumor dataset (Cheng et al., version 8). Order: .mat files already present -> zips the user
    placed in `local` or `raw` -> the four data zips via the figshare API -> the whole-article zip."""
    raw = Path(raw); raw.mkdir(parents=True, exist_ok=True)
    if _figshare_ready(raw):
        return True
    _unpack_zips(raw, local)
    if _figshare_ready(raw) or not download:
        return _figshare_ready(raw)
    import urllib.request
    try:
        print("[figshare] downloading the four data files (~880 MB) via the figshare API")
        req = urllib.request.Request(f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}", headers=UA)
        files = [f for f in json.load(urllib.request.urlopen(req, timeout=60))["files"] if f["name"].endswith(".zip")]
        for f in files:
            dst = raw / f["name"]
            if not zipfile.is_zipfile(dst):
                print(f"  {f['name']} ({f['size'] / 2**20:.0f} MB)")
                _fetch(f["download_url"], dst)
        _unpack_zips(raw)
    except Exception as e:
        print(f"[figshare] API download failed: {e}")
    if not _figshare_ready(raw):
        try:
            print("[figshare] trying the whole-article download")
            if _fetch(f"https://figshare.com/ndownloader/articles/{FIGSHARE_ARTICLE}/versions/8", raw / "figshare.zip"):
                _unpack_zips(raw)
        except Exception as e:
            print(f"[figshare] download failed: {e}")
    if not _figshare_ready(raw):
        print("[figshare] AUTOMATIC DOWNLOAD FAILED. Download the dataset in your browser from\n"
              "  https://doi.org/10.6084/m9.figshare.1512427  (click 'Download all')\n"
              "  and upload the downloaded .zip file into the folder given as `figshare` "
              "(Colab notebook: My Drive > BiFAM > figshare). It is unpacked automatically.")
    return _figshare_ready(raw)


def download_br35h(raw, local=None, download=True):
    """Br35H (Kaggle ahmedhamada0/brain-tumor-detection) via kagglehub; the 'no' folder is the no-tumor class.
    A zip of the dataset placed in `local` or `raw` is used instead when present."""
    raw = Path(raw); raw.mkdir(parents=True, exist_ok=True)
    ready = lambda: any(d.is_dir() and d.name.lower() == "no" for d in raw.rglob("*"))
    if ready():
        return True
    _unpack_zips(raw, local)
    if ready() or not download:
        return ready()
    try:
        _ensure("kagglehub")
        import kagglehub
        path = kagglehub.dataset_download("ahmedhamada0/brain-tumor-detection")
        shutil.copytree(path, raw, dirs_exist_ok=True)
    except Exception as e:
        print(f"[br35h] download failed: {e}")
    if not ready():
        print("[br35h] AUTOMATIC DOWNLOAD FAILED. Either upload kaggle.json (Kaggle > Settings > API > Create New Token),\n"
              "  or download https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection in your browser and upload\n"
              "  the .zip into the folder given as `br35h` (Colab notebook: My Drive > BiFAM > br35h).")
    return ready()


def _has(p, pattern):
    return p is not None and Path(p).exists() and next(Path(p).rglob(pattern), None) is not None


# =====================================================================================================
# Commands
# =====================================================================================================
def cmd_prepare(a):
    """Download what can be downloaded, convert to PNG slices + index CSVs, write partition files."""
    raw, out = Path(a.raw), Path(a.data); out.mkdir(parents=True, exist_ok=True)
    # a.figshare / a.br35h: folders holding the .mat files or the downloaded zips (e.g. on Google Drive)
    fs_dir = Path(a.figshare) if a.figshare and _figshare_ready(a.figshare) else raw / "figshare"
    br_dir = raw / "br35h"
    bt_dir = Path(a.brats) if a.brats else raw / "brats2015"
    if fs_dir == raw / "figshare":
        download_figshare(fs_dir, a.figshare, download=not a.no_download)
    download_br35h(br_dir, a.br35h, download=not a.no_download)
    if not (bt_dir / "HGG").exists() and not _has(bt_dir, "*.mha"):
        print(f"[brats2015] not found in {bt_dir}. BraTS 2015 needs registration (https://www.smir.ch/BRATS/Start2015); "
              f"copy BRATS2015_Training (HGG/, LGG/) there, e.g. from Google Drive, and pass --brats")
    summary = {}
    if _has(fs_dir, "*.mat"):
        fs = prepare_figshare(fs_dir, out)
        summary["figshare"] = dict(slices=len(fs), patients=fs.group.nunique(),
                                   per_class={FOUR_CLASSES[k]: dict(slices=len(v), patients=int(v.group.nunique()))
                                              for k, v in fs.groupby("label")})
        print("[figshare]", summary["figshare"])
    if br_dir.exists() and any(d.is_dir() and d.name.lower() == "no" for d in br_dir.rglob("*")):
        prepare_br35h(br_dir, out, a.n_no_tumor, a.max_hamming, a.seed)
        summary["br35h"] = json.load(open(out / "br35h_dedup.json")); print("[br35h]", summary["br35h"])
    if _has(bt_dir, "*.mha"):
        bt = prepare_brats2015(bt_dir, out, a.slices_per_case)
        cases = bt.groupby("group").label.first()
        summary["brats2015"] = dict(cases=len(cases), hgg=int(cases.sum()), lgg=int((cases == 0).sum()), slices=len(bt),
                                    empty_mask_slices=int((bt.mask_pixels == 0).sum()))
        print("[brats2015]", summary["brats2015"])
    splits = make_splits(out, a.seed)
    for k, v in splits.items():
        summary[f"split_{k}"] = ([dict(train=len(f["train"]), val=len(f["val"]), test=len(f["test"])) for f in v]
                                 if isinstance(v, list) else dict(train=len(v["train"]), val=len(v["val"]), test=len(v["test"])))
    to_json(summary, out / "dataset_summary.json")
    print("[prepare] available tasks:", [k for k in TASKS if (out / "splits" / f"{k}.json").exists()])


def plot_confusion(cm, classes, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm); fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.2 * len(classes) + 1.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=30); ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


MODELS = ["bifamlite", "bifamnet"]          # proposed light model, original manuscript model


def _pred_frame(df, prob, seed, fold):
    p = df[["path", "group", "label"]].reset_index(drop=True).assign(seed=seed, fold=-1 if fold is None else fold)
    if prob.ndim == 1:
        p["prob"] = prob
    else:
        for c in range(prob.shape[1]):
            p[f"prob_{c}"] = prob[:, c]
    return p


def run_training(data, task, out, model="bifamlite", fusion="bifam", no_ca=False, no_ag=False, head="vit",
                 encoder="densenet201", pretrained=True, seeds=SEEDS, folds=None, train_over=None, model_over=None,
                 img_size=512, device=None, keep_checkpoints=True, backbone=None, seg=True, channels="auto"):
    """Train + evaluate one configuration for the given seeds. Returns the summary dict (also saved).

    channels='auto' feeds the four BraTS sequences (T1, T1ce, T2, FLAIR) when they were prepared; 't1ce' keeps the
    manuscript's single T1ce channel. keep_checkpoints=False deletes each checkpoint once its predictions are written."""
    index_task, classes, k, unit = TASKS[task]
    df = load_index(data, index_task).set_index("path", drop=False)
    split = load_split(data, task)
    fl = list(range(len(split))) if isinstance(split, list) else [None]
    if folds is not None and fl != [None]:
        fl = list(folds)
    tcfg = dict(train_over or {})
    mcfg = dict(num_outputs=k, img_size=img_size, pretrained=pretrained)
    if model in MODELS:
        mcfg.update(fusion=fusion, use_ca=not no_ca, use_ag=not no_ag, head=head)
    if model == "bifamnet":
        mcfg.update(encoder=encoder)
    if model == "bifamlite":
        mcfg.update(seg_head=seg, **({"backbone": backbone} if backbone else {}))
    if index_task == "brats" and channels == "auto" and has_multiseq(data):
        mcfg.update(in_channels=4)
    mcfg.update(model_over or {})
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    to_json(dict(task=task, model=model, model_cfg=mcfg, train_cfg=tcfg, seeds=list(seeds)), out / "config.json")
    logf = open(out / "log.txt", "a")
    log = lambda s: (print(s), logf.write(s + "\n"), logf.flush())
    summary = {}
    for seed in seeds:
        preds, vpreds = [], []
        for f in fl:
            sp = split[f] if f is not None else split
            tr, va, te = df.loc[sp["train"]], df.loc[sp["val"]], df.loc[sp["test"]]
            tag = f"seed{seed}" + (f"_fold{f}" if f is not None else "")
            log(f"== {task} {model} {tag}")
            net = None
            ck = Path(f"{out / tag}.pt")
            if ck.exists():                                   # resume: a fold finished before a disconnect
                try:
                    c = torch.load(ck, map_location="cpu", weights_only=False)
                    if c.get("model_cfg") == mcfg and c.get("model_name") == model:
                        net, _ = load_checkpoint(ck, torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")))
                        log(f"== {tag}: finished earlier - loaded {ck.name} instead of retraining")
                    del c
                except Exception as e:                       # incomplete file from an interrupted save
                    log(f"== {tag}: could not read {ck.name} ({e}); retraining")
            if net is None:
                net, info = train(data, tr, va, model, mcfg, {**tcfg, "seed": seed}, unit, out / tag, device, log)
            dev = next(net.parameters()).device
            preds.append(_pred_frame(te, predict(net, data, te, dev, tcfg), seed, f))
            vpreds.append(_pred_frame(va, predict(net, data, va, dev, tcfg), seed, f))     # for calibration
            if not keep_checkpoints:
                Path(f"{out / tag}.pt").unlink(missing_ok=True)
            del net
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pred = pd.concat(preds, ignore_index=True)
        pred.to_csv(out / f"predictions_seed{seed}.csv", index=False)
        pd.concat(vpreds, ignore_index=True).to_csv(out / f"val_predictions_seed{seed}.csv", index=False)
        pcols = [c for c in pred.columns if c.startswith("prob")]
        prob = pred["prob"].values if pcols == ["prob"] else pred[pcols].values
        r = evaluate(pred, prob, classes, unit)
        to_json(r, out / f"metrics_seed{seed}.json")
        plot_confusion(r["confusion"], classes, out / f"confusion_seed{seed}.png")
        summary[str(seed)] = dict(accuracy=r["accuracy"], macro_f1=r["macro_f1"], auc=r.get("auc"), macro_f1_ci=r["macro_f1_ci"])
        log(f"== seed {seed}: accuracy {r['accuracy'] * 100:.2f}  macro F1 {r['macro_f1'] * 100:.2f} "
            f"(95% CI {r['macro_f1_ci'][0] * 100:.1f}-{r['macro_f1_ci'][1] * 100:.1f})  AUC {100 * (r.get('auc') or np.nan):.2f}")
    f1 = [v["macro_f1"] for v in summary.values()]; acc = [v["accuracy"] for v in summary.values()]
    summary["mean"] = dict(macro_f1=float(np.mean(f1)), macro_f1_sd=float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
                           accuracy=float(np.mean(acc)), accuracy_sd=float(np.std(acc, ddof=1)) if len(acc) > 1 else 0.0)
    to_json(summary, out / "summary.json")
    log(f"== {task} {model}: macro F1 {100 * summary['mean']['macro_f1']:.2f} +- {100 * summary['mean']['macro_f1_sd']:.2f} "
        f"over {len(f1)} seed(s)")
    return summary


def _overrides(a):
    over = json.load(open(a.config)) if getattr(a, "config", None) else {}
    t = dict(over.get("train", {}))
    for key in DEFAULT_TRAIN:
        v = getattr(a, key, None)
        if key != "seed" and v is not None:
            t[key] = v
    return t, dict(over.get("model", {}))


def cmd_train(a):
    t, m = _overrides(a)
    run_training(a.data, a.task, a.out, a.model, a.fusion, a.no_ca, a.no_ag, a.head, a.encoder, not a.no_pretrained,
                 a.seeds, a.folds, t, m, a.img_size, a.device, backbone=a.backbone, seg=not a.no_seg,
                 channels="t1ce" if a.t1ce_only else "auto")


def _decisions(run, seed, unit):
    p = pd.read_csv(Path(run) / f"predictions_seed{seed}.csv", keep_default_na=False)
    if unit == "case":
        ids, pred, _ = aggregate_cases(p.group.values, p.prob.values)
        return pd.DataFrame(dict(key=ids, y=p.groupby("group").label.first().loc[ids].values, pred=pred, group=ids))
    pc = [c for c in p.columns if c.startswith("prob_")]
    pred = p[pc].values.argmax(1) if pc else (p.prob.values >= 0.5).astype(int)
    return pd.DataFrame(dict(key=p.path, y=p.label, pred=pred, group=p.group))


def compare_runs(run_a, run_b, task, seed=SEEDS[0], n_boot=2000, out=None):
    """Paired bootstrap of the macro-F1 difference (unit of evaluation), exact McNemar, seed-level t-test."""
    _, classes, k, unit = TASKS[task]
    da, db = _decisions(run_a, seed, unit), _decisions(run_b, seed, unit)
    m = da.merge(db, on="key", suffixes=("_a", "_b"))
    assert (m.y_a == m.y_b).all() and len(m) == len(da) == len(db), "runs are not on the same test samples"
    groups = m.group_a.values if unit == "patient" else None
    res = dict(a=str(run_a), b=str(run_b), task=task, seed=seed, unit=unit, n=len(m),
               paired_bootstrap=paired_bootstrap(m.y_a, m.pred_a, m.pred_b, max(k, 2), groups, n_boot),
               mcnemar=mcnemar_exact(m.y_a, m.pred_a, m.pred_b),
               accuracy_a=float((m.pred_a == m.y_a).mean()), accuracy_b=float((m.pred_b == m.y_b).mean()))
    sa = json.load(open(Path(run_a) / "summary.json")); sb = json.load(open(Path(run_b) / "summary.json"))
    common = sorted(set(sa) & set(sb) - {"mean"})
    if len(common) >= 2:
        res["seed_ttest"] = seed_ttest([sa[s]["macro_f1"] for s in common], [sb[s]["macro_f1"] for s in common])
    pb = res["paired_bootstrap"]
    print(f"[compare] {Path(run_a).name} - {Path(run_b).name}: macro F1 difference {100 * pb['diff']:.2f} points, "
          f"95% CI {100 * pb['ci'][0]:.1f} to {100 * pb['ci'][1]:.1f}; McNemar p = {res['mcnemar']['p']:.3f}"
          + ("  [image-level: descriptive only]" if unit == "image" else ""))
    if out:
        to_json(res, out)
    return res


def cmd_compare(a):
    compare_runs(a.a, a.b, a.task, a.seed, a.n_boot, a.out)


SPACE = dict(lr=[3e-5, 5e-5, 1e-4, 2e-4, 3e-4], weight_decay=[0.01, 0.05, 0.1], batch=[4, 8, 16],
             encoder_lr_mult=[0.05, 0.1, 0.3, 1.0], warmup_epochs=[1, 2, 3])


def cmd_tune(a):
    """Random search (20 configurations) on the inner validation split only; writes <out>/best.json."""
    index_task, classes, k, unit = TASKS[a.task]
    df = load_index(a.data, index_task).set_index("path", drop=False)
    sp = load_split(a.data, a.task); sp = sp[a.fold] if isinstance(sp, list) else sp
    tr, va = df.loc[sp["train"]], df.loc[sp["val"]]
    rng = np.random.default_rng(0); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); results = []
    t_base, m_base = _overrides(a)
    for i in range(a.budget):
        t = {key: v[rng.integers(len(v))] for key, v in SPACE.items()}
        t = {key: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for key, v in t.items()}
        m = dict(dropout=float([0.0, 0.1, 0.2][rng.integers(3)])) if a.model in MODELS else {}
        _, info = train(a.data, tr, va, a.model, dict(num_outputs=k, img_size=a.img_size, **m_base, **m),
                        {**t_base, **t, "seed": SEEDS[0]}, unit)
        results.append(dict(config=dict(train=t, model=m), val_macro_f1=info["best_val_macro_f1"]))
        print(f"[tune] {i + 1}/{a.budget} {results[-1]}"); to_json(results, out / "trials.json")
    best = max(results, key=lambda r: r["val_macro_f1"])
    to_json(best["config"], out / "best.json"); print("[tune] best", best)


def plot_localization(summary, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [("BiFAM", "bifam"), ("Attention gate", "ag"), ("Grad-CAM", "gradcam")]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metrics, title in ((ax[0], ("dice", "iou"), "(a) Dice and IoU"), (ax[1], ("pointing", "coverage"), "(b) Pointing game and coverage")):
        labels = [n for n, _ in names] + ["Null"]
        for j, met in enumerate(metrics):
            vals = [summary.get(f"{key}_{met}", float("nan")) for _, key in names]
            vals.append(summary.get(f"uniform_{met}", summary.get(f"random_{met}", float("nan"))))
            axis.bar([i + 0.4 * j for i in range(len(labels))], vals, 0.4, label=met)
        axis.set_xticks([i + 0.2 for i in range(len(labels))], labels); axis.set_ylim(0, 1); axis.set_title(title)
        axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def run_localization(data, run, out, seed=SEEDS[0], level=2, percentile=90, batch=8, workers=2, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = load_index(data, "brats").set_index("path", drop=False)
    rows = []
    for f, sp in enumerate(load_split(data, "brats_cv")):
        model, _ = load_checkpoint(Path(run) / f"seed{seed}_fold{f}.pt", dev)
        th = fold_thresholds(model, data, df.loc[sp["train"]], dev, percentile, batch=batch, workers=workers, level=level)
        test = df.loc[sp["test"]].reset_index(drop=True)
        r = localization_scores(model, data, test, th, dev, batch, workers, level)
        for x in r:
            x.update(fold=f, path=test.path[x["idx"]])
        rows += r
        print(f"[localize] fold {f}: thresholds {th}; {len(r)} slices")
    pd.DataFrame(rows).to_csv(out / "per_slice.csv", index=False)
    s = summarize_localization(rows)
    to_json(s, out / "summary.json"); plot_localization(s, out / "localization.png"); print("[localize]", s)
    return s


def cmd_localize(a):
    run_localization(a.data, a.run, a.out, a.seed, a.level, a.percentile, a.batch, a.workers, a.device)


def run_features(data, ckpt, out, perplexity=30, iters=1000, device=None):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint(ckpt, dev)
    df = load_index(data, "fourclass").set_index("path", drop=False)
    test = df.loc[load_split(data, "fourclass_image")["test"]]
    prob, feat = predict(model, data, test, dev, features=True)
    emb = tsne(feat, min(perplexity, max(2, len(test) // 4)), iters)
    np.savez(out / "tsne.npz", emb=emb, y=test.label.values, pred=prob.argmax(1))
    plot_tsne(emb, test.label.values, prob.argmax(1), FOUR_CLASSES, out / "tsne.png")
    _, allf = predict(model, data, df, dev, features=True)
    probe = source_probe(allf, (df.source == "br35h").astype(int).values)
    to_json(dict(source_probe=probe, n=len(df)), out / "source_probe.json"); print("[probe] source accuracy", probe)
    return probe


def cmd_features(a):
    run_features(a.data, a.ckpt, a.out, a.perplexity, a.iters, a.device)


def run_profile(out, size=512, warmup=100, runs=1000, model_over=None, only=None, device=None, main="bifamlite"):
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = [((main if f == "bifam" else f"{main}[{f}]"), main, dict(fusion=f)) for f in FUSIONS] + \
            [(b, b, {}) for b in [x for x in MODELS if x != main] + BASELINES]
    if only:
        specs = [s for s in specs if s[0] in only]
    rows = []
    for name, model_name, cfg in specs:
        extra = dict(model_over or {}) if model_name in MODELS else {}
        m = build_model(model_name, num_outputs=4, pretrained=False, img_size=size, **cfg, **extra)
        r = dict(model=name, **profile(m, dev, size, warmup=warmup, runs=runs)); rows.append(r); print("[profile]", r)
        del m
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return rows


def cmd_profile(a):
    run_profile(a.out, a.size, a.warmup, a.runs, _overrides(a)[1], a.only, a.device, a.model)


def cmd_all(a):
    """Every experiment of the manuscript (Tables 2-7, Figures 6, 8-10), skipping runs that already finished."""
    t, m = _overrides(a)
    data, runs = a.data, Path(a.runs)
    avail = [k for k in TASKS if (Path(data) / "splits" / f"{k}.json").exists()]
    print("[all] tasks available:", avail)
    main = getattr(a, "model", "bifamlite")
    kw = dict(seeds=a.seeds, train_over=t, model_over=m, img_size=a.img_size, pretrained=not a.no_pretrained,
              device=a.device, model=main, channels=getattr(a, "channels", "auto"))
    print(f"[all] main model: {main}")

    def go(task, name, **extra):
        if task not in avail:
            return
        d = runs / task / name
        if (d / "summary.json").exists():
            print(f"[all] skip {d} (done)"); return
        try:   # checkpoints are kept only for the main-model runs (needed by MIL, localization, t-SNE, figures)
            run_training(data, task, d, keep_checkpoints=a.keep_all_checkpoints or name == main, **{**kw, **extra})
        except Exception:
            import traceback; traceback.print_exc()
            try:
                (d / "ERROR.txt").write_text(traceback.format_exc())
            except OSError:
                pass

    stages = a.stages
    if "main" in stages:
        for task in ("brats_cv", "fourclass_image", "threeclass_patient", "threeclass_image"):
            go(task, main)
    if "baselines" in stages:                      # includes the other BiFAM model (original vs. light)
        for b in [x for x in MODELS if x != main] + BASELINES:
            go("brats_cv", b, model=b); go("fourclass_image", b, model=b)
    if "ablation" in stages:
        for task in ("brats_cv", "fourclass_image"):
            go(task, "abl_concat", fusion="concat"); go(task, "abl_no_ag", no_ag=True); go(task, "abl_no_ca", no_ca=True)
            go(task, "abl_pool", head="pool")
            if main == "bifamnet":
                go(task, "abl_unet", encoder="unet")
            else:
                go(task, "abl_no_mask_supervision", seg=False)
                if task == "brats_cv":
                    go(task, "abl_t1ce_only", channels="t1ce")
    if "fusion" in stages:
        for f in FUSIONS:
            if f != "bifam":
                go("brats_cv", f"fusion_{f}", fusion=f); go("fourclass_image", f"fusion_{f}", fusion=f)
        run_profile(runs / "profile.csv", a.img_size, a.profile_warmup, a.profile_runs, m, device=a.device, main=main)
    if "analysis" in stages:
        s1 = a.seeds[0]
        wk = t.get("workers", DEFAULT_TRAIN["workers"])

        def safe(name, fn):
            try:
                fn()
            except Exception:
                import traceback; print(f"[analysis] {name} failed:"); traceback.print_exc()

        for task in ("brats_cv", "fourclass_image"):
            ref = runs / task / main
            if not (ref / "summary.json").exists():
                continue
            for other in sorted(p for p in (runs / task).iterdir() if p.is_dir() and p.name != main and (p / "summary.json").exists()):
                safe(f"compare {other.name}", lambda other=other, task=task, ref=ref: compare_runs(
                    ref, other, task, s1, out=runs / task / "compare" / f"{main}_vs_{other.name}.json"))
        mainb = runs / "brats_cv" / main
        if (mainb / "summary.json").exists():
            if not (mainb / "mil" / f"mil_seed{s1}.json").exists():
                safe("MIL", lambda: run_mil(data, mainb, mainb / "mil", s1, a.device, wk))
            if not (mainb / "localization" / "summary.json").exists():
                safe("localization", lambda: run_localization(data, mainb, mainb / "localization", s1, workers=wk, device=a.device))
        for task in ("brats_cv", "fourclass_image", "threeclass_patient", "threeclass_image"):
            r = runs / task / main
            if (r / f"val_predictions_seed{s1}.csv").exists():
                safe(f"calibration {task}", lambda r=r, task=task: run_calibration(r, task, s1))
            if len(a.seeds) > 1 and (r / "summary.json").exists():
                safe(f"ensemble {task}", lambda r=r, task=task: run_ensemble(r, task, a.seeds))
        ck = runs / "fourclass_image" / main / f"seed{s1}.pt"
        if ck.exists() and not (runs / "fourclass_image" / main / "features" / "source_probe.json").exists():
            safe("features", lambda: run_features(data, ck, runs / "fourclass_image" / main / "features", device=a.device))
        run_visualize(data, runs, runs / "figures", getattr(a, "raw", None), a.n_images, s1, a.device, main)
    collect_results(runs)


def collect_results(runs):
    """One table with the mean +- SD macro F1 / accuracy of every finished run."""
    rows = []
    for s in sorted(Path(runs).glob("*/*/summary.json")):
        j = json.load(open(s)); mean = j.get("mean", {})
        rows.append(dict(task=s.parent.parent.name, run=s.parent.name, seeds=len(j) - 1,
                         macro_f1=mean.get("macro_f1"), macro_f1_sd=mean.get("macro_f1_sd"),
                         accuracy=mean.get("accuracy"), accuracy_sd=mean.get("accuracy_sd")))
    if rows:
        df = pd.DataFrame(rows); df.to_csv(Path(runs) / "results_table.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(df.to_string(index=False))


# =====================================================================================================
# Synthetic data (to test the whole pipeline in a few minutes, e.g. on a CPU runtime)
# =====================================================================================================
def make_synthetic(out):
    """Tiny fake copies of the three collections in their original file formats."""
    import h5py
    import SimpleITK as sitk
    out = Path(out); rng = np.random.default_rng(0)

    from scipy.ndimage import gaussian_filter, rotate as nd_rotate

    def blob(n, label, grade=None):
        """Brain-like phantom: skull, cortex, white matter, ventricles and a class-typical lesion.
        label: 0 glioma, 1 meningioma, 2 pituitary, 3 no tumor; grade: 'HGG' / 'LGG' for BraTS-like slices."""
        yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
        sx, sy = rng.uniform(0.85, 0.95), rng.uniform(0.95, 1.05)
        r = np.sqrt((xx / (0.78 * sx)) ** 2 + (yy / (0.9 * sy)) ** 2)
        tex = gaussian_filter(rng.normal(0, 1, (n, n)), n / 40)
        img = np.zeros((n, n))
        img[(r > 0.93) & (r < 1.03)] = 230                                     # skull / scalp
        brain = r < 0.9
        img[brain] = 95 + 25 * tex[brain]                                        # cortex
        img[r < 0.62] = 135 + 15 * tex[r < 0.62]                                 # white matter
        for cx in (-0.12, 0.12):                                                 # lateral ventricles
            img[((xx - cx) / 0.07) ** 2 + ((yy + 0.05) / 0.2) ** 2 < 1] = 35
        mask = np.zeros((n, n), bool)
        if label < 3:
            if grade is not None or label == 0:                                  # intra-axial glioma
                cx, cy, rad = rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.35), rng.uniform(0.14, 0.26)
                d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) * (1 + 0.25 * gaussian_filter(rng.normal(0, 1, (n, n)), n / 25))
                mask = d < rad
                edema = (d < rad * 1.6) & brain & ~mask
                img[edema] += 25
                if grade == "LGG":
                    img[mask] = 80 + 10 * tex[mask]                                  # non-enhancing
                else:
                    img[mask] = 70; img[mask & (d > rad * 0.7)] = 245                # necrotic core, ring enhancement
            elif label == 1:                                                     # extra-axial meningioma at the convexity
                ang = rng.uniform(0, 2 * np.pi); cx, cy = 0.62 * np.cos(ang) * sx, 0.72 * np.sin(ang) * sy
                mask = (np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) < rng.uniform(0.12, 0.2)) & (r < 1.0)
                img[mask] = 225 + 8 * tex[mask]
            else:                                                                # pituitary: sellar region
                cx, cy = rng.uniform(-0.04, 0.04), rng.uniform(0.28, 0.36)
                mask = ((xx - cx) / 0.1) ** 2 + ((yy - cy) / 0.08) ** 2 < 1
                img[mask] = 215 + 8 * tex[mask]
        img = gaussian_filter(img, 0.6) + rng.normal(0, 4, (n, n))
        ang = rng.uniform(-8, 8)
        img = nd_rotate(img, ang, reshape=False, order=1); mask = nd_rotate(mask.astype(float), ang, reshape=False, order=0) > 0.5
        return np.clip(img, 0, None), mask & (img > 20)

    fs = out / "figshare"; fs.mkdir(parents=True, exist_ok=True); i = 0
    for ours, theirs in ((0, 2), (1, 1), (2, 3)):
        for p in range(30):
            for _ in range(2):
                i += 1; img, mask = blob(128, ours)
                with h5py.File(fs / f"{i}.mat", "w") as h:
                    g = h.create_group("cjdata")
                    g["label"] = np.array([[theirs]], float)
                    g["PID"] = np.array([[ord(c)] for c in f"{ours}{p:05d}"], np.uint16)
                    g["image"] = img.T.astype(np.int16); g["tumorMask"] = mask.T.astype(np.uint8)
    no = out / "br35h" / "no"; no.mkdir(parents=True, exist_ok=True)
    for j in range(60):
        Image.fromarray(np.clip(blob(128, 3)[0], 0, 255).astype(np.uint8)).convert("RGB").save(no / f"no{j}.jpg")
    for j in range(3):
        Image.open(no / f"no{j}.jpg").save(no / f"dup{j}.jpg")
    for grade, n in (("HGG", 20), ("LGG", 10)):
        for c in range(n):
            d = out / "brats2015" / grade / f"case{c:03d}"; d.mkdir(parents=True, exist_ok=True)
            sl = [blob(96, 0, grade) for _ in range(16)]
            vol = np.stack([im * (z > 2) for z, (im, _) in enumerate(sl)]).astype(np.float32)
            seg = np.stack([m * (4 < z < 12) for z, (_, m) in enumerate(sl)]).astype(np.uint8)
            brain = vol > 20
            t1 = np.where(seg > 0, 85.0, vol)                                        # no enhancement on T1
            t2 = np.where(brain, 260 - 0.8 * vol, 0) + 60 * (seg > 0)                   # fluid and tumor bright
            fl = np.where(brain, 0.6 * vol + 40, 0) + 90 * gaussian_filter((seg > 0).astype(float), 2)   # edema bright
            for name, v in (("T1c", vol), ("T1", t1), ("T2", t2), ("Flair", fl)):
                sitk.WriteImage(sitk.GetImageFromArray(np.clip(v, 0, None).astype(np.int16)), str(d / f"VSD.Brain.XX.O.MR_{name}.1.mha"))
            sitk.WriteImage(sitk.GetImageFromArray(seg), str(d / "VSD.Brain_3more.XX.O.OT.2.mha"))
    print("[synthetic] raw data in", out)


def cmd_selftest(a):
    """Synthetic data -> prepare -> every experiment at a tiny scale (128 px, 1 epoch, small transformer)."""
    root = Path(a.out); shutil.rmtree(root, ignore_errors=True)
    make_synthetic(root / "raw")
    ns = argparse.Namespace(raw=str(root / "raw"), data=str(root / "data"), figshare=None, br35h=None, brats=None,
                            no_download=True, n_no_tumor=1400, max_hamming=4, slices_per_case=8, seed=0)
    cmd_prepare(ns)
    cfg = root / "tiny.json"
    to_json({"model": {"dim": 64, "depth": 2, "heads": 2, "dec_channels": [32, 16, 8]},
             "train": {"epochs": a.epochs, "workers": 0, "batch": 8}}, cfg)
    ns = argparse.Namespace(data=str(root / "data"), runs=str(root / "runs"), seeds=a.seeds, config=str(cfg), model=a.model,
                            img_size=128, no_pretrained=True, device=a.device, stages=a.stages,
                            profile_warmup=2, profile_runs=5, keep_all_checkpoints=False, raw=str(root / "raw"),
                            n_images=a.n_images, **{k: None for k in DEFAULT_TRAIN if k != "seed"})
    cmd_all(ns)
    print("[selftest] finished; outputs in", root / "runs")



def cmd_doctor(a):
    """Check Python, PyTorch/CUDA, GPUs, packages, disk and datasets; test one full-size training step."""
    import importlib, platform
    ok = True
    print(f"python {sys.version.split()[0]} on {platform.platform()}")
    print(f"torch {torch.__version__}  CUDA build {torch.version.cuda}  cuDNN {torch.backends.cudnn.version()}")
    for pkg in ("torchvision", "timm", "numpy", "scipy", "pandas", "sklearn", "PIL", "h5py", "SimpleITK", "matplotlib"):
        try:
            m = importlib.import_module(pkg); print(f"  {pkg:12s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:
            ok = False; print(f"  {pkg:12s} MISSING ({e})")
    if not torch.cuda.is_available():
        ok = False
        print("GPU: none visible to PyTorch. Install a CUDA build of PyTorch (see README) and check `nvidia-smi`.")
    else:
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {pr.name}, {pr.total_memory / 2**30:.1f} GB, compute capability {pr.major}.{pr.minor}")
    for d in (a.raw, a.data, a.runs):
        Path(d).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(d).free / 2**30
        print(f"disk free at {d}: {free:.0f} GB" + ("   (low: need ~15 GB for data + checkpoints)" if free < 15 else ""))
    raw = Path(a.raw)
    found = dict(figshare=_has(raw / "figshare", "*.mat"),
                 br35h=any(d.is_dir() and d.name.lower() == "no" for d in raw.rglob("*")) if raw.exists() else False,
                 brats2015=_has(Path(a.brats) if a.brats else raw / "brats2015", "*.mha"))
    print("raw datasets found:", found, "(figshare and Br35H are downloaded by `prepare`)")
    if (Path(a.data) / "splits").exists():
        print("prepared tasks:", [k for k in TASKS if (Path(a.data) / "splits" / f"{k}.json").exists()])
    if torch.cuda.is_available() and not a.skip_gpu_test:
        dev = torch.device(a.device or "cuda")
        print(f"testing a full-size training step (512 x 512, AMP) on {dev} ...")
        model = build_model(getattr(a, "model", "bifamlite"), num_outputs=4, pretrained=False).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 1e-4); scaler = torch.amp.GradScaler("cuda")
        best = None
        for bs in (16, 8, 4, 2, 1):
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(dev)
                x = torch.randn(bs, 1, 512, 512, device=dev); y = torch.randint(0, 4, (bs,), device=dev)
                t = time.time()
                for _ in range(2):
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = F.cross_entropy(model(x)["logits"].float(), y)
                    opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                torch.cuda.synchronize(dev)
                sec = (time.time() - t) / 2
                print(f"  batch {bs:2d}: OK, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB, {sec:.2f} s/step")
                best = bs
                break
            except torch.OutOfMemoryError:
                print(f"  batch {bs:2d}: out of memory")
                opt.zero_grad(set_to_none=True)
        if best:
            print(f"recommended: --batch {min(best, 8)}  (manuscript default 8; smaller batches work, results may differ slightly)")
        else:
            ok = False; print("the model does not fit even at batch 1 on this GPU")
        del model, opt; torch.cuda.empty_cache()
    print("doctor:", "all checks passed" if ok else "see the problems above")

# =====================================================================================================
# Command line
# =====================================================================================================
STAGES = ["main", "baselines", "ablation", "fusion", "analysis"]


def _add_train_args(p):
    p.add_argument("--config", help="JSON {'train': {...}, 'model': {...}} overrides (e.g. best.json from tune)")
    for key, v in DEFAULT_TRAIN.items():
        if key != "seed":
            p.add_argument(f"--{key.replace('_', '-')}", dest=key, type=int if isinstance(v, bool) else type(v), default=None)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", default=None)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="download figshare + Br35H, convert all collections, write partitions")
    p.add_argument("--raw", default="data/raw"); p.add_argument("--data", default="data/processed")
    p.add_argument("--figshare"); p.add_argument("--br35h"); p.add_argument("--brats", help="BRATS2015_Training folder")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--n-no-tumor", type=int, default=1400); p.add_argument("--max-hamming", type=int, default=4)
    p.add_argument("--slices-per-case", type=int, default=SLICES_PER_CASE); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train", help="train + evaluate one configuration")
    p.add_argument("--data", default="data/processed"); p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="bifamlite", choices=MODELS + BASELINES)
    p.add_argument("--fusion", default="bifam", choices=sorted(FUSIONS))
    p.add_argument("--backbone", help="bifamlite encoder: any timm model, e.g. densenet121 (default), efficientnet_b0")
    p.add_argument("--no-seg", action="store_true", help="bifamlite: no mask supervision")
    p.add_argument("--t1ce-only", action="store_true", help="BraTS: T1ce only instead of four sequences")
    p.add_argument("--no-ca", action="store_true"); p.add_argument("--no-ag", action="store_true")
    p.add_argument("--head", default="vit", choices=["vit", "pool"])
    p.add_argument("--encoder", default="densenet201", choices=["densenet201", "unet"])
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); p.add_argument("--folds", type=int, nargs="+")
    _add_train_args(p); p.set_defaults(fn=cmd_train)

    p = sub.add_parser("all", help="run every experiment of the manuscript")
    p.add_argument("--data", default="data/processed"); p.add_argument("--runs", default="runs")
    p.add_argument("--model", default="bifamlite", choices=MODELS, help="main model (the other one runs as a baseline)")
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--profile-warmup", type=int, default=100); p.add_argument("--profile-runs", type=int, default=1000)
    p.add_argument("--raw", help="raw download folder (figshare/, br35h/, brats2015/) for the preprocessing figure")
    p.add_argument("--n-images", type=int, default=48, help="slices per task in the attention-map figures")
    p.add_argument("--keep-all-checkpoints", action="store_true",
                   help="keep every checkpoint (default: only the main-model runs)")
    _add_train_args(p); p.set_defaults(fn=cmd_all)

    p = sub.add_parser("compare", help="paired bootstrap + McNemar between two runs")
    p.add_argument("--a", required=True); p.add_argument("--b", required=True)
    p.add_argument("--task", required=True, choices=list(TASKS)); p.add_argument("--seed", type=int, default=SEEDS[0])
    p.add_argument("--n-boot", type=int, default=2000); p.add_argument("--out"); p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("tune", help="20-configuration random search on the inner validation split")
    p.add_argument("--data", default="data/processed"); p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--out", required=True); p.add_argument("--model", default="bifamlite", choices=MODELS + BASELINES)
    p.add_argument("--fold", type=int, default=0); p.add_argument("--budget", type=int, default=20)
    _add_train_args(p); p.set_defaults(fn=cmd_tune)

    p = sub.add_parser("localize", help="BraTS localization of BiFAM / AG / Grad-CAM maps")
    p.add_argument("--data", default="data/processed"); p.add_argument("--run", required=True); p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=SEEDS[0]); p.add_argument("--level", type=int, default=2)
    p.add_argument("--percentile", type=float, default=90); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2); p.add_argument("--device"); p.set_defaults(fn=cmd_localize)

    p = sub.add_parser("features", help="t-SNE of class-token features + source probe (four-class model)")
    p.add_argument("--data", default="data/processed"); p.add_argument("--ckpt", required=True); p.add_argument("--out", required=True)
    p.add_argument("--perplexity", type=float, default=30); p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--device"); p.set_defaults(fn=cmd_features)

    p = sub.add_parser("profile", help="parameters, GFLOPs and latency of all fusion operators and baselines")
    p.add_argument("--out", default="runs/profile.csv"); p.add_argument("--size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=100); p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--only", nargs="*"); p.add_argument("--config"); p.add_argument("--device")
    p.set_defaults(fn=cmd_profile)

    p = sub.add_parser("visualize", help="sample slices, preprocessing, augmentation, attention maps, errors, curves")
    p.add_argument("--data", default="data/processed"); p.add_argument("--runs", default="runs")
    p.add_argument("--out", help="default: <runs>/figures"); p.add_argument("--raw", help="raw download folder")
    p.add_argument("--n", type=int, default=48, help="slices per task in the attention-map figures")
    p.add_argument("--seed", type=int, default=SEEDS[0]); p.add_argument("--device")
    p.add_argument("--model", default="bifamlite", choices=MODELS, help="main model whose runs are drawn")
    p.set_defaults(fn=cmd_visualize)

    p = sub.add_parser("mil", help="BraTS case-level gated-attention MIL on a finished run (vs. majority vote)")
    p.add_argument("--data", default="data/processed"); p.add_argument("--run", required=True)
    p.add_argument("--out"); p.add_argument("--seed", type=int, default=SEEDS[0]); p.add_argument("--device")
    p.add_argument("--workers", type=int, default=2)
    p.set_defaults(fn=lambda a: run_mil(a.data, a.run, a.out or Path(a.run) / "mil", a.seed, a.device, a.workers))

    p = sub.add_parser("calibrate", help="temperature scaling, ECE, conformal sets, accuracy vs. coverage")
    p.add_argument("--run", required=True); p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--seed", type=int, default=SEEDS[0]); p.add_argument("--alpha", type=float, default=0.05)
    p.set_defaults(fn=lambda a: run_calibration(a.run, a.task, a.seed, a.alpha))

    p = sub.add_parser("ensemble", help="average the seeds of a run and evaluate")
    p.add_argument("--run", required=True); p.add_argument("--task", required=True, choices=list(TASKS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.set_defaults(fn=lambda a: run_ensemble(a.run, a.task, a.seeds))

    p = sub.add_parser("doctor", help="check GPU/CUDA, packages, disk and datasets; find a batch size that fits")
    p.add_argument("--raw", default="data/raw"); p.add_argument("--data", default="data/processed")
    p.add_argument("--runs", default="runs"); p.add_argument("--brats"); p.add_argument("--device")
    p.add_argument("--skip-gpu-test", action="store_true"); p.add_argument("--model", default="bifamlite", choices=MODELS)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("selftest", help="whole pipeline on tiny synthetic data (minutes)")
    p.add_argument("--out", default="selftest"); p.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    p.add_argument("--n-images", type=int, default=48); p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--model", default="bifamlite", choices=MODELS)
    p.add_argument("--device"); p.set_defaults(fn=cmd_selftest)
    return ap


def main(argv=None):
    if argv is None and "ipykernel" in sys.modules:          # %run inside a notebook without arguments
        argv = sys.argv[1:] or ["--help"]
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
