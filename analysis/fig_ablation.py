"""Ablation of BiFAM-Next on the three-class patient-level task (seed 11, 610 test slices, 46 patients)."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.linewidth": 1.0, "axes.labelweight": "bold"})
D = json.load(open("results/ablation/ablation_results.json"))
NAMES = {"full": "Full BiFAM-Next", "abl_pool": "Transformer head → global pooling", "abl_no_ppm": "Without pyramid pooling",
         "abl_concat": "BiFAM → concatenation", "abl_no_ag": "Without attention-gated path", "abl_no_mask": "Without mask supervision",
         "abl_no_ca": "Without channel attention", "abl_no_ls": "Without label smoothing", "abl_densenet121": "DenseNet121 encoder"}
keys = ["full"] + sorted([k for k in NAMES if k in D and k != "full"], key=lambda k: -D[k]["f1"])
fig = plt.figure(figsize=(15, 0.62 * len(keys) + 1.6))
a0 = fig.add_axes([0.20, 0.12, 0.40, 0.78]); a1 = fig.add_axes([0.66, 0.12, 0.25, 0.78])
y = np.arange(len(keys))[::-1]
for yi, k in zip(y, keys):
    r = D[k]; c = "#2f5d8a" if k == "full" else "#c9793a"
    a0.barh(yi, r["f1"], color=c, height=0.62, alpha=0.9)
    a0.errorbar(r["f1"], yi, xerr=[[r["f1"] - r["f1_ci"][0]], [r["f1_ci"][1] - r["f1"]]], fmt="none", ecolor="black", capsize=3, lw=1)
    a0.plot(r["acc"], yi, "D", color="black", ms=5, zorder=4)
    d = r["f1"] - D["full"]["f1"]
    a0.text(77.3, yi, f"{r['f1']:.2f}" + ("" if k == "full" else f"  ({d:+.2f})"), va="center", fontsize=10, color="white", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="square,pad=0.15", fc=c, ec="none"))
a0.axvline(D["full"]["f1"], color="#2f5d8a", ls="--", lw=1)
a0.set_yticks(y); a0.set_yticklabels([NAMES[k] for k in keys]); a0.set_xlim(77, 100.5)
a0.set_xlabel("Macro F1-score (%)  (bars, patient-bootstrap 95% CI);  black diamonds: accuracy"); a0.grid(axis="x", alpha=0.3)
a0.text(-0.02, 1.03, "(a)", transform=a0.transAxes, fontsize=14, fontweight="bold", ha="right")
M = np.array([D[k]["f1_class"] for k in keys])
im = a1.imshow(M, cmap="RdYlGn", vmin=78, vmax=100, aspect="auto")
for i in range(len(keys)):
    for j in range(3):
        a1.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=10.5, fontweight="bold")
a1.set_xticks(range(3)); a1.set_xticklabels(["Glioma", "Meningioma", "Pituitary"]); a1.set_yticks([])
a1.set_title("Per-class F1-score (%)", fontsize=11, fontweight="bold")
a1.text(-0.04, 1.03, "(b)", transform=a1.transAxes, fontsize=14, fontweight="bold", ha="right")
plt.colorbar(im, ax=a1, fraction=0.05, pad=0.03)
fig.savefig("figures/fig_ablation.png", dpi=300, bbox_inches="tight", facecolor="white")
print(keys)
