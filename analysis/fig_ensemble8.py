"""Three-class image-level ensemble of eight BiFAM-Next runs: single runs vs ensemble, confusion matrix, ROC."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
CLS = ["Glioma", "Meningioma", "Pituitary"]; CC = ["#d95350", "#4f78b0", "#6cc070"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "font.size": 11, "axes.linewidth": 1.0, "axes.labelweight": "bold"})
def tag(ax, s, x=-0.15): ax.text(x, 1.04, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
J = json.load(open("results/threeclass_image_ensemble8/ensemble8.json")); runs = J["runs"]; ens = J["ens"]
e = pd.read_csv("results/threeclass_image_ensemble8/predictions_ensemble8.csv"); P = e[["prob_0", "prob_1", "prob_2"]].to_numpy(); y = e.label.to_numpy()
cm = confusion_matrix(y, P.argmax(1), labels=range(3)); assert cm.tolist() == ens["cm"]
fig = plt.figure(figsize=(18, 5.2))
a0 = fig.add_axes([0.13, 0.13, 0.22, 0.76]); a1 = fig.add_axes([0.45, 0.13, 0.22, 0.76]); a2 = fig.add_axes([0.75, 0.13, 0.24, 0.76])
labels = [r["run"].replace("ConvNeXt-V2-", "CNX-") for r in runs] + ["Ensemble (mean of 8)"]
vals = [r["acc"] for r in runs] + [ens["acc"]]; errs = [r["errors"] for r in runs] + [int(cm.sum() - np.trace(cm))]
cols = ["#4f78b0" if "CNX" in l else "#c9793a" for l in labels[:-1]] + ["#2f2f2f"]
yy = np.arange(len(vals))[::-1]
a0.barh(yy, vals, color=cols, height=0.65)
for yi, v, n in zip(yy, vals, errs):
    a0.text(v + 0.02, yi, f"{v:.2f}  ({n})", va="center", fontsize=9.5)
a0.set_yticks(yy); a0.set_yticklabels(labels, fontsize=9.5); a0.set_xlim(97.5, 99.4)
a0.axvline(np.mean(vals[:-1]), color="gray", ls="--", lw=1)
a0.set_xlabel("Test accuracy (%)  (errors of 613)"); a0.grid(axis="x", alpha=0.3); tag(a0, "(a)", -0.55)
rowp = cm / cm.sum(1, keepdims=True) * 100
a1.imshow(rowp, cmap="Blues", vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        a1.text(j, i, f"{cm[i, j]}\n({rowp[i, j]:.1f}%)", ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if rowp[i, j] > 60 else "black")
a1.set_xticks(range(3)); a1.set_yticks(range(3)); a1.set_xticklabels(CLS); a1.set_yticklabels(CLS)
a1.set_xlabel("Predicted class"); a1.set_ylabel("True class")
a1.set_xticks(np.arange(-0.5, 3), minor=True); a1.set_yticks(np.arange(-0.5, 3), minor=True)
a1.grid(which="minor", color="white", lw=2); a1.tick_params(which="minor", length=0); tag(a1, "(b)", -0.3)
axin = a2.inset_axes([0.36, 0.40, 0.58, 0.44])
for i in range(3):
    fpr, tpr, _ = roc_curve(y == i, P[:, i])
    for ax in (a2, axin): ax.plot(fpr, tpr, color=CC[i], lw=2, label=f"{CLS[i]} (AUC = {100 * auc(fpr, tpr):.2f}%)")
a2.plot([0, 1], [0, 1], "k--", lw=1); a2.set_xlabel("False positive rate"); a2.set_ylabel("True positive rate"); a2.grid(alpha=0.3)
a2.legend(loc="lower right", fontsize=9)
axin.set_xlim(-0.002, 0.05); axin.set_ylim(0.9, 1.004); axin.grid(alpha=0.3); axin.tick_params(labelsize=8); axin.set_title("zoom: FPR ≤ 0.05", fontsize=9)
a2.set_xlim(-0.01, 1.01); a2.set_ylim(-0.01, 1.01); tag(a2, "(c)")
fig.savefig("figures/fig_res3_ensemble8.png", dpi=300, bbox_inches="tight", facecolor="white")
