# BiFAM-Next

Code, result files and figures for the paper

> **BiFAM-Next: A Mask-Supervised ConvNeXt-V2 U-Net Transformer with Bilinear Fusion of Dual Attention Skips for Brain MRI Tumor Classification**
> V. Vanitha, M. Srivani, G. Arulkumaran, B. Bhasker, Oana Geman, Alexandru Burlacu (manuscript submitted).

BiFAM-Next is a brain-tumor MRI classifier. A ConvNeXt-V2-Nano encoder and a pyramid pooling bottleneck feed a U-Net decoder. In this decoder, a Bilinear Feature Aggregation Module (BiFAM) fuses a channel-attended and an attention-gated version of each skip feature, and a six-layer transformer reads tokens taken directly from the decoder map. During training only, expert tumor masks supervise an auxiliary segmentation head and the BiFAM map, so the attention follows the tumor at no extra cost at inference. The network has 21.7 M parameters and needs 9.9 GMAC per 384 × 384 slice.

## Main results (all from the files in `results/`)

| Task | Protocol | Result |
|---|---|---|
| Four classes (figshare + 728 de-duplicated Br35H no-tumor images) | image level, 80:20, seed 11 | accuracy 99.47%, macro F1 99.45% (4 errors in 759 images) |
| Three classes (figshare) | image level, equal-weight ensemble of 8 runs | accuracy 98.86%, macro F1 98.70% (single runs 98.04–98.86%) |
| Three classes (figshare) | patient level, 187 training / 46 test patients | 96.39% per slice (macro F1 95.83%); 45 of 46 patients correct |
| Mask supervision (ablation) | patient level, 610 test slices | 86.5% of the BiFAM activation inside the tumor with mask supervision vs 1.5% without; macro F1 2.34 points lower without it |

## Repository layout

```
src/            bifamnet.py (models, data, training, evaluation)
                jobs_core.py (BiFAM-Next and the experiment jobs), run_local.py
notebooks/      BiFAM_Next_jobs.ipynb (Colab: data preparation and all training jobs)
                Paper1_localization.ipynb (attention localization of the full model and the ablation variants)
                Attention_gallery.ipynb (attention-map figures)
colab_cells/    the analysis notebooks as plain Python cells
analysis/       scripts that redraw the result figures from results/
results/        predictions, metrics, training histories and calibration of the reported runs
figures/        figures of the paper
```

`bifamlite` is the internal name of BiFAM-Next. It appears in the result files and in the code.

## Reproducing the results

1. **Data.** Download the figshare brain tumor dataset (Cheng, doi:10.6084/m9.figshare.1512427) and the Br35H dataset (Kaggle, `ahmedhamada0/brain-tumor-detection`). The notebook prepares the slices, removes the 772 duplicated Br35H images and writes the partition files.
2. **Training.** Open `notebooks/BiFAM_Next_jobs.ipynb` in Google Colab with a T4 GPU, set `JOB` and run all cells. Each job saves its predictions, metrics, history and checkpoint to `MyDrive/BiFAM/runs/next/<task>/<run>/`.
3. **Attention analyses.** Run `notebooks/Paper1_localization.ipynb` and `notebooks/Attention_gallery.ipynb`. They use the saved checkpoints and do not train anything.
4. **Figures from the result files** (no GPU needed). From the repository root:
   ```bash
   pip install -r requirements.txt
   python analysis/fig_ensemble8.py      # three-class ensemble figure
   python analysis/fig_ablation.py       # ablation figure
   python analysis/fig_localization.py   # localization of the decoder attention
   ```

**Training settings:**
- AdamW, learning rate 1e-4 (3e-5 for the encoder), weight decay 0.05.
- 2 warm-up epochs, then cosine decay.
- Batch size 16, at most 30 epochs, early stopping after 10 epochs without improvement of the inner-validation macro F1.
- Label smoothing 0.1; mask-loss weights 0.5 (segmentation) and 0.1 (attention).
- Horizontal-flip test-time augmentation.

## Data and weights

The imaging data are not redistributed here; please use the original sources above. Trained weights: [add the link to the release or archive].

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).
