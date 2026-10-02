"""One-click driver for BiFAM-Net on a local GPU. Keep this file next to bifamnet.py.

  1. Edit SETTINGS below (dataset paths, GPU, seeds, epochs, which experiments to run).
  2. python run_local.py              # uses PRESET below (default "fast": about 1-2 h on one RTX 3090/4090)
     python run_local.py --preset balanced      # or: fast / balanced / full
     python run_local.py --selftest   # whole pipeline on tiny synthetic data (minutes, no datasets needed)

Steps: doctor -> prepare datasets -> main results for the four tasks -> (optional) baselines (incl. the original
96.8 M BiFAM-Net), ablations, fusion operators -> analysis (paired tests, case-level MIL, calibration + conformal
prediction, seed ensemble, localization, t-SNE, source probe) -> figures -> results table.
Every step skips work that has already finished, so re-running continues where it stopped.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bifamnet as B  # noqa: E402

# =====================================================================================================
# SETTINGS - edit these
# =====================================================================================================
# How long a run takes (four main tasks + analysis + figures, BiFAM-Lite, one RTX 3090/4090; Colab T4 ~3-4x longer):
#   fast      1 seed,  256 px, 20 BraTS slices/case, <=15 epochs   ~1-2 h    first real results, all four tasks
#   balanced  3 seeds, 384 px, 30 BraTS slices/case, <=25 epochs   ~6-8 h    seed mean +- SD, ensemble, overnight
#   full      5 seeds, 512 px, 50 BraTS slices/case, <=40 epochs   ~1 day    manuscript protocol
PRESET = "fast"
PRESETS = dict(
    fast=dict(seeds=[11], img_size=256, slices_per_case=20, epochs=15, patience=4, batch=16, n_images=40),
    balanced=dict(seeds=[11, 22, 33], img_size=384, slices_per_case=30, epochs=25, patience=6, batch=8, n_images=60),
    full=dict(seeds=[11, 22, 33, 44, 55], img_size=512, slices_per_case=50, epochs=40, patience=8, batch=8, n_images=100),
)

SETTINGS = dict(
    # ---- where things live -------------------------------------------------------------------------
    raw="data/raw",                  # downloads go here: data/raw/figshare, data/raw/br35h
    data="data/processed",           # prepared PNG slices, index CSVs and partition files
    runs="runs",                     # checkpoints, predictions, metrics, figures
    figshare=None,                   # folder with the figshare .mat files if already downloaded (else auto)
    br35h=None,                      # folder containing Br35H's 'no' folder if already downloaded (else auto)
    brats=None,                      # folder containing BraTS 2015 HGG/ and LGG/ (registration required),
                                     # e.g. "D:/datasets/BRATS2015_Training" or "/data/BRATS2015_Training"
    download=True,                   # try to download figshare (direct) and Br35H (kagglehub)

    # ---- hardware ----------------------------------------------------------------------------------
    device="cuda:0",                 # "cuda:0", "cuda:1", ... or "cpu"
    batch=8,                         # run `python bifamnet.py doctor` to see what fits; 4 for 12-16 GB GPUs
    workers=4,                       # data-loading processes (0 if you get DataLoader errors on Windows)

    # ---- model -------------------------------------------------------------------------------------
    model="bifamlite",               # "bifamlite" (proposed light model, ~14 M parameters) or "bifamnet" (original 96.8 M)
    backbone="densenet121",          # bifamlite encoder (any timm name, e.g. "efficientnet_b0" for ~10 M total)
    multisequence_brats=True,        # BraTS input = T1, T1ce, T2, FLAIR (False: T1ce only, as in the manuscript)
    tta=True,                        # average each prediction with its horizontal flip

    # ---- experiment size (seeds, epochs, img_size, slices_per_case, batch come from PRESET) ---------
    tasks=["brats_cv", "threeclass_patient", "threeclass_image", "fourclass_image"],

    # ---- which experiments -------------------------------------------------------------------------
    run_main=True,                   # Tables 2-4: the four tasks
    run_baselines=False,             # Table 5: 13 baselines + the other BiFAM model x 2 tasks (long)
    run_ablation=False,              # Table 6: 5 variants x 2 tasks
    run_fusion=False,                # Table 7 / Fig. 8: 7 other fusion operators x 2 tasks + cost profile
    run_analysis=True,               # paired tests, MIL, calibration/conformal, ensemble, localization, t-SNE, probe
    run_figures=True,                # sample / preprocessing / augmentation / attention-map / error figures
    keep_all_checkpoints=False,      # True keeps every checkpoint (~390 MB each)
)


def ns(**kw):
    return argparse.Namespace(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=list(PRESETS), help=f"speed preset (default: {PRESET})")
    ap.add_argument("--quick", action="store_true", help="same as --preset fast")
    ap.add_argument("--tasks", nargs="+", choices=["brats_cv", "threeclass_patient", "threeclass_image", "fourclass_image"],
                    help="run only these tasks (default: SETTINGS['tasks'])")
    ap.add_argument("--selftest", action="store_true", help="tiny synthetic run of the whole pipeline")
    ap.add_argument("--skip-doctor", action="store_true")
    a = ap.parse_args()
    S = dict(SETTINGS)
    if S["device"].startswith("cuda") and not B.torch.cuda.is_available():
        print("[run_local] CUDA is not available - falling back to CPU (very slow). Run `python bifamnet.py doctor`.")
        S["device"] = "cpu"
    if a.selftest:
        B.cmd_selftest(ns(out=str(HERE / "selftest"), stages=B.STAGES, seeds=[11, 22], n_images=24, epochs=1,
                          device=S["device"], model=S["model"]))
        return
    if a.tasks:
        S["tasks"] = a.tasks
    preset = "fast" if a.quick else (a.preset or PRESET)
    S.update(PRESETS[preset])
    if S["device"] == "cpu":
        S["batch"] = min(S["batch"], 8)
    # each preset keeps its own prepared data (slices per case differ) and its own results folder
    if S["slices_per_case"] != B.SLICES_PER_CASE:
        S["data"] = f"{S['data']}_s{S['slices_per_case']}"
    S["runs"] = str(Path(S["runs"]) / preset)
    print(f"[run_local] preset '{preset}': seeds {S['seeds']}, {S['img_size']} px, {S['slices_per_case']} BraTS slices/case, "
          f"<= {S['epochs']} epochs (patience {S['patience']}), batch {S['batch']} -> results in {S['runs']}")

    if not a.skip_doctor:
        B.cmd_doctor(ns(raw=S["raw"], data=S["data"], runs=S["runs"], brats=S["brats"], device=S["device"],
                        skip_gpu_test=S["device"] == "cpu", model=S["model"]))

    # ---- 1. datasets ---------------------------------------------------------------------------------
    need = [t for t in S["tasks"] if not (Path(S["data"]) / "splits" / f"{t}.json").exists()]
    if need:
        print(f"[run_local] preparing data (missing for {need}); tasks that were already prepared get identical partitions")
        B.cmd_prepare(ns(raw=S["raw"], data=S["data"], figshare=S["figshare"], br35h=S["br35h"], brats=S["brats"],
                         no_download=not S["download"], n_no_tumor=1400, max_hamming=4,
                         slices_per_case=S["slices_per_case"], seed=0))
    else:
        print(f"[run_local] using prepared data in {S['data']} (delete its 'splits' folder to prepare again)")
    avail = [t for t in S["tasks"] if (Path(S["data"]) / "splits" / f"{t}.json").exists()]
    missing = [t for t in S["tasks"] if t not in avail]
    if missing:
        print(f"[run_local] tasks without data (skipped): {missing}")

    train_over = dict(epochs=S["epochs"], batch=S["batch"], workers=S["workers"], tta=int(S["tta"]),
                      patience=S["patience"])
    model_over = {"backbone": S["backbone"]} if S["model"] == "bifamlite" else {}
    runs = Path(S["runs"])

    # ---- 2. main results -----------------------------------------------------------------------------
    if S["run_main"]:
        for t in avail:
            out = runs / t / S["model"]
            if (out / "summary.json").exists():
                print(f"[run_local] {out} done - skipping"); continue
            B.run_training(S["data"], t, out, model=S["model"], seeds=S["seeds"], train_over=train_over,
                           model_over=model_over, img_size=S["img_size"], device=S["device"],
                           channels="auto" if S["multisequence_brats"] else "t1ce")

    # ---- 3. baselines / ablations / fusion / analysis ------------------------------------------------
    stages = [s for s, on in (("baselines", S["run_baselines"]), ("ablation", S["run_ablation"]),
                              ("fusion", S["run_fusion"]), ("analysis", S["run_analysis"])) if on]
    if stages:
        cfg_file = runs / "run_local_overrides.json"; runs.mkdir(parents=True, exist_ok=True)
        B.to_json({"model": model_over}, cfg_file)
        B.cmd_all(ns(data=S["data"], runs=str(runs), seeds=S["seeds"], stages=stages, config=str(cfg_file), model=S["model"],
                     channels="auto" if S["multisequence_brats"] else "t1ce",
                     img_size=S["img_size"], no_pretrained=False, device=S["device"], raw=S["raw"],
                     n_images=S["n_images"], profile_warmup=100, profile_runs=1000,
                     keep_all_checkpoints=S["keep_all_checkpoints"],
                     **{k: train_over.get(k) for k in B.DEFAULT_TRAIN if k != "seed"}))

    # ---- 4. figures (also made inside 'analysis'; here when analysis is off) --------------------------
    if S["run_figures"] and not S["run_analysis"]:
        B.run_visualize(S["data"], runs, runs / "figures", S["raw"], S["n_images"], S["seeds"][0], S["device"], S["model"])
    B.collect_results(runs)
    print(f"[run_local] finished. Results: {runs / 'results_table.csv'}   Figures: {runs / 'figures'}")


if __name__ == "__main__":
    main()
