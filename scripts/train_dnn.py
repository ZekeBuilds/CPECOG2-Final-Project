"""DNN families (MLP, AE+MLP, 1D-CNN), each normal vs APDropout-trained. Logs to results/<family>/.

  python -m scripts.train_dnn <family> arch     # pick architecture on CLEAN val (normal model, seed 0)
  python -m scripts.train_dnn <family> hparams  # sweep lr, batch, l2 one at a time (seed 0) + learning curves
  python -m scripts.train_dnn <family> apdrop   # pick APDropout max_rate on val OUTAGES (seed 0)
  python -m scripts.train_dnn <family> final    # 5 seeds each, outage curves on val and test, save models
  python -m scripts.train_dnn <family> all      # the four stages in order

family is one of: mlp, ae, cnn.

Selection rules (fixed before looking at the test set, same for every family):
- arch: highest clean val macro-F1 (ties -> fewer parameters).
- apdrop: highest val macro-F1 averaged over all 3 scenarios x 6 rates. Clean cost is logged too.
Training seed s is paired with outage seed s, so the 5 runs per point cover both sources of noise.
"""
import itertools
import json
import sys
import time

import keras
import pandas as pd

from src.data import ROOT, load_splits, to_model_input
from src.evaluate import outage_curve, plot_by_scenario, scores, summarize
from src.models import build_ae_classifier, build_autoencoder, build_cnn, build_mlp, fit, predict_labels
from src.outage import OutageSimulator

MODEL_DIR = ROOT / "models"
FIXED = {"l2": 1e-4, "lr": 1e-3}
BATCH = 256
APDROP_GRID = [0.2, 0.4, 0.6, 0.8, 0.9]
SEEDS = (0, 1, 2, 3, 4)


# lr, l2 and batch default to FIXED/BATCH; the "hparams" stage overrides them one at a time.
# Each train_* returns (model, info, history of the supervised fit).

def train_mlp(s, seed, ap_dropout=0.0, lr=FIXED["lr"], l2=FIXED["l2"], batch=BATCH, **cfg):
    keras.utils.set_random_seed(seed)
    model = build_mlp(seed=seed, ap_dropout=ap_dropout, lr=lr, l2=l2, **cfg)
    hist = fit(model, s["x_train"], s["y_train"], s["x_val"], s["y_val"], batch_size=batch)
    return model, {"epochs": len(hist.history["loss"])}, hist.history


def train_ae(s, seed, ap_dropout=0.0, encoder=(256, 128), dropout=0.2, head_units=128,
             lr=FIXED["lr"], l2=FIXED["l2"], batch=BATCH):
    """Unsupervised (denoising if ap_dropout > 0) pretraining on the train split, then fine-tuning."""
    keras.utils.set_random_seed(seed)
    ae, enc = build_autoencoder(encoder=encoder, ap_dropout=ap_dropout, lr=lr, seed=seed)
    h_ae = fit(ae, s["x_train"], s["x_train"], s["x_val"], s["x_val"], epochs=100, patience=10,
               batch_size=batch)
    model = build_ae_classifier(enc, head_units=head_units, dropout=dropout, seed=seed, lr=lr, l2=l2)
    hist = fit(model, s["x_train"], s["y_train"], s["x_val"], s["y_val"], batch_size=batch)
    return model, {"ae_epochs": len(h_ae.history["loss"]), "epochs": len(hist.history["loss"]),
                   "ae_val_mse": float(min(h_ae.history["val_loss"]))}, hist.history


def train_cnn(s, seed, ap_dropout=0.0, lr=FIXED["lr"], l2=FIXED["l2"], batch=BATCH, **cfg):
    keras.utils.set_random_seed(seed)
    model = build_cnn(seed=seed, ap_dropout=ap_dropout, lr=lr, l2=l2, **cfg)
    hist = fit(model, s["x_train"], s["y_train"], s["x_val"], s["y_val"], batch_size=batch)
    return model, {"epochs": len(hist.history["loss"])}, hist.history


# One-at-a-time sensitivity sweep around the defaults (lr 1e-3, batch 256, l2 1e-4).
HPARAM_SWEEP = {"lr": [3e-4, 1e-3, 3e-3], "batch": [128, 256, 512], "l2": [0.0, 1e-4, 1e-3]}
HPARAM_DEFAULTS = {"lr": FIXED["lr"], "batch": BATCH, "l2": FIXED["l2"]}
# Chosen: a value replaces the default only if it beats it by more than 0.5 points of clean val
# macro-F1 (single seed). Smaller gaps are inside the seed-to-seed spread of the 5-seed runs.
ADOPT_MARGIN = 0.005


FAMILIES = {
    "mlp": {"name": "MLP", "train": train_mlp,
            "grid": {"hidden": [[256, 128], [512, 256], [512, 256, 128], [1024, 512]], "dropout": [0.2, 0.4]}},
    "ae": {"name": "AE+MLP", "train": train_ae,
           "grid": {"encoder": [[256, 128], [256, 64]], "dropout": [0.2, 0.4]}},
    "cnn": {"name": "1D-CNN", "train": train_cnn,
            "grid": {"proj": [128, 256], "dropout": [0.2, 0.4]}},
}


class Run:
    def __init__(self, family):
        self.key = family
        self.spec = FAMILIES[family]
        self.out = ROOT / "results" / family
        self.out.mkdir(parents=True, exist_ok=True)
        s = load_splits()
        s["x_train"] = to_model_input(s["x_train_raw"])
        s["x_val"] = to_model_input(s["x_val_raw"])
        s["sim"] = OutageSimulator(s["x_train_raw"])
        self.s = s

    def train(self, seed, **cfg):
        t0 = time.time()
        model, info, self.last_history = self.spec["train"](self.s, seed, **cfg)
        return model, {**info, "train_seconds": round(time.time() - t0, 1)}

    def predictor(self, model):
        return lambda xr: predict_labels(model, to_model_input(xr))

    def load(self, name):
        return json.loads((self.out / name).read_text())

    def best_arch(self):
        """Architecture from the arch stage, plus any lr/batch/l2 adopted by the hparams stage."""
        best = self.load("arch_tuning.json")["best"]
        cfg = {k: best[k] for k in self.spec["grid"]}
        if (self.out / "hparam_tuning.json").exists():
            cfg.update(self.load("hparam_tuning.json")["adopted"])
        return cfg

    def hparams(self):
        """Sweep lr, batch and l2 one at a time on the normal model (seed 0, clean val) and save
        learning curves. Runs after arch; if anything is adopted, rerun apdrop and final."""
        import matplotlib.pyplot as plt

        from src import plotstyle

        best = self.load("arch_tuning.json")["best"]
        arch = {k: best[k] for k in self.spec["grid"]}
        log, histories = [], {}
        model, info = self.train(0, **arch)  # default config, shared by all three sweeps
        default_row = {"hparam": "default", "value": None, **HPARAM_DEFAULTS, **info,
                       **scores(self.s["y_val"], predict_labels(model, self.s["x_val"]))}
        log.append(default_row)
        default_hist = self.last_history
        print(default_row, flush=True)
        for name, values in HPARAM_SWEEP.items():
            for v in values:
                if v == HPARAM_DEFAULTS[name]:
                    histories[(name, v)] = default_hist
                    continue
                model, info = self.train(0, **arch, **{name: v})
                row = {"hparam": name, "value": v, **info,
                       **scores(self.s["y_val"], predict_labels(model, self.s["x_val"]))}
                log.append(row)
                histories[(name, v)] = self.last_history
                print(row, flush=True)
        adopted = {}
        for name in HPARAM_SWEEP:
            top = max([r for r in log if r["hparam"] == name] + [default_row], key=lambda r: r["macro_f1"])
            if top is not default_row and top["macro_f1"] > default_row["macro_f1"] + ADOPT_MARGIN:
                adopted[name] = top["value"]
        (self.out / "hparam_tuning.json").write_text(json.dumps({
            "rule": f"one at a time around defaults {HPARAM_DEFAULTS}; adopt only if clean val macro-F1 "
                    f"beats the default by more than {ADOPT_MARGIN}",
            "arch": arch, "adopted": adopted, "runs": log,
            "histories": {f"{k[0]}={k[1]:g}": {m: [float(x) for x in h[m]] for m in h}
                          for k, h in histories.items()}}, indent=2))

        plotstyle.apply()
        fig, axes = plt.subplots(1, len(HPARAM_SWEEP), figsize=(13, 3.6), sharey=True)
        for ax, (name, values) in zip(axes, HPARAM_SWEEP.items()):
            for i, v in enumerate(values):
                h = histories[(name, v)]
                ep = range(1, len(h["val_loss"]) + 1)
                tag = f"{name} = {v:g}" + (" (default)" if v == HPARAM_DEFAULTS[name] else "")
                ax.plot(ep, h["val_loss"], color=plotstyle.SERIES[i], label=tag)
                ax.plot(ep, h["loss"], color=plotstyle.SERIES[i], linewidth=1, linestyle=":")
            ax.set_title(f"Effect of {name}")
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Loss, log scale")
        fig.suptitle(f"{self.spec['name']} learning curves, seed 0 (solid = val, dotted = train)",
                     fontweight="bold", y=1.03)
        fig.savefig(self.out / "learning_curves.png")
        plt.close(fig)
        print("ADOPTED", adopted, flush=True)

    def arch(self):
        grid = self.spec["grid"]
        log = []
        for values in itertools.product(*grid.values()):
            cfg = dict(zip(grid, values))
            model, info = self.train(0, **cfg)
            row = {**cfg, "params": model.count_params(), **info,
                   **scores(self.s["y_val"], predict_labels(model, self.s["x_val"]))}
            log.append(row)
            print(row, flush=True)
        best = max(log, key=lambda r: (round(r["macro_f1"], 4), -r["params"]))
        (self.out / "arch_tuning.json").write_text(json.dumps({
            "rule": "max clean val macro-F1, ties -> fewer params",
            "fixed": FIXED | {"batch": BATCH}, "best": best, "grid": log}, indent=2))
        print("BEST", best, flush=True)

    def apdrop(self):
        arch = self.best_arch()
        log = []
        for rate in [0.0] + APDROP_GRID:
            model, info = self.train(0, ap_dropout=rate, **arch)
            df = outage_curve(self.predictor(model), self.s["sim"], self.s["x_val_raw"], self.s["y_val"],
                              f"r={rate}", "val", seeds=(0,))
            row = {"ap_dropout": rate, **info, "clean_macro_f1": float(df[df.rate == 0].macro_f1.mean()),
                   "mean_outage_macro_f1": float(df.macro_f1.mean()),
                   **{f"{sc}_mean_macro_f1": float(df[df.scenario == sc].macro_f1.mean())
                      for sc in df.scenario.unique()}}
            log.append(row)
            print(row, flush=True)
        best = max((r for r in log if r["ap_dropout"] > 0), key=lambda r: round(r["mean_outage_macro_f1"], 4))
        (self.out / "apdrop_tuning.json").write_text(json.dumps({
            "rule": "max val macro-F1 averaged over 3 scenarios x 6 rates (rate 0.0 = normal model, reference only)",
            "arch": arch, "best": best, "grid": log}, indent=2))
        print("BEST", best, flush=True)

    def final(self):
        arch = self.best_arch()
        ap_rate = self.load("apdrop_tuning.json")["best"]["ap_dropout"]
        base = self.spec["name"]
        variants = {base: 0.0, f"{base} + APDropout": ap_rate}
        MODEL_DIR.mkdir(exist_ok=True)
        frames, train_log = [], []
        for name, rate in variants.items():
            for seed in SEEDS:
                model, info = self.train(seed, ap_dropout=rate, **arch)
                train_log.append({"model": name, "seed": seed, "ap_dropout": rate, **info})
                print(train_log[-1], flush=True)
                if seed == 0:
                    model.save(MODEL_DIR / f"{self.key}{'_apdrop' if rate else ''}_seed0.keras")
                for split in ("val", "test"):
                    frames.append(outage_curve(self.predictor(model), self.s["sim"], self.s[f"x_{split}_raw"],
                                               self.s[f"y_{split}"], name, split, seeds=(seed,)))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(self.out / "outage_runs.csv", index=False)
        summarize(df).to_csv(self.out / "outage_summary.csv", index=False)
        (self.out / "final_training.json").write_text(json.dumps(
            {"arch": arch, "fixed": FIXED | {"batch": BATCH}, "runs": train_log}, indent=2))

        knn = pd.read_csv(ROOT / "results" / "baselines" / "outage_runs.csv")
        both = pd.concat([knn[knn.model == "KNN"], df], ignore_index=True)
        order = ["KNN", *variants]
        for split in ("test", "val"):
            for metric in ("accuracy", "macro_f1"):
                label = "accuracy" if metric == "accuracy" else "macro-F1"
                plot_by_scenario(both[both.split == split], metric,
                                 f"{split.capitalize()} {label} under AP outages (mean ± std, 5 seeds)",
                                 self.out / f"compare_{split}_{metric}.png", models=order)
        table = (both.groupby(["split", "model", "scenario", "rate"])[["accuracy", "macro_f1"]].mean() * 100).round(1)
        print(table.unstack("rate").to_string())


if __name__ == "__main__":
    family, stage = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "all")
    run = Run(family)
    for st in (["arch", "hparams", "apdrop", "final"] if stage == "all" else [stage]):
        getattr(run, st)()
