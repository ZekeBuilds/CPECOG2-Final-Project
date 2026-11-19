"""Error analysis and the Assumption 2 check, on the saved seed-0 models. Writes results/analysis/.

  python -m scripts.analyze errors       # confusion matrices, building vs floor errors, errors vs APs left
  python -m scripts.analyze importance   # is "most detected" the same as "most important"? (Assumption 2)
  python -m scripts.analyze              # both
"""
import json
import sys

import keras
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score

import src.models  # noqa: F401  registers APDropout before load_model
from src import plotstyle
from src.data import CLASS_NAMES, NOT_DETECTED, ROOT, load_splits, to_model_input
from src.evaluate import outage_seed
from src.models import predict_labels
from src.outage import RATES, OutageSimulator

import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "results" / "analysis"
MODELS = {"MLP": "mlp", "MLP + APDropout": "mlp_apdrop", "AE+MLP": "ae", "AE+MLP + APDropout": "ae_apdrop",
          "1D-CNN": "cnn", "1D-CNN + APDropout": "cnn_apdrop"}
CONDITIONS = [("Clean", "per_scan", 0.0), ("30% targeted", "targeted", 0.3), ("30% random per-scan", "per_scan", 0.3)]
# Blue ramp from the dataviz reference palette (sequential = one hue, light to dark).
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load(key):
    return keras.models.load_model(ROOT / "models" / f"{key}_seed0.keras")


def corrupted(sim, x_raw, scenario, rate, seed=0):
    return sim.apply(x_raw, scenario, rate, np.random.default_rng(outage_seed(seed, scenario, rate)))


def errors(s, sim):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    y = s["y_test"]
    report = {}
    preds = {}
    for name, key in MODELS.items():
        model = load(key)
        for cond, scenario, rate in CONDITIONS:
            x_raw = corrupted(sim, s["x_test_raw"], scenario, rate)
            p = predict_labels(model, to_model_input(x_raw))
            preds[(name, cond)] = (p, x_raw)
            wrong = p != y
            n_wrong = max(int(wrong.sum()), 1)
            wrong_building = wrong & (building(p) != building(y))
            same_building = wrong & ~wrong_building
            report[f"{name} | {cond}"] = {
                "accuracy": float((~wrong).mean()),
                "errors": int(wrong.sum()),
                "wrong_building_share_of_errors": float(wrong_building.sum() / n_wrong),
                "wrong_floor_same_building_share": float(same_building.sum() / n_wrong),
                "floor_off_by_one_share": float((same_building & (np.abs(floor(p) - floor(y)) == 1)).sum()
                                                / n_wrong),
                "per_class_recall": {CLASS_NAMES[c]: float((p[y == c] == c).mean()) for c in range(13)},
            }

    # Confusion matrices: normal vs APDropout MLP, clean vs 30% targeted (row-normalised = recall).
    plotstyle.apply()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (name, cond) in zip(axes.flat, [("MLP", "Clean"), ("MLP + APDropout", "Clean"),
                                             ("MLP", "30% targeted"), ("MLP + APDropout", "30% targeted")]):
        p, _ = preds[(name, cond)]
        cm = confusion_matrix(y, p, labels=range(13), normalize="true")
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        for i in range(13):
            for j in range(13):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                            color="#fcfcfb" if cm[i, j] > 0.55 else plotstyle.INK)
        ax.set_xticks(range(13), CLASS_NAMES, rotation=90, fontsize=7)
        ax.set_yticks(range(13), CLASS_NAMES, fontsize=7)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.grid(False)
        ax.set_title(f"{name}, {cond} (acc {(p == y).mean() * 100:.1f}%)")
    fig.suptitle("Test confusion matrices, seed 0 (% of each true class)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_mlp.png")
    plt.close(fig)

    # Accuracy vs how many APs the scan still has, under 30% targeted outage.
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = [0, 1, 3, 6, 10, 15, 100]
    labels = ["0", "1-2", "3-5", "6-9", "10-14", "15+"]
    rows = []
    for i, name in enumerate(["MLP", "MLP + APDropout"]):
        p, x_raw = preds[(name, "30% targeted")]
        left = (x_raw != NOT_DETECTED).sum(axis=1)
        b = pd.cut(left, bins, right=False, labels=labels)
        acc = pd.Series(p == y).groupby(b, observed=False).mean() * 100
        cnt = pd.Series(p == y).groupby(b, observed=False).size()
        xs = np.arange(len(labels)) + (i - 0.5) * 0.38
        ax.bar(xs, acc.values, 0.34, color=plotstyle.SERIES[i], label=name)
        rows += [{"model": name, "aps_left": l, "scans": int(cnt[l]), "accuracy": float(acc[l])} for l in labels]
    counts = pd.Series(pd.cut((preds[("MLP", "30% targeted")][1] != NOT_DETECTED).sum(1), bins, right=False,
                              labels=labels)).value_counts().reindex(labels)
    ax.set_xticks(range(len(labels)), [f"{l}\n(n={counts[l]})" for l in labels], fontsize=8)
    ax.set_xlabel("APs still detected in the scan after 30% targeted outage")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Where the errors are: scans left with few APs")
    ax.legend()
    fig.savefig(OUT / "accuracy_vs_aps_left.png")
    plt.close(fig)

    (OUT / "errors.json").write_text(json.dumps({"by_model_condition": report, "accuracy_vs_aps_left": rows},
                                                indent=2))
    for k, v in report.items():
        print(f"{k:45s} acc {v['accuracy'] * 100:5.1f}  errors {v['errors']:4d}  wrong building "
              f"{v['wrong_building_share_of_errors'] * 100:5.1f}%  off-by-one floor "
              f"{v['floor_off_by_one_share'] * 100:5.1f}%")


def building(label):
    """Building from a class label B*4 + F. Label 12 is building 2, floor 4, so plain // 4 would
    wrongly give building 3; clip to 2."""
    return np.minimum(np.asarray(label) // 4, 2)


def floor(label):
    return np.asarray(label) - 4 * building(label)


def importance(s, sim):
    """Occlusion importance on the val split: drop in val macro-F1 when one AP is switched off
    everywhere. Compared with detection frequency (Spearman), then used as a 'model-aware' attack."""
    model = load("mlp")
    x_val = s["x_val_raw"]
    base = f1_score(s["y_val"], predict_labels(model, to_model_input(x_val)), average="macro")
    imp = np.zeros(len(sim.active))
    for i, ap in enumerate(sim.active):
        x = x_val.copy()
        x[:, ap] = NOT_DETECTED
        imp[i] = base - f1_score(s["y_val"], predict_labels(model, to_model_input(x)), average="macro")
    freq = sim.freq[sim.active]
    # APs never detected in the val split score 0 by construction, so leave them out of the correlation.
    in_val = (x_val[:, sim.active] != NOT_DETECTED).any(axis=0)
    rho, pval = spearmanr(freq[in_val], imp[in_val])
    # Many APs tie at 0 importance; break ties by detection frequency instead of AP index.
    by_importance = sim.active[np.lexsort((-freq, -imp))]
    n = len(sim.active)
    overlap = {f"top {int(r * 100)}%": float(len(set(by_importance[:round(r * n)]) & set(sim.ranked[:round(r * n)]))
                                             / round(r * n)) for r in (0.1, 0.2, 0.3)}

    # Attack the test set with the importance ranking instead of the frequency ranking.
    rows = []
    for name, key in [("MLP", "mlp"), ("MLP + APDropout", "mlp_apdrop")]:
        m = load(key)
        for rate in RATES:
            k = round(rate * n)
            for rank_name, order in (("frequency (our targeted)", sim.ranked), ("model importance", by_importance)):
                x = s["x_test_raw"].copy()
                x[:, order[:k]] = NOT_DETECTED
                p = predict_labels(m, to_model_input(x))
                rows.append({"model": name, "ranking": rank_name, "rate": rate,
                             "accuracy": float((p == s["y_test"]).mean()),
                             "macro_f1": float(f1_score(s["y_test"], p, average="macro"))})
    df = pd.DataFrame(rows)
    (OUT / "importance.json").write_text(json.dumps({
        "method": "occlusion on val split with the seed-0 normal MLP: importance = drop in val macro-F1",
        "base_val_macro_f1": float(base), "spearman_rho_freq_vs_importance": float(rho), "p_value": float(pval),
        "top_set_overlap": overlap, "attack": rows}, indent=2))

    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].scatter(freq * 100, imp * 100, s=10, color=plotstyle.SERIES[0], alpha=0.7)
    axes[0].set_xlabel("Scans that detect the AP (%), training split")
    axes[0].set_ylabel("Val macro-F1 drop when removed (pts)")
    axes[0].set_title(f"Detection frequency vs importance (Spearman ρ = {rho:.2f})")
    styles = {"frequency (our targeted)": "-", "model importance": "--"}
    for i, name in enumerate(["MLP", "MLP + APDropout"]):
        for rank_name, ls in styles.items():
            sub = df[(df.model == name) & (df.ranking == rank_name)]
            axes[1].plot(sub.rate * 100, sub.accuracy * 100, linestyle=ls, marker="o", color=plotstyle.SERIES[i],
                         label=f"{name}, by {rank_name}")
    axes[1].set_xlabel("APs failed (%)")
    axes[1].set_ylabel("Test accuracy (%)")
    axes[1].set_title("Which ranking hurts more?")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.savefig(OUT / "assumption2_importance.png")
    plt.close(fig)
    print(f"Spearman rho {rho:.3f} (p={pval:.2g}); top-set overlap {overlap}")
    print((df.pivot_table(index=["model", "ranking"], columns="rate", values="accuracy") * 100).round(1).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    sim = OutageSimulator(s["x_train_raw"])
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("errors", "all"):
        errors(s, sim)
    if stage in ("importance", "all"):
        importance(s, sim)
