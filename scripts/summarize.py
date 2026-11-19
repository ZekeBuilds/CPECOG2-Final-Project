"""Cross-model summary from saved outage runs. Writes results/summary/.

  python -m scripts.summarize            # test split (default)
  python -m scripts.summarize val
"""
import sys

import numpy as np
import pandas as pd

from src import plotstyle
from src.data import ROOT

import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "results" / "summary"
SOURCES = ["baselines", "mlp", "ae", "cnn"]
FAMILIES = ["KNN", "MLP", "AE+MLP", "1D-CNN"]
CONDITIONS = [("Clean", None, 0.0), ("30% random per-scan", "per_scan", 0.3),
              ("30% random persistent", "persistent", 0.3), ("30% targeted", "targeted", 0.3)]


def load_runs():
    frames = [pd.read_csv(p) for src in SOURCES if (p := ROOT / "results" / src / "outage_runs.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    return df[df.model != "Majority class"]


def condition(df, scenario, rate):
    # Rate 0 is identical across scenarios; use per_scan rows so each seed counts once.
    return df[(df.rate == rate) & (df.scenario == (scenario or "per_scan"))]


def main(split="test"):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    df = df[df.split == split]

    rows = []
    for model, sub in df.groupby("model", sort=False):
        row = {"model": model}
        for label, scenario, rate in CONDITIONS:
            c = condition(sub, scenario, rate)
            for m in ("accuracy", "macro_f1"):
                row[f"{label} {m}"] = f"{c[m].mean() * 100:.1f} ± {c[m].std() * 100:.1f}"
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / f"{split}_table.csv", index=False, encoding="utf-8-sig")  # BOM so Excel shows ±

    plotstyle.apply()
    present = [f for f in FAMILIES if (df.model == f).any()]
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(13, 3.6), sharey=True)
    x = np.arange(len(present))
    w = 0.38
    for ax, (label, scenario, rate) in zip(axes, CONDITIONS):
        c = condition(df, scenario, rate)
        series = [("KNN", "", plotstyle.SERIES[2], "KNN baseline"),
                  ("DNN", "", plotstyle.SERIES[0], "DNN, normal training"),
                  ("DNN", " + APDropout", plotstyle.SERIES[1], "DNN, with APDropout")]
        for kind, suffix, color, name in series:
            means, stds, xs = [], [], []
            for i, fam in enumerate(present):
                if (fam == "KNN") != (kind == "KNN"):
                    continue
                vals = c[c.model == fam + suffix].macro_f1
                if len(vals):
                    xs.append(i if kind == "KNN" else i + (-0.5 if not suffix else 0.5) * w)
                    means.append(vals.mean() * 100)
                    stds.append(vals.std() * 100)
            if not xs:
                continue
            ax.bar(xs, means, w - 0.04, yerr=stds, color=color, label=name,
                   error_kw={"ecolor": plotstyle.INK_2, "elinewidth": 1, "capsize": 2})
            for xi, m, sd in zip(xs, means, stds):
                ax.text(xi, m + sd + 1.5, f"{m:.0f}", ha="center", fontsize=7, color=plotstyle.INK_2)
        ax.set_title(label, pad=10)
        ax.set_xticks(x, present, fontsize=8)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Macro-F1 (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.08), fontsize=9)
    fig.suptitle(f"{split.capitalize()} macro-F1 by model (mean ± std, 5 seeds)", fontweight="bold", y=1.04)
    fig.savefig(OUT / f"{split}_macro_f1_bars.png")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
