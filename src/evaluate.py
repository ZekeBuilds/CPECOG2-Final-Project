"""Model-agnostic outage evaluation, shared by KNN and every DNN."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import plotstyle
from .outage import RATES, SCENARIOS

SEEDS = (0, 1, 2, 3, 4)


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", labels=np.unique(y_true))}


def outage_seed(seed, scenario, rate):
    """Seed keyed on what the outage is, not on loop position, so a run over a subset of
    scenarios or rates corrupts scans exactly like the full run did."""
    rate_key = RATES.index(rate) if rate in RATES else 1000 + int(round(rate * 1000))
    return [seed, SCENARIOS.index(scenario), rate_key]


def outage_curve(predict_raw, sim, x_raw, y, model_name, split_name,
                 scenarios=SCENARIOS, rates=RATES, seeds=SEEDS):
    """predict_raw: function mapping a raw RSSI matrix to class labels.

    The outage RNG is seeded by (seed, scenario index, rate index) so every model sees the
    exact same corrupted scans.
    """
    rows = []
    for scenario in scenarios:
        for rate in rates:
            for seed in seeds:
                rng = np.random.default_rng(outage_seed(seed, scenario, rate))
                y_pred = predict_raw(sim.apply(x_raw, scenario, rate, rng))
                rows.append({"model": model_name, "split": split_name, "scenario": scenario,
                             "rate": rate, "seed": seed, **scores(y, y_pred)})
    return pd.DataFrame(rows)


def summarize(df):
    return (df.groupby(["model", "split", "scenario", "rate"])[["accuracy", "macro_f1"]]
            .agg(["mean", "std"]).reset_index())


def plot_by_scenario(df, metric, title, path, models=None):
    """One panel per scenario; models as series (color + marker + legend), mean ± std."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    models = models or list(dict.fromkeys(df["model"]))
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13, 3.8), sharey=True)
    for ax, scenario in zip(axes, SCENARIOS):
        sub = df[df["scenario"] == scenario]
        for i, model in enumerate(models):
            g = sub[sub["model"] == model].groupby("rate")[metric]
            mean, std = g.mean(), g.std().fillna(0)
            x = mean.index * 100
            ax.plot(x, mean * 100, marker=markers[i], color=plotstyle.SERIES[i], label=model)
            ax.fill_between(x, (mean - std) * 100, (mean + std) * 100,
                            color=plotstyle.SERIES[i], alpha=0.15, linewidth=0)
        ax.set_title(plotstyle.SCENARIO_LABEL[scenario])
        ax.set_xlabel("APs failed (%)")
        ax.set_xticks([int(r * 100) for r in RATES])
    axes[0].set_ylabel(f"{'Accuracy' if metric == 'accuracy' else 'Macro-F1'} (%)")
    axes[0].legend(loc="lower left")
    fig.suptitle(title, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)


def plot_curves(df, metric, title, path):
    """One panel per model; scenario identity by color + marker + legend."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    models = list(dict.fromkeys(df["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.5), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        for scenario in SCENARIOS:
            g = sub[sub["scenario"] == scenario].groupby("rate")[metric]
            mean, std = g.mean(), g.std().fillna(0)
            x = mean.index * 100
            ax.plot(x, mean * 100, marker=plotstyle.SCENARIO_MARKER[scenario],
                    color=plotstyle.SCENARIO_COLOR[scenario], label=plotstyle.SCENARIO_LABEL[scenario])
            ax.fill_between(x, (mean - std) * 100, (mean + std) * 100,
                            color=plotstyle.SCENARIO_COLOR[scenario], alpha=0.15, linewidth=0)
        ax.set_title(model if len(models) > 1 else title)
        ax.set_xlabel("APs failed (%)")
        ax.set_xticks([int(r * 100) for r in RATES])
    axes[0][0].set_ylabel(f"{'Accuracy' if metric == 'accuracy' else 'Macro-F1'} (%)")
    axes[0][0].legend(loc="lower left")
    if len(models) > 1:
        fig.suptitle(title, fontweight="bold")
    fig.savefig(path)
    plt.close(fig)
