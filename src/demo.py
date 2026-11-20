"""Logic and figures for the Streamlit demo (app.py). Kept free of Streamlit so it can be tested.

Every number the demo shows is computed here at run time from the saved seed-0 models and the
test file, using the same outage seeds as src/evaluate.py. With 5 outage seeds KNN matches the
report table exactly; DNNs use only the seed-0 model, so they differ from the 5-model averages.
"""
import numpy as np

from . import plotstyle
from .data import CLASS_NAMES, NOT_DETECTED, ROOT, RSSI_FLOOR, to_model_input
from .evaluate import outage_seed, scores
from .outage import RATES

import matplotlib.pyplot as plt  # noqa: E402  (plotstyle sets the Agg backend first)

MODELS_DIR = ROOT / "models"
FAMILIES = {"MLP": "mlp", "AE + MLP": "ae", "1D-CNN": "cnn"}
SCENARIO_HELP = {
    "per_scan": "Each scan loses its own random share of the APs it detected.",
    "persistent": "One random set of APs is off for every scan.",
    "targeted": "The APs phones detect most often are off for every scan.",
}
# Model roles keep the colors of the report's main figure (scripts/summarize.py).
ROLE_COLOR = {"normal": plotstyle.SERIES[0], "apdrop": plotstyle.SERIES[1], "knn": plotstyle.SERIES[2]}
ROLE_MARKER = {"normal": "o", "apdrop": "s", "knn": "^"}
FAILED_COLOR = plotstyle.SERIES[7]
NAVY = "#1e3a5f"
# Sequential blue ramp from the dataviz reference palette, same as scripts/analyze.py.
BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def role_names(family):
    return {"normal": family, "apdrop": f"{family} + APDropout", "knn": "KNN"}


def is_deterministic(scenario, rate):
    """Targeted outages and clean input do not depend on the outage seed."""
    return rate <= 0 or scenario == "targeted"


def seeds_for(scenario, rate, n_seeds):
    return (0,) if is_deterministic(scenario, rate) else tuple(range(n_seeds))


def corrupt(sim, x_raw, scenario, rate, seed):
    """Same RNG keying as evaluate.outage_curve, so every model sees identical corrupted scans."""
    return sim.apply(x_raw, scenario, rate, np.random.default_rng(outage_seed(seed, scenario, rate)))


def evaluate_point(predict_raw, sim, x_raw, y, scenario, rate, seed):
    y_pred = predict_raw(corrupt(sim, x_raw, scenario, rate, seed))
    return {**scores(y, y_pred), "pred": y_pred}


def building(label):
    """Label 12 is building 2, floor 4, so plain // 4 would give building 3; clip to 2."""
    return np.minimum(np.asarray(label) // 4, 2)


def floor(label):
    return np.asarray(label) - 4 * building(label)


def error_split(y, y_pred):
    """Counts of wrong-building and right-building-wrong-floor errors."""
    wrong = y_pred != y
    wrong_building = wrong & (building(y_pred) != building(y))
    return {"scans": int(len(y)), "errors": int(wrong.sum()),
            "wrong_building": int(wrong_building.sum()), "wrong_floor": int((wrong & ~wrong_building).sum())}


def dnn_predictor(model):
    """Same prediction path as the training scripts (models.predict_labels)."""
    from .models import predict_labels

    return lambda x_raw: predict_labels(model, to_model_input(x_raw))


def dnn_probabilities(model, x_raw):
    return np.asarray(model(to_model_input(x_raw), training=False))


# ---------------------------------------------------------------- figures

def _metric_label(metric):
    return "Macro-F1 (%)" if metric == "macro_f1" else "Accuracy (%)"


def plot_curve(df, metric, family, scenario, selected_rate):
    """Metric vs share of APs failed, one line per model, mean ± std over outage seeds.

    df columns: role, rate, seed, accuracy, macro_f1.
    """
    plotstyle.apply()
    names = role_names(family)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for role in ("normal", "apdrop", "knn"):
        sub = df[df["role"] == role]
        if sub.empty:
            continue
        g = sub.groupby("rate")[metric]
        mean, std = g.mean() * 100, g.std().fillna(0) * 100
        x = mean.index * 100
        ax.plot(x, mean, marker=ROLE_MARKER[role], color=ROLE_COLOR[role], label=names[role])
        ax.fill_between(x, mean - std, mean + std, color=ROLE_COLOR[role], alpha=0.15, linewidth=0)
    ax.axvline(selected_rate * 100, color=plotstyle.INK_2, linewidth=1, linestyle=":")
    ax.set_xticks([int(r * 100) for r in RATES])
    ax.set_xlabel("APs failed (%)")
    ax.set_ylabel(_metric_label(metric))
    ax.set_ylim(0, 100)
    ax.set_title(f"{plotstyle.SCENARIO_LABEL[scenario]} outage, test set")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    return fig


def plot_ap_strip(scan_clean, scan_out, ranked, n_active, network_failed=None):
    """All active APs of one scan, ordered by how often phones detect them in training
    (most detected on the left). Bar height = signal strength; red = switched off by the outage.

    scan_clean, scan_out: raw RSSI rows (+100 = not detected). ranked: AP indices, most detected
    first. network_failed: AP indices off for every scan (persistent/targeted), drawn as a band.
    """
    plotstyle.apply()
    order = np.asarray(ranked)[:n_active]
    detected_all = scan_clean != NOT_DETECTED
    failed_all = detected_all & (scan_out == NOT_DETECTED)
    # APs heard here but never in training have no rank; count them, but they cannot be drawn.
    unseen = np.ones(len(scan_clean), dtype=bool)
    unseen[order] = False
    n_unseen = int((detected_all & unseen).sum())
    clean = scan_clean[order]
    detected = detected_all[order]
    failed = failed_all[order]
    kept = detected & ~failed
    height = np.where(detected, (clean - RSSI_FLOOR) / -RSSI_FLOOR, 0) * 100
    pos = np.arange(len(order))

    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 2.6))
    handles = [Patch(color=NAVY, label=f"Heard ({(detected_all & ~failed_all).sum()})"),
               Patch(color=FAILED_COLOR, label=f"Heard, then failed ({failed_all.sum()})")]
    if network_failed is not None and len(network_failed):
        is_off = np.isin(order, network_failed)
        ax.bar(pos[is_off], np.full(is_off.sum(), 100), width=1.0, color=FAILED_COLOR, alpha=0.08,
               linewidth=0)
        handles.append(Patch(color=FAILED_COLOR, alpha=0.08, label="AP off for all scans"))
    ax.bar(pos[kept], height[kept], width=2.2, color=NAVY, linewidth=0)
    ax.bar(pos[failed], height[failed], width=2.2, color=FAILED_COLOR, linewidth=0)
    ax.set_xlim(-4, len(order) + 4)
    ax.set_ylim(0, 100)
    note = (f"\n{n_unseen} heard AP{'s' if n_unseen != 1 else ''} never seen in training: "
            "counted in the legend, not drawn") if n_unseen else ""
    ax.set_xlabel(f"{len(order)} access points, ranked by how often training scans detect them "
                  f"(most detected on the left){note}")
    ax.set_ylabel("Signal (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=handles, loc="upper right", ncol=3)
    fig.tight_layout()
    return fig


def plot_buildings(true_label, preds):
    """Three buildings as stacked floors. The true floor is filled; each model's guess is a marker.

    preds: list of (role, name, label).
    """
    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    width, gap = 3.0, 1.0
    floors = {0: 4, 1: 4, 2: 5}
    true_b, true_f = int(building(true_label)), int(floor(true_label))
    for b, n in floors.items():
        x0 = b * (width + gap)
        for f in range(n):
            is_true = (b, f) == (true_b, true_f)
            ax.add_patch(plt.Rectangle((x0, f), width, 0.9, facecolor=BLUES[1] if is_true else "#f1f0ec",
                                       edgecolor=NAVY if is_true else plotstyle.GRID,
                                       linewidth=2 if is_true else 1))
            ax.text(x0 - 0.12, f + 0.45, f"F{f}", ha="right", va="center", fontsize=8, color=plotstyle.INK_2)
        ax.text(x0 + width / 2, -0.45, f"Building {b}", ha="center", va="center", fontsize=9)
    for i, (role, name, label) in enumerate(preds):
        b, f = int(building(label)), int(floor(label))
        x = b * (width + gap) + width * (i + 1) / (len(preds) + 1)
        ax.plot(x, f + 0.45, marker=ROLE_MARKER[role], markersize=11, color=ROLE_COLOR[role],
                markeredgecolor="white", markeredgewidth=1.2, linestyle="none", label=name)
    ax.set_xlim(-0.8, 3 * width + 2 * gap + 0.2)
    ax.set_ylim(-0.8, 5.1)
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=len(preds))
    ax.set_title(f"True location: {CLASS_NAMES[true_label]} (filled floor)", loc="left")
    fig.tight_layout()
    return fig


def plot_confusion(y, y_pred, title):
    """Row-normalized confusion matrix; building blocks outlined."""
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    plotstyle.apply()
    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y, y_pred, labels=labels).astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.imshow(cm * 100, cmap=LinearSegmentedColormap.from_list("blues", BLUES), vmin=0, vmax=100)
    for i in labels:
        for j in labels:
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.55 else plotstyle.INK)
    for start, size in ((0, 4), (4, 4), (8, 5)):
        ax.add_patch(plt.Rectangle((start - 0.5, start - 0.5), size, size, fill=False,
                                   edgecolor=NAVY, linewidth=1.2))
    ax.set_xticks(labels, CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(labels, CLASS_NAMES, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig
