"""Shared matplotlib style. Colors are slots 1-3 of the dataviz reference categorical palette
(validated adjacent-pair CVD separation), assigned to entities in a fixed order."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

SCENARIO_COLOR = {"per_scan": SERIES[0], "persistent": SERIES[1], "targeted": SERIES[2]}
SCENARIO_LABEL = {"per_scan": "Random per-scan", "persistent": "Random persistent",
                  "targeted": "Targeted (most-detected)"}
SCENARIO_MARKER = {"per_scan": "o", "persistent": "s", "targeted": "^"}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
