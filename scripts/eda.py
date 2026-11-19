"""Exploratory data analysis. Writes results/eda/summary.json and figures.

Run from the project root:  python -m scripts.eda
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import plotstyle
from src.data import (AP_COLS, CLASS_NAMES, NOT_DETECTED, ROOT, clean_train, load_raw,
                      load_splits, reference_point_groups)
from src.outage import detection_frequency

import matplotlib.pyplot as plt  # noqa: E402  (after plotstyle sets the backend)

OUT = ROOT / "results" / "eda"


def detected_counts(df):
    return (df[AP_COLS].to_numpy() != NOT_DETECTED).sum(axis=1)


def describe(a):
    return {"mean": float(np.mean(a)), "median": float(np.median(a)),
            "min": int(np.min(a)), "max": int(np.max(a))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plotstyle.apply()
    train_raw, test = load_raw()
    train, clean_report = clean_train(train_raw)
    s = load_splits()

    x_tr = train[AP_COLS].to_numpy()
    seen_train = (x_tr != NOT_DETECTED).any(axis=0)
    seen_test = (test[AP_COLS].to_numpy() != NOT_DETECTED).any(axis=0)
    freq = detection_frequency(s["x_train_raw"])
    readings = x_tr[x_tr != NOT_DETECTED]

    ytr, yva, yte = s["y_train"], s["y_val"], s["y_test"]
    overlap = set(s["groups_train"]) & set(s["groups_val"])
    summary = {
        "cleaning": clean_report,
        "rows": {"train_split": int(len(ytr)), "val_split": int(len(yva)), "test": int(len(yte))},
        "reference_points": {"train_split": int(len(set(s["groups_train"]))),
                             "val_split": int(len(set(s["groups_val"]))),
                             "shared_between_train_and_val": len(overlap)},
        "class_counts": {
            name: {"train": int((ytr == c).sum()), "val": int((yva == c).sum()), "test": int((yte == c).sum())}
            for c, name in enumerate(CLASS_NAMES)},
        "detected_aps_per_scan": {"train": describe(detected_counts(train)),
                                  "test": describe(detected_counts(test))},
        "aps": {"total": len(AP_COLS), "never_detected_in_train": int((~seen_train).sum()),
                "never_detected_in_test": int((~seen_test).sum()),
                "in_test_but_not_train": int((seen_test & ~seen_train).sum()),
                "active_in_train_split": int((freq > 0).sum())},
        "rssi_dbm": {"min": int(readings.min()), "max": int(readings.max()),
                     "readings_at_0_dbm": int((readings == 0).sum())},
        "test_users_phones": {"USERID": sorted(map(int, test.USERID.unique())),
                              "PHONEID": sorted(map(int, test.PHONEID.unique()))},
        "train_users_phones": {"USERID": sorted(map(int, train.USERID.unique())),
                               "PHONEID": sorted(map(int, train.PHONEID.unique()))},
    }
    # How much of the detection mass the top APs carry: this is what makes targeted outage hard.
    active_sorted = np.sort(freq[freq > 0])[::-1]
    mass = np.cumsum(active_sorted) / active_sorted.sum()
    summary["detection_mass_in_top"] = {f"{p}%": float(mass[int(round(p / 100 * len(mass))) - 1])
                                        for p in (10, 20, 30, 50)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # 1. Class counts per split.
    fig, ax = plt.subplots(figsize=(9, 3.6))
    idx = np.arange(len(CLASS_NAMES))
    w = 0.28
    for i, (name, y) in enumerate([("Train split", ytr), ("Val split", yva), ("Test (official val file)", yte)]):
        counts = np.bincount(y, minlength=len(CLASS_NAMES))
        ax.bar(idx + (i - 1) * w, counts, w - 0.03, color=plotstyle.SERIES[i], label=name)
    ax.set_xticks(idx, CLASS_NAMES, rotation=0, fontsize=8)
    ax.set_ylabel("Scans")
    ax.set_title("Scans per building-floor class")
    ax.legend(ncols=3, loc="upper left")
    fig.savefig(OUT / "class_counts.png")
    plt.close(fig)

    # 2. Detected APs per scan.
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bins = np.arange(0, 60, 2)
    for i, (name, df) in enumerate([("Training file", train), ("Test (official val file)", test)]):
        ax.hist(detected_counts(df), bins=bins, density=True, histtype="step", linewidth=2,
                color=plotstyle.SERIES[i], label=name)
    ax.set_xlabel("APs detected in one scan")
    ax.set_ylabel("Share of scans")
    ax.set_title("Detected APs per scan")
    ax.legend()
    fig.savefig(OUT / "detected_per_scan.png")
    plt.close(fig)

    # 3. Ranked detection frequency (what the targeted scenario removes first).
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ranks = np.arange(1, len(active_sorted) + 1)
    ax.plot(ranks, active_sorted * 100, color=plotstyle.SERIES[0])
    for p, ls in ((10, ":"), (30, "--"), (50, "-.")):
        r = int(round(p / 100 * len(active_sorted)))
        ax.axvline(r, color=plotstyle.INK_2, linewidth=1, linestyle=ls)
        ax.text(r + 4, ax.get_ylim()[1] * 0.92, f"{p}% targeted", fontsize=8, color=plotstyle.INK_2)
    ax.set_xlabel("AP rank (most detected first, training split)")
    ax.set_ylabel("Scans that detect the AP (%)")
    ax.set_title("AP detection frequency")
    fig.savefig(OUT / "ap_detection_rank.png")
    plt.close(fig)

    # 4. Scans per reference point (why the split is grouped).
    per_rp = pd.Series(reference_point_groups(train)).value_counts()
    summary_rp = describe(per_rp.to_numpy())
    summary["scans_per_reference_point"] = summary_rp
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
