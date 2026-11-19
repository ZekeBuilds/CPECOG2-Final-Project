"""Majority-class and KNN baselines, clean and under all outage scenarios.

KNN hyperparameters are picked on the clean validation split only. Writes results/baselines/.
Run from the project root:  python -m scripts.run_baselines
"""
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.data import ROOT, load_splits, to_model_input
from src.evaluate import outage_curve, plot_curves, scores, summarize
from src.outage import OutageSimulator

OUT = ROOT / "results" / "baselines"
K_GRID = (1, 3, 5, 7, 9, 11, 15)
METRICS = ("euclidean", "manhattan")
WEIGHTS = ("uniform", "distance")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    x_tr = to_model_input(s["x_train_raw"])
    x_va = to_model_input(s["x_val_raw"])

    # Tune KNN on the clean validation split.
    tuning = []
    for k, metric, weights in itertools.product(K_GRID, METRICS, WEIGHTS):
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1).fit(x_tr, s["y_train"])
        tuning.append({"k": k, "metric": metric, "weights": weights, **scores(s["y_val"], knn.predict(x_va))})
    # Rank by val macro-F1, then accuracy; ties go to the smaller k (simpler model).
    best = max(tuning, key=lambda r: (round(r["macro_f1"], 4), round(r["accuracy"], 4), -r["k"]))
    knn = KNeighborsClassifier(n_neighbors=best["k"], metric=best["metric"], weights=best["weights"],
                               n_jobs=-1).fit(x_tr, s["y_train"])

    majority = int(np.bincount(s["y_train"]).argmax())
    sim = OutageSimulator(s["x_train_raw"])
    predictors = {
        "Majority class": lambda xr: np.full(len(xr), majority),
        "KNN": lambda xr: knn.predict(to_model_input(xr)),
    }
    frames = []
    for name, fn in predictors.items():
        for split in ("val", "test"):
            frames.append(outage_curve(fn, sim, s[f"x_{split}_raw"], s[f"y_{split}"], name, split))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "outage_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / "outage_summary.csv", index=False)
    (OUT / "knn_tuning.json").write_text(json.dumps({"best": best, "grid": tuning}, indent=2))

    plot_curves(df[(df.split == "test") & (df.model == "KNN")], "accuracy",
                "KNN under AP outages, test set (mean ± std, 5 seeds)", OUT / "knn_test_accuracy.png")
    plot_curves(df[(df.split == "val") & (df.model == "KNN")], "accuracy",
                "KNN under AP outages, val split (mean ± std, 5 seeds)", OUT / "knn_val_accuracy.png")

    print("Best KNN on val:", best)
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
