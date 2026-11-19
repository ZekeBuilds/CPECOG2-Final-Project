"""Loading, cleaning, labelling and splitting UJIIndoorLoc.

Conventions used across the project:
- Raw RSSI matrices keep the dataset's own code for "not detected": +100.
- Outages (src/outage.py) are applied to raw matrices, then `to_model_input` scales them.
- Class label = BUILDINGID * 4 + FLOOR. Buildings 0 and 1 have floors 0-3, building 2 has
  floors 0-4, so labels run 0..12 with no gaps (13 classes).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "UJIndoorLoc"
PROCESSED_DIR = ROOT / "data" / "processed"

N_APS = 520
NOT_DETECTED = 100
# Chosen floor for scaling: below the weakest reading in the data (-104 dBm), so an absent AP
# maps to exactly 0 and the weakest real reading stays above 0.
RSSI_FLOOR = -110
SPLIT_SEED = 42
N_FOLDS = 5  # one fold held out -> 80/20 train/val by reference point

AP_COLS = [f"WAP{i:03d}" for i in range(1, N_APS + 1)]
CLASS_NAMES = [f"B{b}-F{f}" for b in range(3) for f in range(5) if not (b < 2 and f == 4)]


def load_raw():
    """Return (train_df, test_df). The official validation file is our test set."""
    train = pd.read_csv(RAW_DIR / "trainingData.csv")
    test = pd.read_csv(RAW_DIR / "validationData.csv")
    return train, test


def labels(df):
    return (df["BUILDINGID"].to_numpy() * 4 + df["FLOOR"].to_numpy()).astype(np.int64)


def rssi(df):
    return df[AP_COLS].to_numpy(dtype=np.float32)


def clean_train(df):
    """Drop exact duplicate rows and scans with no detected AP. Returns (df, report)."""
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)
    detected = (df[AP_COLS].to_numpy() != NOT_DETECTED).sum(axis=1)
    df = df[detected > 0]
    n_empty = n0 - n_dup - len(df)
    return df.reset_index(drop=True), {"rows_in": n0, "exact_duplicates": n_dup,
                                        "no_ap_detected": n_empty, "rows_out": len(df)}


def reference_point_groups(df):
    """Scans taken at the same spot share BUILDINGID, FLOOR, SPACEID and RELATIVEPOSITION."""
    keys = df[["BUILDINGID", "FLOOR", "SPACEID", "RELATIVEPOSITION"]].astype(str).agg("-".join, axis=1)
    return pd.factorize(keys)[0]


def grouped_split(df, seed=SPLIT_SEED):
    """Train/val indices with no reference point on both sides, stratified by class."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf.split(df, labels(df), reference_point_groups(df)))
    return np.sort(train_idx), np.sort(val_idx)


def to_model_input(x_raw):
    """Scale raw RSSI to [0, 1]: absent -> 0, -110 dBm -> 0, 0 dBm -> 1."""
    x = np.where(x_raw == NOT_DETECTED, RSSI_FLOOR, x_raw).astype(np.float32)
    return np.clip((x - RSSI_FLOOR) / -RSSI_FLOOR, 0.0, 1.0)


def load_splits(cache=True):
    """Everything later stages need, with the split cached so every model uses the same one.

    Returns dict with raw RSSI arrays (x_*_raw), labels (y_*), and the cleaning report.
    """
    train_df, test_df = load_raw()
    train_df, report = clean_train(train_df)
    split_file = PROCESSED_DIR / "split.npz"
    groups = reference_point_groups(train_df)
    if cache and split_file.exists():
        s = np.load(split_file)
        tr, va = s["train_idx"], s["val_idx"]
        if (len(tr) + len(va) != len(train_df) or max(tr.max(), va.max()) >= len(train_df)
                or set(groups[tr]) & set(groups[va])):
            raise RuntimeError("Cached split does not match the cleaned data; delete data/processed/split.npz")
    else:
        tr, va = grouped_split(train_df)
        if cache:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            np.savez(split_file, train_idx=tr, val_idx=va)
    x_all, y_all = rssi(train_df), labels(train_df)
    return {
        "x_train_raw": x_all[tr], "y_train": y_all[tr],
        "x_val_raw": x_all[va], "y_val": y_all[va],
        "x_test_raw": rssi(test_df), "y_test": labels(test_df),
        "groups_train": groups[tr],
        "groups_val": groups[va],
        "clean_report": report,
    }
