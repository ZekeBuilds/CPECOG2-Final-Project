import numpy as np

from src import demo
from src.data import NOT_DETECTED
from src.outage import OutageSimulator


def raw_scans(n=300, d=520, seed=0):
    rng = np.random.default_rng(seed)
    x = np.full((n, d), NOT_DETECTED, dtype=np.float32)
    detected = rng.random((n, d)) < 0.05
    x[detected] = rng.uniform(-100, -30, detected.sum())
    return x


def test_building_and_floor_handle_label_12():
    labels = np.arange(13)
    assert list(demo.building(labels)) == [0] * 4 + [1] * 4 + [2] * 5
    assert list(demo.floor(labels)) == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4]


def test_error_split_counts():
    y = np.array([0, 0, 4, 8, 12])
    p = np.array([0, 1, 8, 9, 12])  # right, wrong floor, wrong building, wrong floor, right
    assert demo.error_split(y, p) == {"scans": 5, "errors": 3, "wrong_building": 1, "wrong_floor": 2}


def test_deterministic_settings_use_one_seed():
    assert demo.seeds_for("targeted", 0.3, 5) == (0,)
    assert demo.seeds_for("per_scan", 0.0, 5) == (0,)
    assert demo.seeds_for("persistent", 0.3, 3) == (0, 1, 2)


def test_corrupt_is_reproducible_and_matches_evaluate_seeding():
    x = raw_scans()
    sim = OutageSimulator(x)
    a = demo.corrupt(sim, x, "per_scan", 0.3, 2)
    b = demo.corrupt(sim, x, "per_scan", 0.3, 2)
    assert np.array_equal(a, b)
    rng = np.random.default_rng(demo.outage_seed(2, "per_scan", 0.3))
    assert np.array_equal(a, sim.apply(x, "per_scan", 0.3, rng))


def test_figures_build_without_error():
    import matplotlib.pyplot as plt
    import pandas as pd

    x = raw_scans()
    sim = OutageSimulator(x)
    out = demo.corrupt(sim, x, "targeted", 0.3, 0)
    failed = sim.failed_aps("targeted", 0.3)
    df = pd.DataFrame([{"role": r, "rate": rate, "seed": 0, "accuracy": 0.5, "macro_f1": 0.4}
                       for r in ("normal", "apdrop", "knn") for rate in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)])
    figs = [demo.plot_curve(df, "macro_f1", "MLP", "targeted", 0.3),
            demo.plot_ap_strip(x[0], out[0], sim.ranked, len(sim.active), failed),
            demo.plot_buildings(12, [("normal", "MLP", 3), ("apdrop", "MLP + APDropout", 12)]),
            demo.plot_confusion(np.arange(13), np.arange(13), "test")]
    for fig in figs:
        plt.close(fig)
