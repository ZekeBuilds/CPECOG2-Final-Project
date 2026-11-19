import numpy as np

from src.data import NOT_DETECTED, to_model_input
from src.outage import OutageSimulator, drop_per_scan

ND = NOT_DETECTED


def toy():
    # 4 scans x 6 APs; AP5 never detected. Detection counts: AP0=4, AP1=3, AP2=2, AP3=1, AP4=1.
    return np.array([
        [-50, -60, -70, -80, ND, ND],
        [-55, -65, -75, ND, ND, ND],
        [-52, -62, ND, ND, -90, ND],
        [-51, ND, ND, ND, ND, ND],
    ], dtype=np.float32)


def test_targeted_removes_most_detected_first():
    sim = OutageSimulator(toy())
    assert list(sim.active) == [0, 1, 2, 3, 4]
    assert list(sim.failed_aps("targeted", 0.4)) == [0, 1]  # round(0.4 * 5) = 2


def test_persistent_only_picks_active_aps_and_is_seeded():
    sim = OutageSimulator(toy())
    a = sim.failed_aps("persistent", 0.6, rng=3)
    assert len(a) == 3 and set(a) <= {0, 1, 2, 3, 4}
    assert list(a) == list(sim.failed_aps("persistent", 0.6, rng=3))


def test_per_scan_drops_round_rate_times_detected():
    x = toy()
    out = drop_per_scan(x, 0.5, rng=0)
    before = (x != ND).sum(1)
    after = (out != ND).sum(1)
    assert list(before - after) == list(np.rint(0.5 * before).astype(int))
    # Surviving readings are unchanged; nothing new appears.
    kept = out != ND
    assert np.array_equal(out[kept], x[kept])


def test_rate_zero_is_identity_and_input_not_mutated():
    x = toy()
    sim = OutageSimulator(x)
    for sc in ("per_scan", "persistent", "targeted"):
        assert np.array_equal(sim.apply(x, sc, 0.0), x)
        sim.apply(x, sc, 0.5, rng=1)
    assert np.array_equal(x, toy())


def test_scaling_maps_absent_to_zero():
    z = to_model_input(np.array([[ND, -110, -55, 0]], dtype=np.float32))
    assert np.allclose(z, [[0, 0, 0.5, 1]])


def test_outage_seed_is_stable_for_subsets():
    from src.evaluate import outage_seed
    # Matches the original positional seeding for full runs, so saved KNN results stay valid.
    assert outage_seed(2, "targeted", 0.3) == [2, 2, 3]
    assert outage_seed(0, "per_scan", 0.0) == [0, 0, 0]
