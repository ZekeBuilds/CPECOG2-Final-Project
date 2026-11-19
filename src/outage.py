"""AP outage simulator. All functions take raw RSSI (+100 = not detected) and return a copy.

Rates are fractions. Scenarios:
- per_scan:   each scan independently loses round(rate * k) of the k APs it detected.
- persistent: one random set of round(rate * n_active) active APs is off for every scan.
- targeted:   the round(rate * n_active) most frequently detected APs are off for every scan.

"Active" APs and detection frequencies come from the training split only, so the ranking never
sees validation or test data.
"""
import numpy as np

from .data import NOT_DETECTED

SCENARIOS = ("per_scan", "persistent", "targeted")
RATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def detection_frequency(x_train_raw):
    """Fraction of training scans in which each AP was detected."""
    return (x_train_raw != NOT_DETECTED).mean(axis=0)


class OutageSimulator:
    def __init__(self, x_train_raw):
        self.freq = detection_frequency(x_train_raw)
        self.active = np.flatnonzero(self.freq > 0)
        # Most-detected first; stable sort so ties break by AP index, keeping the ranking reproducible.
        order = np.argsort(-self.freq[self.active], kind="stable")
        self.ranked = self.active[order]

    def failed_aps(self, scenario, rate, rng=None):
        """AP indices switched off for a whole-network scenario (persistent or targeted)."""
        n = int(round(rate * len(self.active)))
        if scenario == "targeted":
            return self.ranked[:n]
        if scenario == "persistent":
            rng = np.random.default_rng(rng)
            return rng.choice(self.active, size=n, replace=False)
        raise ValueError(f"{scenario!r} has no fixed AP set")

    def apply(self, x_raw, scenario, rate, rng=None):
        x = np.array(x_raw, copy=True)
        if rate <= 0:
            return x
        if scenario == "per_scan":
            return drop_per_scan(x, rate, rng)
        x[:, self.failed_aps(scenario, rate, rng)] = NOT_DETECTED
        return x


def drop_per_scan(x_raw, rate, rng=None):
    """Each row loses round(rate * k) of its k detected APs, chosen uniformly at random."""
    rng = np.random.default_rng(rng)
    x = np.array(x_raw, copy=True)
    detected = x != NOT_DETECTED
    n_drop = np.rint(rate * detected.sum(axis=1)).astype(int)
    # Random score per detected AP, +inf for absent ones; the n_drop lowest scores in a row fail.
    scores = np.where(detected, rng.random(x.shape), np.inf)
    rank = np.argsort(np.argsort(scores, axis=1), axis=1)
    x[rank < n_drop[:, None]] = NOT_DETECTED
    return x
