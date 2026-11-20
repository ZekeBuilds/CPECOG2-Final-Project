"""Group 6 live demo: floor classification under simulated Wi-Fi AP outages.

Run from the project root:  streamlit run app.py
Every number and graph here is computed live from the saved seed-0 models and the test file.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import demo
from src.data import CLASS_NAMES, NOT_DETECTED, ROOT, load_splits, to_model_input
from src.outage import RATES, SCENARIOS, OutageSimulator
from src.plotstyle import SCENARIO_LABEL

st.set_page_config(page_title="AP outage demo, Group 6", layout="wide")
st.markdown(
    """<style>
    /* The font stylesheet is loaded by headingFont in .streamlit/config.toml. */
    h1, h2, h3 { font-family: "Source Serif 4", Georgia, serif !important; letter-spacing: 0; }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"], [data-testid="stDataFrame"] {
        font-variant-numeric: tabular-nums;
    }
    </style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- cached loading

@st.cache_resource(show_spinner=False)
def load_assets():
    from sklearn.neighbors import KNeighborsClassifier

    s = load_splits()
    sim = OutageSimulator(s["x_train_raw"])
    # KNN settings picked on clean val by scripts/run_baselines.py.
    best = json.loads((ROOT / "results" / "baselines" / "knn_tuning.json").read_text())["best"]
    knn = KNeighborsClassifier(n_neighbors=best["k"], metric=best["metric"], weights=best["weights"], n_jobs=-1)
    knn.fit(to_model_input(s["x_train_raw"]), s["y_train"])
    return s, sim, knn


@st.cache_resource(show_spinner=False)
def load_model(key):
    import keras
    import src.models  # noqa: F401  registers APDropout before load_model

    return keras.models.load_model(demo.MODELS_DIR / f"{key}_seed0.keras")


def predictor(key):
    if key == "knn":
        knn = load_assets()[2]
        return lambda x_raw: knn.predict(to_model_input(x_raw))
    return demo.dnn_predictor(load_model(key))


@st.cache_data(show_spinner=False, max_entries=2000)
def run_point(key, scenario, rate, seed):
    s, sim, _ = load_assets()
    return demo.evaluate_point(predictor(key), sim, s["x_test_raw"], s["y_test"], scenario, rate, seed)


def model_keys(family_key, with_knn):
    keys = {"normal": family_key, "apdrop": f"{family_key}_apdrop"}
    if with_knn:
        keys["knn"] = "knn"
    return keys


def run_curve(keys, scenario, n_seeds):
    jobs = [(role, key, rate, seed) for rate in RATES for role, key in keys.items()
            for seed in demo.seeds_for(scenario, rate, n_seeds)]
    bar = st.progress(0.0, text="Running the outage simulation on the test scans")
    rows = []
    for i, (role, key, rate, seed) in enumerate(jobs):
        r = run_point(key, scenario, rate, seed)
        rows.append({"role": role, "rate": rate, "seed": seed,
                     "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]})
        bar.progress((i + 1) / len(jobs), text=f"Simulated {i + 1} of {len(jobs)} runs")
    bar.empty()
    return pd.DataFrame(rows)


def show(fig):
    # clear_figure only calls fig.clf(); pyplot keeps the figure open unless it is closed.
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- page

with st.spinner("Loading the dataset, KNN and the six DNNs. The first start takes about 30 seconds."):
    try:
        s, sim, _ = load_assets()
    except FileNotFoundError:
        st.error("Dataset not found. Run `python -m scripts.download_data` from the project root, then reload.")
        st.stop()
    try:
        for k in demo.FAMILIES.values():
            load_model(k)
            load_model(f"{k}_apdrop")
    except (FileNotFoundError, ValueError) as err:
        st.error(f"Could not load a saved model from `models/`: {err}")
        st.stop()

with st.sidebar:
    st.header("Settings")
    family = st.radio("Model family", list(demo.FAMILIES))
    scenario = st.radio("Outage type", SCENARIOS, index=SCENARIOS.index("targeted"),
                        format_func=SCENARIO_LABEL.get,
                        captions=[demo.SCENARIO_HELP[sc] for sc in SCENARIOS])
    rate = st.select_slider("APs failed", options=RATES, value=0.3, format_func=lambda r: f"{r:.0%}")
    metric = st.radio("Metric", ["macro_f1", "accuracy"],
                      format_func={"macro_f1": "Macro-F1", "accuracy": "Accuracy"}.get, horizontal=True)
    n_seeds = st.slider("Outage seeds", 1, 5, 1,
                        help="Random outages are repeated with this many seeds and averaged. "
                             "Targeted outages and clean input are the same for every seed.")
    with_knn = st.checkbox("Include KNN baseline", value=True,
                           help="About 2.5 s per run on a laptop. Finished runs are cached.")

names = demo.role_names(family)
keys = model_keys(demo.FAMILIES[family], with_knn)
metric_name = "Macro-F1" if metric == "macro_f1" else "Accuracy"

st.title("Floor classification under Wi-Fi AP outages")
st.caption(f"UJIIndoorLoc test file: {len(s['y_test']):,} scans from a different user and phones than "
           f"training, 520 APs, {len(CLASS_NAMES)} building-floor classes.")

tab_curve, tab_scan, tab_errors = st.tabs(["Model comparison", "One scan", "Error analysis"])

with tab_curve:
    df = run_curve(keys, scenario, n_seeds)
    at = df[df["rate"] == rate].groupby("role")[metric].mean() * 100
    clean = df[df["rate"] == 0.0].groupby("role")[metric].mean() * 100
    st.subheader(f"{metric_name} at {rate:.0%} {SCENARIO_LABEL[scenario].lower()} outage")
    readout, chart = st.columns([1, 3], gap="large")
    with readout:
        for role in keys:
            st.metric(names[role], f"{at[role]:.1f}", f"{at[role] - clean[role]:+.1f} vs clean")
        n_used = len(demo.seeds_for(scenario, rate, n_seeds))
        st.markdown(f"APDropout gain: **{at['apdrop'] - at['normal']:+.1f} points** "
                    f"({n_used} outage seed{'s' if n_used > 1 else ''})")
    with chart:
        show(demo.plot_curve(df, metric, family, scenario, rate))
        st.caption("DNN numbers come from the seed-0 trained model; the report table averages five "
                   "trained models. With 5 outage seeds, KNN matches the table exactly. With fewer seeds, "
                   "random outages (persistent most of all) can differ from the table by several points. "
                   "Shaded bands are ±1 std over outage seeds.")

with tab_scan:
    n_test = len(s["y_test"])
    if "scan" not in st.session_state:
        # Start on the first scan that hears at least the median number of APs (scan 0 hears one).
        heard = (s["x_test_raw"] != NOT_DETECTED).sum(axis=1)
        st.session_state.scan = int(np.argmax(heard >= np.median(heard)))
    c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
    if c2.button("Pick a random scan"):
        st.session_state.scan = int(np.random.default_rng().integers(n_test))
    idx = c1.number_input("Test scan", 0, n_test - 1, key="scan")

    x_out = demo.corrupt(sim, s["x_test_raw"], scenario, rate, 0)
    clean_row, out_row = s["x_test_raw"][idx], x_out[idx]
    network_failed = None
    if scenario != "per_scan" and rate > 0:
        network_failed = sim.failed_aps(scenario, rate,
                                        np.random.default_rng(demo.outage_seed(0, scenario, rate)))

    st.subheader(f"Scan {idx} at {rate:.0%} {SCENARIO_LABEL[scenario].lower()} outage")
    show(demo.plot_ap_strip(clean_row, out_row, sim.ranked, len(sim.active), network_failed))

    y_true = int(s["y_test"][idx])
    rows, marks = [], []
    for role, key in keys.items():
        if key == "knn":
            pred = int(predictor("knn")(out_row[None, :])[0])
            conf = None
        else:
            p = demo.dnn_probabilities(load_model(key), out_row[None, :])[0]
            pred, conf = int(p.argmax()), float(p.max())
        marks.append((role, names[role], pred))
        rows.append({"Model": names[role], "Prediction": CLASS_NAMES[pred],
                     "Confidence": "1-NN, no score" if conf is None else f"{conf:.0%}",
                     "Correct": "yes" if pred == y_true else "no"})
    c3, c4 = st.columns([3, 2])
    with c3:
        show(demo.plot_buildings(y_true, marks))
    with c4:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tab_errors:
    st.subheader(f"Where the errors go at {rate:.0%} {SCENARIO_LABEL[scenario].lower()} outage")
    st.caption("Outage seed 0. Each row of a matrix sums to 100% of that true class.")
    y = s["y_test"]
    split_rows = []
    cols = st.columns(2)
    for col, role in zip(cols, ("normal", "apdrop")):
        pred = run_point(keys[role], scenario, rate, 0)["pred"]
        with col:
            show(demo.plot_confusion(y, pred, names[role]))
        split_rows.append({"Model": names[role], **demo.error_split(y, pred)})
    if with_knn:
        split_rows.append({"Model": "KNN", **demo.error_split(y, run_point("knn", scenario, rate, 0)["pred"])})
    table = pd.DataFrame(split_rows).rename(columns={
        "scans": "Scans", "errors": "Errors", "wrong_building": "Wrong building",
        "wrong_floor": "Right building, wrong floor"})
    st.dataframe(table, hide_index=True, width="stretch")
