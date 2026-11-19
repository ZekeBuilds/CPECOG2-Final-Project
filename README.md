# Floor-Level Wi-Fi Localization Under Targeted Access Point Outages

CPECOG2 (Deep Neural Networks) design project, Group 6.

Phones can tell which building and floor they are on from the Wi-Fi access points (APs) they hear. When APs break, that guess gets worse. We test how badly, and whether a model trained with simulated AP failures (our custom **APDropout** layer) holds up better.

We focus on **targeted outages**: the APs that phones hear most often fail first. Earlier work ([me-nwa/robust-wifi-indoor-localization](https://github.com/me-nwa/robust-wifi-indoor-localization)) covers random AP loss only.

- **Task:** classify each Wi-Fi scan into 1 of 13 building-floor classes.
- **Data:** [UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc) (UCI, CC BY 4.0). 19,937 training scans, 1,111 test scans, 520 APs.
- **Models:** KNN baseline, MLP, autoencoder + MLP, 1D-CNN. Each DNN is trained normally and with APDropout.
- **Outages tested:** random per-scan, random persistent, and targeted, each at 0 to 50% of APs failed.

## Status (Nov 20, 2026)

| Part | Status | Done |
|---|---|---|
| Data pipeline and analysis | Built and tested | 100% |
| Model code (KNN, MLP, AE + MLP, 1D-CNN, APDropout layer) | Built and tested | 100% |
| Training and hyperparameter tuning | All runs done, 5 seeds | ~95% |
| Testing and error analysis | Done on the test file | 100% |
| Streamlit demo | Planned for Nov 21 | 0% |
| Final written report (IEEE) | Planned for Nov 22 | ~20% |
| Presentation slides | Planned for Nov 22 | 0% |

**Deadlines:** presentation, demo, report and code on **Nov 23**. Final revisions on Dec 4.

## Results so far

Test macro-F1 (%), mean of 5 seeds. Every number comes from `results/summary/test_table.csv`.

| Model | Clean | 30% random per-scan | 30% random persistent | 30% targeted |
|---|---|---|---|---|
| KNN | 85.4 | 66.9 | 64.9 | 13.6 |
| MLP | 91.1 | 78.8 | 78.5 | 22.3 |
| MLP + APDropout | 91.5 | 87.3 | 87.0 | 42.0 |
| AE + MLP | 91.2 | 80.2 | 80.1 | 25.6 |
| AE + MLP + APDropout | 89.0 | 85.9 | 84.7 | 42.1 |
| 1D-CNN | 90.9 | 79.1 | 79.1 | 22.1 |
| 1D-CNN + APDropout | 90.3 | 86.3 | 86.2 | 40.0 |

- Targeted outage is much harder than random outage.
- APDropout adds 16 to 20 points at 30% targeted outage for every model. On clean input it costs 0.6 points or less for the MLP and 1D-CNN, and 2.2 points for AE + MLP.
- The choice of architecture matters much less than how the model is trained.

Main figure: `results/summary/test_macro_f1_bars.png`.

## Setup

Python 3.12. From the project folder:

```
pip install -r requirements.txt
python -m scripts.download_data
python -m pytest -q tests
```

The dataset goes to `data/raw/UJIndoorLoc/`. It is not in the repo.

## Running the experiments

Run every command from the project root. Results are saved to `results/`, and trained models to `models/`. You only need to rerun these if you change the code, because the results and the seed-0 models are already in the repo.

| Command | What it does | Time on a laptop |
|---|---|---|
| `python -m scripts.eda` | Data analysis and figures (`results/eda/`) | < 1 min |
| `python -m scripts.run_baselines` | Majority-class and KNN baselines (`results/baselines/`) | 10+ min |
| `python -m scripts.train_dnn mlp all` | Tune and train one DNN family; also `ae` or `cnn` (`results/<family>/`) | 15 to 30 min each |
| `python -m scripts.summarize` | Cross-model table and bar chart (`results/summary/`) | < 1 min |
| `python -m scripts.analyze` | Confusion matrices, error analysis, Assumption 2 check (`results/analysis/`) | ~3 min |

`train_dnn` has four stages that can run one at a time: `arch`, `hparams`, `apdrop`, `final`. The test file is used only in `final`. Every tuning choice is made on the validation split.

## Repo layout

```
src/
  data.py        load, clean, label, location-grouped train/val split, scaling
  outage.py      AP outage simulator (per-scan, persistent, targeted)
  models.py      APDropout layer, MLP, autoencoder + MLP, 1D-CNN
  evaluate.py    shared outage evaluation and plots
  plotstyle.py   shared chart style
scripts/         runnable steps (see the table above)
tests/           unit tests for the simulator and models
results/         generated tables, logs and figures
models/          trained seed-0 models (.keras), used by the demo
```

Load a saved model with `import src.models` first, so Keras knows the custom APDropout layer:

```python
import keras
import src.models  # registers APDropout
model = keras.models.load_model("models/mlp_apdrop_seed0.keras")
```

## Team

- Charles Andrew Petisme
- Ezekiel John Libao
- Gavino Rafael Fuerte
- Rilen Gawat
