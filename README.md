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
| Streamlit demo (`app.py`) | Built; backup video not recorded yet | ~90% |
| Final written report (IEEE) | Planned for Nov 22 | ~20% |
| Presentation slides | Planned for Nov 22 | 0% |

**Deadlines:** presentation, demo, report and code on **Nov 23**. Final revisions on Dec 4.

## Next steps

Code
- [x] Streamlit demo (`app.py`): pick a model, outage type and outage rate, run the outage simulation on the test scans live, and draw the graphs in the app. Every graph we present must come from the app.
- [x] Same train/validation split for everyone: `load_splits()` regenerates it identically on a fresh clone (checked), so `split.npz` stays out of the repo.
- [x] Pin exact library versions in `requirements.txt`.
- [x] Test the setup in a fresh environment by following this README from scratch (Nov 20: install, download, tests and demo all pass with identical numbers).
- [ ] Record a backup video of the demo in case the live demo fails.

Report (IEEE format)
- [ ] Introduction and problem formulation (reuse the proposal).
- [ ] Review of related literature: a critical review, beyond a summary of each paper.
- [ ] Dataset and preparation: source, license, analysis, cleaning, split, and leakage checks.
- [ ] Model design and selection: candidate models, a comparison table, the APDropout design.
- [ ] Experiments, tuning, and results: hyperparameter register (every setting marked tuned or fixed, with the reason), tuning logs, learning curves, test results, error analysis.
- [ ] Implementation: framework features used, code structure, how to run it.
- [ ] Discussion and conclusions: success criteria, limitations, ethics, SDG 9, future work.
- [ ] Appendix: member contribution record.

Presentation
- [ ] Slides.
- [ ] Every member can explain any part of the project.

Optional
- [ ] Let the autoencoder pretrain past its 100-epoch cap.
- [ ] Try an APDropout rate above 0.9 for the MLP (0.9 was the edge of the grid).

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

You need Python 3.12, 64-bit. On Windows, get "Windows installer (64-bit)" from the [Python 3.12.10 page](https://www.python.org/downloads/release/python-31210/) and tick "Add python.exe to PATH". The main python.org download button gives a newer version, which TensorFlow 2.21 and the saved models were not built with.

**Windows, one command.** From the project folder:

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```

It checks for Python 3.12, creates `.venv`, installs the pinned requirements, downloads the dataset and runs the tests. The first run takes several minutes. Running it again only re-checks, so it is safe to repeat.

**Manual (any OS).** From the project folder. On macOS or Linux, activate with `source .venv/bin/activate`. If Windows PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope Process Bypass` in that window first.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.download_data
python -m pytest -q tests
```

The dataset goes to `data/raw/UJIndoorLoc/`. It is not in the repo.

## Running the demo

On Windows after `setup.ps1`:

```
powershell -ExecutionPolicy Bypass -File run_demo.ps1
```

After a manual setup, with `.venv` activated:

```
streamlit run app.py
```

Then open http://localhost:8501. The first start takes about 30 seconds (it loads the dataset, fits KNN and loads the six saved DNNs). Pick a model family, outage type and share of APs failed in the sidebar. Every number and graph is computed live on the test file:

- **Model comparison:** macro-F1 or accuracy of the normal DNN, the APDropout DNN and KNN at the chosen setting, and the curve from 0 to 50% APs failed.
- **One scan:** the APs one test scan heard, ranked by how often training scans detect them, with the ones the outage switched off in red (APs never heard in training are counted but not drawn); and where each model places the scan.
- **Error analysis:** confusion matrices and wrong-building vs wrong-floor counts.

KNN takes about 2.5 s per run, so a random outage curve with 5 seeds and KNN on takes about a minute the first time. Finished runs are cached. With 5 outage seeds, KNN numbers match the results table exactly. With fewer seeds, random outages (persistent most of all) can differ from the table by several points. DNN numbers come from the seed-0 models only, so they differ from the 5-model averages in the table.

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
  demo.py        logic and figures for the Streamlit demo
  plotstyle.py   shared chart style
scripts/         runnable steps (see the table above)
app.py           Streamlit demo
setup.ps1        one-command Windows setup (Python check, .venv, requirements, dataset, tests)
run_demo.ps1     starts the demo with .venv
tests/           unit tests for the simulator, models and demo helpers
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
