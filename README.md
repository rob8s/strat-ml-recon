# `strat-ml-recon`

> Note: this repo is part of research accepted, pending edits, to **[The Sedimentary Record](https://www.sepm.org/the-sedimentary-record)**. Paper located **[here](https://thesedimentaryrecord.scholasticahq.com/article/160567-recovering-geomorphic-history-from-stratigraphy-application-of-machine-learning-techniques-to-an-experimental-basin)**.

This repo contains code for transforming fluviodeltaic experimental data and applying ML methodologies to reconstruct vacuities. This code lays the groundwork for conclusions discussed in the above paper, while creating a foundation for authors to continue work in this area.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the `stratml` package + its dependencies
```

Installing the package (editable mode) is what makes the `stratml` imports resolve;
the runnable pipeline scripts live in `scripts/`.

> Requirements: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`,
> `opencv-python`, `joblib` (see `requirements.txt` / `pyproject.toml`).

---

## Data

In collaboration with Kyle Straub at Tulane University's Sediment Dynamics Laboratory, fluviodeltaic experimental data (TDB17-1) was provided for use in this repo. Experimental data is represented as a 3D tensor, with horizontal coordinates representing location and the vertical axis representing time steps of the experiment.

The experimental data is **not committed** to this repo. Point `DATA_ROOT` in
[`stratml/config.py`](stratml/config.py) at the folder holding the `.mat` cubes and
intermediate CSVs before running any pipeline script.

---

## Organization

```
stratml/                  importable package (all reusable logic)
├── config.py             DATA_ROOT + scaling factors + hyperparameters (single source of truth)
├── io.py                 load_mat_data()
├── preprocessing/        pure transforms: flatten, error_check, strat, layer_stats, tag
└── modeling/             shared RF train/eval core + scatter plotting

scripts/                  runnable entry points (each `python -m scripts.<name>`)
├── preprocess_env.py            env-based strat extraction (incl. zeroing) -> Layer_Stats_Env_Tagged.csv
├── train_all_data.py            single model, all envs
├── train_all_data_tagged.py     single model + High_Erosion feature
└── plot_true_pred.py            true-vs-predicted vacuity plots
```

Run a stage with, e.g. `python -m scripts.train_all_data` from the repo root.