# `strat-ml-recon`

> Note: this repo is part of research accepted, pending edits, to **[The Sedimentary Record](https://www.sepm.org/the-sedimentary-record)**. Paper located here.

This repo contains code for transforming fluviodeltaic experimental data and applying ML methodologies to reconstruct vacuities. This code lays the groundwork for conclusions discussed in the above paper, while creating a foundation for authors to continue work in this area.

---

## Requirements

Base requirements:
- `Python 3.x`
- `Numpy` + `Pandas` + `Pickle` + `Seaborn` + `Matplotlib`
- `Scipy` + `CV2`
- `scikit-learn`

> Individual scripts and notebooks may require additional supporting libraries.

---

## Organization

The `strat-ml-recon` module contains the following submodules:

| Submodule | Description |
|---|---|
| `layer_stats` | Model setups, dataset preprocessing, and residual plotting |

---

## Data

In collaboration with Kyle Straub at Tulane University's Sediment Dynamics Laboratory, fluviodeltaic experimental data (TDB17-1) was provided for use in this repo. Experimental data is represented as a 3D tensor, with horizontal coordinates representing location and the vertical axis representing time steps of the experiment.