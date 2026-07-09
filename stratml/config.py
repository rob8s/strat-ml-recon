"""Central configuration for strat-ml-recon.

Single source of truth for data locations and the constants/hyperparameters that
the driver/training scripts use -- changing them changes results, so edit with care.

The experimental data is *not* committed to this repo. Point ``DATA_ROOT`` at the
folder that holds the ``.mat`` cubes and intermediate CSVs before running any
pipeline script.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
# The TDB17-1 dataset folder: holds the raw ``.mat`` cubes and is where the
# pipeline writes its final CSV. Edit this to wherever your data lives.
DATA_ROOT = Path("/home/rob8s/Geology/Datasets/tdb-17-1")

# Where trained models and prediction plots are written (anchored to the repo
# root, so outputs land there regardless of the working directory).
OUTPUT_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = OUTPUT_ROOT / "saved_models"
PLOTS_DIR = OUTPUT_ROOT / "y_pred_plots"

# Canonical processed dataset used by the trainers and the plotting script.
LAYER_STATS_CSV = DATA_ROOT / "final_data" / "Layer_Stats_Env_Tagged.csv"

# --------------------------------------------------------------------------- #
# Normalisation / filtering used in the driver & training scripts
# (constants baked inside the pure preprocessing functions stay inline there)
# --------------------------------------------------------------------------- #
THICKNESS_SCALE = 6.5
TIME_SCALE = 115.0
MIN_THICKNESS = 0.065

# --------------------------------------------------------------------------- #
# Model features / targets
# --------------------------------------------------------------------------- #
TARGETS = ["Total_Dep", "Total_Time", "Stasis_Proportion", "Deposition_Proportion"]

# Single-model (all-data) feature sets.
FEATURES_ALL = ["Layer_Thickness", "Layer_Time", "Lobe", "Channel",
                "Wet_Floodplain", "Dry_Floodplain", "Marine"]
FEATURES_ALL_TAGGED = FEATURES_ALL + ["High_Erosion"]

# Sub-sample sizes for the all-data trainers.
SAMPLE_N_ALL = 100_000
SAMPLE_N_TAGGED = 200_000

RANDOM_STATE = 42

# --------------------------------------------------------------------------- #
# Random-forest hyperparameters (tuned)
# --------------------------------------------------------------------------- #
RF_PARAMS_ALL = {
    "n_estimators": 154,
    "max_depth": 18,
    "min_samples_split": 13,
    "min_samples_leaf": 19,
    "max_features": None,
}
