"""Shared paths and settings for the collaboration prediction models."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
DATASET = DATA / "dataset.csv"                 # positives
NEGATIVES = DATA / "negative_samples.csv"      # matched negatives (01_generate_negatives.py)
# Artist, song, chart-sales and news lookup tables used only to generate the
# negatives; not distributed with the repository.
REFERENCE = DATA / "reference"
OUTPUTS = REPO / "prediction_modeling" / "outputs"      # results (not tracked)

SEED = 42
KINDS = ["first_time", "recurring"]

# Negative sampling ----------------------------------------------------------
# Each negative is anchored to one positive: same original artist, same upload
# date, guest replaced by another artist from the same guest-activity tier.
TIER_QUANTILES = [0, 0.4, 0.7, 1.0]          # low 40% / medium 30% / high 30%
TITLE_TRACK_WINDOW_DAYS = 14                 # has_recent_title_track: +/- days
MAX_NEGATIVES_PER_PAIR_DATE = 3
TRAIN_FRACTION = 0.8                         # chronological split

# Model features (31), in the order used by the models -----------------------
FEATURES = [
    # MS - music similarity
    "music_similarity",
    # OAS - original artist song audio
    "main_bpm", "main_energy", "main_danceability", "main_happiness",
    "main_acousticness", "main_instrumentalness", "main_liveness",
    "main_speechiness", "main_loudness",
    # GAS - guest artist song audio
    "collab_bpm", "collab_energy", "collab_danceability", "collab_happiness",
    "collab_acousticness", "collab_instrumentalness", "collab_liveness",
    "collab_speechiness", "collab_loudness",
    # SP - social / relational
    "same_gender", "same_agency", "past_engagement", "past_engagement_raw",
    "debut_date_difference",
    # TP - title-track promotion
    "has_recent_title_track",
    # SP - social / relational (cultural)
    "main_korean_member_ratio", "collab_korean_member_ratio",
    "shared_non_korean_nationality",
    # P - popularity
    "avg_past_album_sales_main", "avg_past_album_sales_collab",
    "ratio_avg_album_sales",
]

CATEGORIES = {
    "MS": ["music_similarity"],
    "OAS": [f for f in FEATURES if f.startswith("main_") and "ratio" not in f],
    "GAS": [f for f in FEATURES if f.startswith("collab_") and "ratio" not in f],
    "SP": ["same_gender", "same_agency", "past_engagement", "past_engagement_raw",
           "debut_date_difference", "main_korean_member_ratio",
           "collab_korean_member_ratio", "shared_non_korean_nationality"],
    "TP": ["has_recent_title_track"],
    "P": ["avg_past_album_sales_main", "avg_past_album_sales_collab",
          "ratio_avg_album_sales"],
}

# Random forest grid search ----------------------------------------------------
GRID = {"rf__n_estimators": [200, 300],
        "rf__max_depth": [1, 3, 5],
        "rf__min_samples_split": [20, 30],
        "rf__min_samples_leaf": [15, 20, 25],
        "rf__max_features": ["sqrt"]}
RF_FIXED = {"bootstrap": True, "class_weight": "balanced", "random_state": SEED, "n_jobs": 1}
CV_FOLDS = 4
