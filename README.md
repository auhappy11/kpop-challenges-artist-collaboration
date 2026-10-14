# Artist-to-Artist challenge collaboration Shorts Dataset

File: `data/dataset.csv`

| | |
|---|---|
| Rows | 6,008 directed collaboration ties (original artist → guest artist in one Short) |
| Shorts | 5,940 (66 Shorts feature more than one guest artist) |
| Date Range | 2022-01-01 to 2024-10-29 |
| Artists | 207 (186 original, 206 guest) |
| Songs | 597 |
| Columns | 35 (4 identifiers + 31 model features) |

## Model features (31), by category

| Category | Features |
|---|---|
| **MS** – Music similarity | `music_similarity` |
| **OAS** – Original artist song audio | `main_bpm`, `main_energy`, `main_danceability`, `main_happiness`, `main_acousticness`, `main_instrumentalness`, `main_liveness`, `main_speechiness`, `main_loudness` |
| **GAS** – Guest artist song audio | `collab_bpm`, `collab_energy`, `collab_danceability`, `collab_happiness`, `collab_acousticness`, `collab_instrumentalness`, `collab_liveness`, `collab_speechiness`, `collab_loudness` |
| **SP** – Social / relational | `same_gender`, `same_agency`, `past_engagement`, `past_engagement_raw`, `debut_date_difference`, `main_korean_member_ratio`, `collab_korean_member_ratio`, `shared_non_korean_nationality` |
| **TP** – Title-track promotion | `has_recent_title_track` |
| **P** – Popularity | `avg_past_album_sales_main`, `avg_past_album_sales_collab`, `ratio_avg_album_sales` |

## Identifier columns

| Column | Description |
|---|---|
| `short_id` | Anonymized identifier of the Short (`short_00001` … `short_05940`). A Short with several guest artists has one row per guest, all sharing the same `short_id`; (`short_id`, `collaboration_groups`) is unique. |
| `main_artist_group` | Original artist whose song the Short features |
| `collaboration_groups` | Guest artist appearing in the Short |
| `published_at` | Upload date of the Short |

Column order: the 4 identifiers, then the 31 features in category order above
(35 columns). Different Shorts can share the same artist pair, date and
feature values, so some rows differ only in `short_id`; they are distinct
Shorts, not data errors.

## Negative samples (`data/negative_samples.csv`)

5,949 matched negatives (4,508 first-time, 1,441 recurring): artist pairs that
did not collaborate. Each negative is anchored to one positive: same original
artist and upload date, with the guest replaced by another artist from the same
guest-activity tier (low 40% / medium 30% / high 30%) who had debuted and
released a song by that date. First-time negatives had never been featured by
the original artist before that date; recurring negatives had, but not on it.
Features are computed exactly as for the positives.

| Column | Description |
|---|---|
| `collab_type` | `first_time` or `recurring` |
| `anchor_short_id`, `anchor_collaboration_groups` | The positive tie this negative is matched to |
| `main_artist_group`, `collaboration_groups`, `published_at` | Original artist, replacement guest, date |
| 31 features | Same columns as in `dataset.csv` |

59 positives (13 first-time, 46 recurring) had no eligible replacement guest
and are excluded from the models. Rows are sorted by upload date and split
80/20 into train and test (`prediction_modeling/splits.py`).
`prediction_modeling/01_generate_negatives.py` documents how the negatives were
generated; it uses artist metadata, song audio features, title-track dates,
retail-chart album sales and news-article counts that are not distributed.

## Granger data (`data/granger/`)

| File | Contents |
|---|---|
| `songs.csv` | The 224 candidate songs (title, original artist, release date) |
| `daily_series.csv` | Per song and day from release (up to 90 days): chart score (201 − Circle digital chart rank, 0 when off chart), collaboration Shorts, top-tier guest collaboration Shorts, and retrieved comments and replies |

## Code

```bash
pip install -r requirements.txt   # Python 3.11
```

| Analysis | Command | What it does |
|---|---|---|
| Network construction | `python network_modeling/01_network_construction.py` | Yearly directed weighted networks (2022, 2023, 2024) from `dataset.csv`; motif-analysis inputs |
| Motif analysis | `bash network_modeling/run_motif_analysis.sh` | Compiles `02_motif_analysis.c` (C, igraph 0.10) and counts three-node directed motifs for 2022, 2023 and 2024 against 5,000 degree-preserving randomizations (10 × edges switches); FANMOD-style Z-scores |
| Negative sampling | `prediction_modeling/01_generate_negatives.py` | How `data/negative_samples.csv` was generated (needs undistributed source data) |
| Prediction models | `python prediction_modeling/02_train_grid_search.py` | Chronological 80/20 split; first-time and recurring random forests, date-grouped CV grid search; TreeSHAP on each test set |
| Granger causality | `python granger_causality/run_granger.py` | Toda–Yamamoto tests; prints the summary tables (60-day primary; 30/90-day and max lag 3/14 sensitivities) |

`02_train_grid_search.py` writes its results to `prediction_modeling/outputs/`:

| File | Contents |
|---|---|
| `metrics.csv` | Training/test accuracy, ROC AUC and average precision, and CV AUC, for both models |
| `<model>/best_parameters.json` | Selected random-forest hyperparameters |
| `<model>/feature_importance.csv` | Impurity-based feature importance |
| `<model>/shap_feature_importance.csv` | Mean absolute and mean SHAP value per feature, with category |
| `<model>/shap_beeswarm.png` | SHAP beeswarm plot (top 20 features) |

`<model>` is `first_time` or `recurring`. Generated files
(`network_modeling/networks/`, `prediction_modeling/outputs/`) are not tracked.
