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
| **MS** – Music Similarity | `music_similarity` |
| **OAS** – Original Artist Song | `main_bpm`, `main_energy`, `main_danceability`, `main_happiness`, `main_acousticness`, `main_instrumentalness`, `main_liveness`, `main_speechiness`, `main_loudness` |
| **GAS** – Guest Artist Song | `collab_bpm`, `collab_energy`, `collab_danceability`, `collab_happiness`, `collab_acousticness`, `collab_instrumentalness`, `collab_liveness`, `collab_speechiness`, `collab_loudness` |
| **SP** – Social Proximity | `same_gender`, `same_agency`, `past_engagement`, `past_engagement_raw`, `debut_date_difference`, `main_korean_member_ratio`, `collab_korean_member_ratio`, `shared_non_korean_nationality` |
| **TP** – Temporal Proximity | `has_recent_title_track` |
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
