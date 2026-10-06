# Artist-to-Artist challenge collaboration Shorts Dataset

File: `data/dataset.csv`

| | |
|---|---|
| Rows | 6,008 directed collaboration ties (original artist → guest artist in one Short) |
| Date Range | 2022-01-01 to 2024-10-29 |
| Artists | 207 (186 original, 206 guest) |
| Songs | 597 |
| Columns | 35 (3 identifiers + 32 model features) |

## Model features (32), by category

| Category | Features |
|---|---|
| **MS** – Music similarity | `music_similarity` |
| **OAS** – Original artist song audio | `main_bpm`, `main_energy`, `main_danceability`, `main_happiness`, `main_acousticness`, `main_instrumentalness`, `main_liveness`, `main_speechiness`, `main_loudness` |
| **GAS** – Guest artist song audio | `collab_bpm`, `collab_energy`, `collab_danceability`, `collab_happiness`, `collab_acousticness`, `collab_instrumentalness`, `collab_liveness`, `collab_speechiness`, `collab_loudness` |
| **SP** – Social / relational | `same_gender`, `same_agency`, `past_engagement`, `past_engagement_raw`, `debut_date_difference`, `main_korean_member_ratio`, `collab_korean_member_ratio`, `shared_non_korean_nationality` |
| **TP** – Title-track promotion | `has_recent_title_track` |
| **P** – Popularity | `avg_past_album_sales_main`, `avg_past_album_sales_collab`, `ratio_avg_album_sales`, `temporal_shorts_count` |

## Identifier columns

| Column | Description |
|---|---|
| `main_artist_group` | Original artist whose song the Short features |
| `collaboration_groups` | Guest artist appearing in the Short |
| `published_at` | Upload date of the Short |

Column order: the 3 identifiers, then the 32 features in category order above
(35 columns). Different Shorts can share the same artist pair, date and
feature values, so 224 rows are exact duplicates of another row; they are
distinct Shorts, not data errors.
