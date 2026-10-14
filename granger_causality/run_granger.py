#!/usr/bin/env python3
"""Release-anchored Toda-Yamamoto Granger tests between daily chart performance
and daily collaboration activity, per song.

Chart score = 201 - Circle digital chart rank (0 when off chart).  Activity
predictors: collaboration Shorts uploaded, collaboration Shorts featuring a
top-tier guest (top 20% of the 207 study artists by retail album sales up to the
day before release), and retrieved comments and replies on those Shorts.

Each song is analysed over the first W days from release (release day included);
songs without a complete W-day window are counted but not tested.  Primary
specification: W = 60, BIC max lag 7.  Sensitivities: W = 30 / 90, max lag 3 / 14.
Benjamini-Hochberg FDR (q < .05) is applied separately per predictor, direction,
window, sample and max lag.  Two samples are reported: all complete-window songs
(non-estimable tests enter the correction with p = 1) and songs whose model is
estimable and passes both residual diagnostics.

Usage: python granger_causality/run_granger.py
"""
from pathlib import Path

import pandas as pd
from statsmodels.stats.multitest import multipletests

from granger_ty import toda_yamamoto

DATA = Path(__file__).resolve().parents[1] / "data" / "granger"
PREDICTORS = {"collab_shorts": "Collaboration Shorts",
              "top_tier_collab_shorts": "Top-tier guest Shorts",
              "comments": "Comments"}
WINDOWS = [60, 30, 90]
MAX_LAGS = [7, 3, 14]
DIRECTIONS = {"x_to_rank": "activity -> chart", "rank_to_x": "chart -> activity"}


def run_tests(songs, daily, window, max_lag):
    rows = []
    for song in songs.song_id:
        g = daily.get(song)
        complete = g is not None and len(g) >= window
        for predictor in PREDICTORS:
            row = {"song_id": song, "predictor": predictor, "complete_window": complete}
            if complete:
                w = g.iloc[:window]
                row.update(toda_yamamoto(w.rank_score.to_numpy(), w[predictor].to_numpy(), max_lag))
            else:
                row.update(estimable=False, diagnostics_pass=False, p_x_to_rank=None, p_rank_to_x=None)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(models):
    out = []
    for sample in ["window_coverage", "diagnostic_passed"]:
        for predictor in PREDICTORS:
            all_rows = models[models.predictor.eq(predictor)]
            family = all_rows[all_rows.complete_window]
            if sample == "diagnostic_passed":
                family = family[family.estimable & family.diagnostics_pass]
            row = {"sample": sample, "predictor": predictor,
                   "candidate_songs": len(all_rows), "complete_window": int(all_rows.complete_window.sum()),
                   "family_size": len(family), "estimable": int(family.estimable.sum()),
                   "diagnostics_pass": int((family.estimable & family.diagnostics_pass).sum())}
            for d in DIRECTIONS:
                p = family[f"p_{d}"].astype(float)
                q = multipletests(p.fillna(1), method="fdr_bh")[1] if len(p) else []
                row[f"raw_{d}"] = int(p.lt(.05).sum())
                row[f"fdr_{d}"] = int((p.notna() & (pd.Series(q, index=p.index) < .05)).sum())
            out.append(row)
    return pd.DataFrame(out)


def main():
    songs = pd.read_csv(DATA / "songs.csv")
    daily = pd.read_csv(DATA / "daily_series.csv", parse_dates=["date"])
    daily = {s: g.sort_values("date").reset_index(drop=True) for s, g in daily.groupby("song_id")}
    pd.set_option("display.width", 200)
    results = []
    for max_lag in MAX_LAGS:
        for window in WINDOWS:
            summary = summarize(run_tests(songs, daily, window, max_lag))
            summary.insert(0, "window", window)
            summary.insert(0, "max_lag", max_lag)
            results.append(summary)
            label = "PRIMARY" if (window, max_lag) == (60, 7) else "sensitivity"
            print(f"\n=== {window}-day window, BIC max lag {max_lag} ({label}) ===")
            print(summary.drop(columns=["max_lag", "window"]).to_string(index=False))
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    main()
