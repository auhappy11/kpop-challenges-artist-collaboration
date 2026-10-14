#!/usr/bin/env python3
"""Generate the matched negative samples in data/negative_samples.csv.

Positives are the collaboration ties in data/dataset.csv.  They are split into
first-time ties (the original artist had never featured this guest before the
upload date) and recurring ties (they had).

For every positive, one negative is drawn:
  * same original artist and same upload date as the positive;
  * the guest is replaced by a different artist from the same guest-activity
    tier (low 40% / medium 30% / high 30% by number of guest appearances) who
    has debuted and released a song by that date;
  * first-time negatives: the original artist had not featured that guest
    before the date; recurring negatives: they had, but not on that date.

Features of a negative are computed exactly as for the positives: the
original artist's song is their latest title track as of the date, the guest's
song is a random released song (seeded).  Each negative records the positive it
is matched to (anchor_short_id, anchor_collaboration_groups); positives without
an eligible negative are later dropped, so classes are balanced.

This script documents how data/negative_samples.csv was produced.  It needs
artist metadata, song audio features, title-track dates, retail-chart album
sales and news-article counts (data/reference/), which are not distributed.

Output: data/negative_samples.csv
Usage : python prediction_modeling/01_generate_negatives.py
"""
import re
from collections import defaultdict

import numpy as np
import pandas as pd

import config as C
from splits import DATE, GUEST, ID, MAIN, load_positives
AUDIO = ["key", "bpm", "energy", "danceability", "happiness", "acousticness",
         "instrumentalness", "liveness", "speechiness", "loudness", "duration_sec"]


# ----------------------------------------------------------------------------
# Feature helpers
# ----------------------------------------------------------------------------
def norm_agency(s):
    """Agency comparison key: case/punctuation-insensitive, 'entertainment' -> 'ent'."""
    s = re.sub(r"[^a-z0-9]", "", str(s).lower())
    return s.replace("entertainment", "ent")


def parse_key(key):
    """'C-sharp Minor' -> ('C#', 'minor'); (None, None) if unparseable."""
    if pd.isna(key):
        return None, None
    m = re.match(r"([A-G](?:[-]?(?:[Ss]harp|[Ff]lat|[#b♯♭])|[#b♯♭])?)\s+([Mm]ajor|[Mm]inor)",
                 key.strip().title())
    if not m:
        return None, None
    note, mode = m.groups()
    note = re.sub(r"[-]?[Ss]harp", "#", note)
    note = re.sub(r"[-]?[Ff]lat", "-", note).replace("b", "-")
    return note, mode.lower()


CIRCLE_OF_FIFTHS = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
                    "G#": 8, "D#": 9, "A#": 10, "F": -1, "B-": -2, "E-": -3,
                    "A-": -4, "D-": -5, "G-": -6}


def music_similarity(key_a, key_b):
    """0.7 x circle-of-fifths closeness + 0.3 x mode match (1.0 same, 0.8 different)."""
    (n1, m1), (n2, m2) = parse_key(key_a), parse_key(key_b)
    if n1 is None or n2 is None or n1 not in CIRCLE_OF_FIFTHS or n2 not in CIRCLE_OF_FIFTHS:
        return 0
    p1, p2 = CIRCLE_OF_FIFTHS[n1], CIRCLE_OF_FIFTHS[n2]
    distance = min((p1 - p2) % 12, (p2 - p1) % 12)
    return 0.7 * (1 - distance / 6) + 0.3 * (1.0 if m1 == m2 else 0.8)


def sales_ratio(main, guest):
    """Guest / original average album sales; 0 if either is unknown."""
    return guest / main if main > 0 and guest > 0 else 0


# ----------------------------------------------------------------------------
# Reference data
# ----------------------------------------------------------------------------
class Reference:
    def __init__(self):
        ref = C.REFERENCE
        a = pd.read_csv(ref / "artists.csv")
        self.gender = dict(zip(a.artist, a.gender))
        self.agency = {k: ("Unknown" if pd.isna(v) else v) for k, v in zip(a.artist, a.agency)}
        self.debut = dict(zip(a.artist, pd.to_datetime(a.debut_date)))
        self.nationalities = {k: set(v.split(";")) for k, v in
                              zip(a.artist, a.non_korean_nationalities) if isinstance(v, str)}

        # Songs per artist, in release order (stable within equal dates).
        s = pd.read_csv(ref / "songs.csv")
        s["release_date"] = pd.to_datetime(s.release_date)
        self.songs = {}
        for artist, g in s.groupby("artist", sort=False):
            rows = [{**{k: r[k] for k in AUDIO}, "is_title_track": bool(r.is_title_track),
                     "release_date": r.release_date} for _, r in g.iterrows()]
            self.songs[artist] = (g.release_date.values, rows)

        t = pd.read_csv(ref / "title_tracks.csv")
        t["release_date"] = pd.to_datetime(t.release_date)
        self.title_tracks = {k.lower(): np.sort(g.release_date.values)
                             for k, g in t.groupby(t.artist.str.strip())}

        # Cumulative chart sales per (artist, album), in first-seen album order.
        sales = pd.read_csv(ref / "album_sales.csv", parse_dates=["chart_date"])
        self.albums = defaultdict(dict)
        for (artist, album), g in sales.groupby(["artist", "album"], sort=False):
            self.albums[artist][album] = (g.chart_date.values, np.cumsum(g.sales.to_numpy(float)))
        self._sales = {}

        n = pd.read_csv(ref / "news_counts.csv", parse_dates=["date"])
        self.news = {}
        for (feature, a1, a2), g in n.groupby(["feature", "artist_1", "artist_2"], sort=False):
            g = g.dropna(subset=["date"]).sort_values("date")
            self.news[(feature, a1, a2)] = (g.date.values, np.cumsum(g.n_articles.to_numpy()))

    def eligible(self, artist, when):
        """Debuted and has at least one released song by `when`."""
        debut = self.debut.get(artist)
        if debut is None or pd.isna(debut) or when < debut:
            return False
        songs = self.songs.get(artist)
        return songs is not None and np.searchsorted(songs[0], np.datetime64(when), "right") > 0

    def latest_title_song(self, artist, when):
        dates, rows = self.songs.get(artist, (None, None))
        if rows is None:
            return None
        n = np.searchsorted(dates, np.datetime64(when), "right")
        return next((r for r in reversed(rows[:n]) if r["is_title_track"]), None)

    def random_song(self, artist, when, rng):
        dates, rows = self.songs.get(artist, (None, None))
        if rows is None:
            return None
        n = np.searchsorted(dates, np.datetime64(when), "right")
        return rows[rng.integers(n)] if n > 0 else None

    def has_recent_title_track(self, artist, when):
        """Artist released a title track within +/- TITLE_TRACK_WINDOW_DAYS of `when`."""
        dates = self.title_tracks.get(str(artist).strip().lower())
        if dates is None:
            return 0
        d = np.datetime64(pd.Timestamp(when).normalize())
        w = np.timedelta64(C.TITLE_TRACK_WINDOW_DAYS, "D")
        return int(((dates >= d - w) & (dates <= d + w)).any())

    def avg_past_album_sales(self, artist, when):
        """Mean cumulative sales, up to `when`, over albums that had charted by `when`."""
        key = (artist, pd.Timestamp(when).normalize())
        if key not in self._sales:
            cut = np.datetime64(key[1])
            per_album = [cum[np.searchsorted(dates, cut, "right") - 1]
                         for dates, cum in self.albums.get(artist, {}).values()
                         if dates[0] <= cut]
            self._sales[key] = float(np.mean(per_album)) if per_album else 0.0
        return self._sales[key]

    def news_before(self, feature, main, guest, when):
        """Articles mentioning both artists published strictly before `when`."""
        for k in [(feature, main, guest), (feature, guest, main)]:
            if k in self.news:
                dates, cum = self.news[k]
                i = np.searchsorted(dates, np.datetime64(when), "left")
                return int(cum[i - 1]) if i > 0 else 0
        return 0


def activity_tiers(counts):
    tiers = pd.qcut(counts, C.TIER_QUANTILES, labels=["low", "medium", "high"]).to_dict()
    by_tier = {t: [a for a in counts.index if tiers.get(a) == t] for t in ["high", "medium", "low"]}
    return tiers, by_tier


# ----------------------------------------------------------------------------
# Negative sampling
# ----------------------------------------------------------------------------
def generate_negatives(positives, kind, ref):
    rng = np.random.default_rng(C.SEED)
    subset = positives[(positives.past_collab == 0) if kind == "first_time"
                       else (positives.past_collab > 0)]

    main_tier, main_by_tier = activity_tiers(positives[MAIN].value_counts())
    guest_tier, guest_by_tier = activity_tiers(positives[GUEST].value_counts())
    main_by_tier = {t: [a for a in v if a in ref.gender] for t, v in main_by_tier.items()}
    guest_by_tier = {t: [a for a in v if a in ref.gender] for t, v in guest_by_tier.items()}

    pair_dates = defaultdict(set)
    for a, b, t in zip(positives[MAIN], positives[GUEST], positives[DATE]):
        pair_dates[(a, b)].add(pd.Timestamp(t).normalize())
    pair_dates = {k: np.sort(pd.to_datetime(pd.Series(list(v))).values) for k, v in pair_dates.items()}

    def valid_pair(main, guest, when):
        """No collaboration on `when`; first-time: none before; recurring: some before."""
        d = pair_dates.get((main, guest))
        w = np.datetime64(when)
        if d is None:
            return kind == "first_time"
        if (d < w).any() != (kind == "recurring"):
            return False
        return not (d == w).any()

    median_debut_gap = positives.debut_date_difference.median()

    def korean_ratio(artist, role_col, ratio_col):
        """Artist's mean Korean-member ratio over their positives in that role."""
        r = positives.loc[positives[role_col] == artist, ratio_col].mean()
        return positives[ratio_col].median() if pd.isna(r) else r

    columns = ["anchor_short_id", "anchor_main", "anchor_guest", MAIN, GUEST, DATE,
               "past_collab", "release_date", "main_key", "collab_key", *C.FEATURES]
    negatives, pair_counts = [], defaultdict(int)
    for anchor_index in rng.permutation(len(subset)):
        anchor = subset.iloc[int(anchor_index)]
        main, when = anchor[MAIN], pd.Timestamp(anchor[DATE])
        main_pool = [a for a in main_by_tier[main_tier[main]] if ref.eligible(a, when)]
        guest_pool = [a for a in guest_by_tier[guest_tier[anchor[GUEST]]] if ref.eligible(a, when)]
        if not main_pool or not guest_pool or main not in ref.gender:
            continue
        candidates = [g for g in guest_pool
                      if g != main and g != anchor[GUEST] and valid_pair(main, g, when)]
        if not candidates:
            continue
        guest = candidates[rng.integers(len(candidates))]
        if pair_counts[(main, guest, when)] >= C.MAX_NEGATIVES_PER_PAIR_DATE:
            continue
        pair_counts[(main, guest, when)] += 1

        main_song = ref.latest_title_song(main, when)
        guest_song = ref.random_song(guest, when, rng)
        if main_song is None or guest_song is None:
            continue

        row = pd.Series(index=columns, dtype="object")
        row[["anchor_short_id", "anchor_main", "anchor_guest"]] = [anchor[ID], main, anchor[GUEST]]
        row[[MAIN, GUEST, DATE]] = [main, guest, when]
        d = pair_dates.get((main, guest))
        row["past_collab"] = int((d < np.datetime64(when)).sum()) if d is not None else 0
        for prefix, song in [("main", main_song), ("collab", guest_song)]:
            for k in AUDIO:
                if f"{prefix}_{k}" in columns and pd.notna(song[k]):
                    row[f"{prefix}_{k}"] = song[k]
        row["release_date"] = main_song["release_date"]
        row["music_similarity"] = music_similarity(main_song["key"], guest_song["key"])

        row["same_gender"] = int(ref.gender.get(main) == ref.gender.get(guest))
        row["same_agency"] = int(norm_agency(ref.agency.get(main, "Unknown")) ==
                                 norm_agency(ref.agency.get(guest, "Unknown")))
        d1, d2 = ref.debut.get(main), ref.debut.get(guest)
        row["debut_date_difference"] = (abs((d1 - d2).days) if pd.notna(d1) and pd.notna(d2)
                                        else median_debut_gap)
        row["has_recent_title_track"] = ref.has_recent_title_track(guest, row["release_date"])
        row["past_engagement"] = ref.news_before("past_engagement", main, guest, when)
        row["past_engagement_raw"] = ref.news_before("past_engagement_raw", main, guest, when)
        row["main_korean_member_ratio"] = korean_ratio(main, MAIN, "main_korean_member_ratio")
        row["collab_korean_member_ratio"] = korean_ratio(guest, GUEST, "collab_korean_member_ratio")
        n1, n2 = ref.nationalities.get(main, set()), ref.nationalities.get(guest, set())
        row["shared_non_korean_nationality"] = int(bool(n1 & n2))
        s1, s2 = ref.avg_past_album_sales(main, when), ref.avg_past_album_sales(guest, when)
        row["avg_past_album_sales_main"], row["avg_past_album_sales_collab"] = s1, s2
        row["ratio_avg_album_sales"] = sales_ratio(s1, s2)
        negatives.append(row)

    neg = pd.DataFrame(negatives)
    # Songs with a missing audio value: fill with the most common negative value,
    # then recompute music_similarity from the final keys.
    for col in neg.columns[neg.isna().any()]:
        fill = (neg[col].median() if pd.api.types.is_numeric_dtype(neg[col])
                else neg[col].mode()[0])
        neg[col] = neg[col].fillna(fill)
    neg["music_similarity"] = [float(music_similarity(a, b))
                               for a, b in zip(neg.main_key, neg.collab_key)]
    return subset, neg


def main():
    ref = Reference()
    positives = load_positives()
    out = []
    for kind in C.KINDS:
        _, neg = generate_negatives(positives, kind, ref)
        neg = neg.rename(columns={"anchor_guest": "anchor_collaboration_groups"})
        neg.insert(0, "collab_type", kind)
        neg[DATE] = pd.to_datetime(neg[DATE]).dt.strftime("%Y-%m-%d")
        out.append(neg[["collab_type", "anchor_short_id", "anchor_collaboration_groups",
                        MAIN, GUEST, DATE, *C.FEATURES]])
        print(f"{kind}: {len(neg)} negatives")
    pd.concat(out).to_csv(C.NEGATIVES, index=False)


if __name__ == "__main__":
    main()
