"""Positives, negatives and the chronological train/test split."""
from collections import defaultdict

import pandas as pd

import config as C

ID, MAIN, GUEST, DATE = "short_id", "main_artist_group", "collaboration_groups", "published_at"
LABEL = "collaboration_occurred"


def load_positives():
    """Collaboration ties with past_collab = earlier upload dates of the same ordered pair."""
    df = pd.read_csv(C.DATASET)
    df[DATE] = pd.to_datetime(df[DATE])
    day = df[DATE].dt.normalize()
    pairs = list(zip(df[MAIN], df[GUEST]))
    seen = defaultdict(set)
    for pair, t in zip(pairs, day):
        seen[pair].add(t)
    df["past_collab"] = [sum(x < t for x in seen[p]) for p, t in zip(pairs, day)]
    df["collab_type"] = ["first_time" if n == 0 else "recurring" for n in df.past_collab]
    return df


def load_negatives():
    df = pd.read_csv(C.NEGATIVES)
    df[DATE] = pd.to_datetime(df[DATE])
    return df


def make_split(positives, negatives, kind):
    """Keep positives that have a matched negative, sort by date, split 80/20."""
    pos = positives[positives.collab_type.eq(kind)]
    neg = negatives[negatives.collab_type.eq(kind)]
    key = [ID, MAIN, GUEST]
    anchors = neg[["anchor_short_id", MAIN, "anchor_collaboration_groups"]].set_axis(key, axis=1)
    assert not anchors.duplicated().any()
    matched = pos[key].merge(anchors.assign(_m=1), on=key, how="left")["_m"].notna().to_numpy()
    df = pd.concat([pos[matched].assign(**{LABEL: 1}), neg.assign(**{LABEL: 0})])
    df = df.sort_values(DATE).reset_index(drop=True)
    cut = int(len(df) * C.TRAIN_FRACTION)
    return df.iloc[:cut], df.iloc[cut:], int((~matched).sum())
