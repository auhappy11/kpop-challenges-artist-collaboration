#!/usr/bin/env python3
"""Tune, fit and explain the first-time and recurring random-forest models.

For each model, a 36-candidate grid is searched with 4 expanding-window folds
over the training rows' unique upload dates (no date is split between folds).
The candidate with the highest mean validation ROC-AUC is refit on all training
rows and evaluated once on the chronological test set (threshold 0.5).
Median imputation and scaling are fitted inside each fold.

SHAP: TreeSHAP values for the positive class are computed on the test rows.
The forest is fitted on imputed, standardized features, so SHAP is computed in
that space; raw feature values are used only to colour the beeswarm plot.

Training and test rows: positives (data/dataset.csv) that have a matched
negative plus the negatives (data/negative_samples.csv), sorted by upload date
and split 80/20 (splits.py).

Input : data/dataset.csv, data/negative_samples.csv
Output: prediction_modeling/outputs/metrics.csv                 performance (both models)
        prediction_modeling/outputs/<kind>/best_parameters.json
        prediction_modeling/outputs/<kind>/feature_importance.csv  (impurity-based)
        prediction_modeling/outputs/<kind>/shap_feature_importance.csv
        prediction_modeling/outputs/<kind>/shap_beeswarm.png
Usage : python prediction_modeling/02_train_grid_search.py [--jobs 4]
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C  # noqa: E402
from splits import LABEL, load_negatives, load_positives, make_split  # noqa: E402

SHAP_MAX_DISPLAY = 20
CATEGORY = {f: c for c, features in C.CATEGORIES.items() for f in features}


def date_folds(train):
    """Expanding-window folds over sorted unique dates."""
    dates = pd.to_datetime(train.published_at).dt.normalize()
    assert dates.is_monotonic_increasing
    unique = np.sort(dates.unique())
    folds = []
    for a, b in TimeSeriesSplit(n_splits=C.CV_FOLDS).split(unique):
        tr = np.flatnonzero(dates.isin(unique[a]))
        va = np.flatnonzero(dates.isin(unique[b]))
        assert dates.iloc[tr].max() < dates.iloc[va].min()
        folds.append((tr, va))
    return folds


def scores(model, frame):
    y = frame[LABEL].astype(int)
    prob = model.predict_proba(frame[C.FEATURES])[:, 1]
    pred = model.predict(frame[C.FEATURES])
    return {"accuracy": accuracy_score(y, pred), "auc": roc_auc_score(y, prob),
            "ap": average_precision_score(y, prob)}


def explain(kind, model, test, out):
    """SHAP table and beeswarm plot for the test rows."""
    rf = model.named_steps["rf"]
    x_raw = test[C.FEATURES].astype(float)
    x_model = pd.DataFrame(model[:-1].transform(x_raw), columns=C.FEATURES)
    explainer = shap.TreeExplainer(rf)
    values = np.asarray(explainer.shap_values(x_model, check_additivity=False))
    values = values[:, :, 1] if values.ndim == 3 else values
    base = float(np.ravel(explainer.expected_value)[-1])
    assert np.abs(base + values.sum(axis=1) - rf.predict_proba(x_model)[:, 1]).max() < 1e-6

    pd.DataFrame({"feature": C.FEATURES, "category": [CATEGORY[f] for f in C.FEATURES],
                  "mean_abs_shap": np.abs(values).mean(axis=0), "mean_shap": values.mean(axis=0)}
                 ).sort_values("mean_abs_shap", ascending=False).to_csv(
                     out / "shap_feature_importance.csv", index=False)
    plt.figure()
    shap.plots.beeswarm(shap.Explanation(values=values, base_values=np.full(len(values), base),
                                         data=x_raw.fillna(x_raw.median()).values,
                                         feature_names=C.FEATURES),
                        max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"SHAP summary — {kind.replace('_', '-')} collaborations", fontsize=13)
    plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()


def fit(kind, positives, negatives, jobs):
    train, test, dropped = make_split(positives, negatives, kind)
    print(f"{kind}: {dropped} positives without a matched negative dropped; "
          f"train {len(train)}, test {len(test)}")
    model = Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()),
                      ("rf", RandomForestClassifier(**C.RF_FIXED))])
    search = GridSearchCV(model, C.GRID, cv=date_folds(train), refit="auc",
                          scoring={"auc": "roc_auc", "ap": "average_precision", "accuracy": "accuracy"},
                          n_jobs=jobs, error_score="raise", return_train_score=True)
    with joblib.parallel_backend("threading", n_jobs=jobs):
        search.fit(train[C.FEATURES], train[LABEL].astype(int))
    best = search.best_estimator_

    out = C.OUTPUTS / kind
    out.mkdir(parents=True, exist_ok=True)
    params = {k.removeprefix("rf__"): v for k, v in search.best_params_.items()}
    (out / "best_parameters.json").write_text(json.dumps({**C.RF_FIXED, **params}, indent=2) + "\n")
    pd.DataFrame({"feature": C.FEATURES, "importance": best.named_steps["rf"].feature_importances_}
                 ).sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)

    explain(kind, best, test, out)

    train_scores = scores(best, train)
    test_scores = scores(best, test)
    return {"model": kind, "n_train": len(train), "n_test": len(test),
            "cv_auc": search.best_score_,
            **{f"train_{k}": v for k, v in train_scores.items()},
            **{f"test_{k}": v for k, v in test_scores.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    positives, negatives = load_positives(), load_negatives()
    results = pd.DataFrame([fit(kind, positives, negatives, args.jobs) for kind in C.KINDS])
    results.to_csv(C.OUTPUTS / "metrics.csv", index=False)
    print(results[["model", "cv_auc", "test_auc", "test_ap", "test_accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
