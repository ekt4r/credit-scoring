"""Последовательная оценка моделей по неделям принятия решений."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring.scoring import FEATURES
from credit_scoring.train_baselines import build_models, load_cases

FOLD_BOUNDARIES = (24, 34, 44, 55)
CALIBRATION_END = 73


def temporal_folds(cases):
    weeks = cases["WEEK_NUM"]
    for start, end in zip(FOLD_BOUNDARIES, FOLD_BOUNDARIES[1:]):
        train = cases.loc[weeks < start]
        validation = cases.loc[weeks.between(start, end, inclusive="left")]
        if train.empty or validation.empty or train["date_decision"].max() >= validation["date_decision"].min():
            raise ValueError(f"Некорректное временное разбиение для недель {start}–{end - 1}.")
        yield start, end, train, validation


def ranking_metrics(validation, probability):
    frame = validation[["WEEK_NUM", "target"]].assign(probability=probability)
    weekly = pd.Series({
        week: 2 * roc_auc_score(group["target"], group["probability"]) - 1
        for week, group in frame.groupby("WEEK_NUM")
    })
    slope, intercept = np.polyfit(weekly.index, weekly, deg=1)
    residuals = weekly - (slope * weekly.index + intercept)
    return {
        "Gini": 2 * roc_auc_score(frame["target"], probability) - 1,
        "Средняя недельная Gini": weekly.mean(),
        "Gini Stability": weekly.mean() + 88 * min(0, slope) - 0.5 * residuals.std(ddof=0),
    }


def evaluate(cases, name, features, make_model):
    rows = []
    for start, end, train, validation in temporal_folds(cases):
        model = make_model()
        model.fit(train[features], train["target"])
        probability = model.predict_proba(validation[features])[:, 1]
        row = {"Вариант": name, "Начало": start, "Конец": end - 1,
               **ranking_metrics(validation, probability)}
        rows.append(row)
        print(row, flush=True)
    return rows


def catboost():
    return build_models()["CatBoost"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal.csv"))
    args = parser.parse_args()
    rows = evaluate(load_cases(), "CatBoost · static", FEATURES, catboost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
