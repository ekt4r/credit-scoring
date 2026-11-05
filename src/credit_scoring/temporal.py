"""Последовательная оценка моделей по неделям принятия решений."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring.boosters import BOOSTERS, make_booster
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import FEATURES
from credit_scoring.train_baselines import TRAIN_DIR, load_cases

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


def weekly_gini(validation, probability):
    frame = validation[["WEEK_NUM", "target"]].assign(probability=probability)
    return pd.Series({
        week: 2 * roc_auc_score(group["target"], group["probability"]) - 1
        for week, group in frame.groupby("WEEK_NUM")
    })


def stability_penalties(weekly):
    slope, intercept = np.polyfit(weekly.index, weekly, deg=1)
    residuals = weekly - (slope * weekly.index + intercept)
    return 88 * min(0, slope), -0.5 * residuals.std(ddof=0)


def ranking_metrics(validation, probability, weekly=None):
    if weekly is None:
        weekly = weekly_gini(validation, probability)
    trend, variation = stability_penalties(weekly)
    return {
        "Gini": 2 * roc_auc_score(validation["target"], probability) - 1,
        "Средняя недельная Gini": weekly.mean(),
        "Gini Stability": weekly.mean() + trend + variation,
    }


def evaluate(cases, name, features, make_model, *, coverage_column=None, weekly_rows=None):
    rows = []
    for start, end, train, validation in temporal_folds(cases):
        model = make_model()
        model.fit(train[features], train["target"])
        probability = model.predict_proba(validation[features])[:, 1]
        weekly = weekly_gini(validation, probability)
        row = {"Вариант": name, "Начало": start, "Конец": end - 1,
               **ranking_metrics(validation, probability, weekly)}
        if coverage_column is not None:
            covered = validation[coverage_column].to_numpy(dtype=bool)
            row["Покрытие"] = covered.mean()
            for label, mask in (("С бюро", covered), ("Без бюро", ~covered)):
                target = validation["target"].to_numpy()[mask]
                if np.unique(target).size == 2:
                    row[f"Gini · {label}"] = 2 * roc_auc_score(target, probability[mask]) - 1
        if weekly_rows is not None:
            trend, variation = stability_penalties(weekly)
            row.update({"Штраф за тренд": trend, "Штраф за разброс": variation})
            coverage = (validation.groupby("WEEK_NUM")[coverage_column].mean()
                        if coverage_column is not None else None)
            for week, gini in weekly.items():
                week_row = {"Неделя": week, "Gini": gini}
                if coverage is not None:
                    week_row["Покрытие"] = coverage[week]
                weekly_rows.append(week_row)
        rows.append(row)
        print(row, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/temporal.csv"))
    parser.add_argument("--history", action="store_true", help="Добавить историю предыдущих заявок.")
    parser.add_argument("--model", choices=BOOSTERS, default="CatBoost")
    args = parser.parse_args()
    cases = load_cases()
    features = FEATURES
    feature_set = "static"
    if args.history:
        cases = add_application_history(cases, TRAIN_DIR)
        features = FEATURES + HISTORY_FEATURES
        feature_set = "history"
    rows = evaluate(cases, f"{args.model} · {feature_set}", features,
                    lambda: make_booster(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
