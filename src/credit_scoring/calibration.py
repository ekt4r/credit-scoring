"""Калибровка выбранного по временной валидации бустинга."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_scoring.boosters import BOOSTERS, make_booster
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import FEATURES
from credit_scoring.temporal import CALIBRATION_END, FOLD_BOUNDARIES, ranking_metrics
from credit_scoring.train_baselines import TRAIN_DIR, load_cases, score


def select_candidate(paths):
    results = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected_folds = set(zip(FOLD_BOUNDARIES[:-1], (end - 1 for end in FOLD_BOUNDARIES[1:])))
    for name, group in results.groupby("Вариант"):
        observed = set(zip(group["Начало"], group["Конец"]))
        if len(group) != len(expected_folds) or observed != expected_folds:
            raise ValueError(f"Неполные или повторяющиеся результаты для {name}.")
    means = results.groupby("Вариант")["Gini Stability"].mean().sort_values(ascending=False)
    print("Средняя Gini Stability по фолдам\n", means.to_string(), flush=True)
    model_name, feature_set = means.idxmax().split(" · ")
    if model_name not in BOOSTERS or feature_set not in {"static", "history"}:
        raise ValueError(f"Неизвестный вариант: {means.idxmax()}.")
    return model_name, feature_set


def logit(probability):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability)).reshape(-1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="CSV с временной валидацией.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/calibration.csv"))
    args = parser.parse_args()
    model_name, feature_set = select_candidate(args.results)
    cases = load_cases()
    features = FEATURES
    if feature_set == "history":
        cases = add_application_history(cases, TRAIN_DIR)
        features = FEATURES + HISTORY_FEATURES

    weeks = cases["WEEK_NUM"]
    train = cases.loc[weeks < FOLD_BOUNDARIES[-1]]
    calibration = cases.loc[weeks.between(FOLD_BOUNDARIES[-1], CALIBRATION_END, inclusive="left")]
    evaluation = cases.loc[weeks >= CALIBRATION_END]
    assert train["date_decision"].max() < calibration["date_decision"].min()
    assert calibration["date_decision"].max() < evaluation["date_decision"].min()

    print(f"Модель: {model_name} · {feature_set}. Кейсы: {len(train):,} обучение, "
          f"{len(calibration):,} калибровка, {len(evaluation):,} проверка.", flush=True)
    model = make_booster(model_name)
    model.fit(train[features], train["target"])
    calibration_scores = model.predict_proba(calibration[features])[:, 1]
    calibrator = LogisticRegression().fit(logit(calibration_scores), calibration["target"])

    print(f"Коэффициент калибровки: {calibrator.coef_[0, 0]:.4f}", flush=True)

    raw = model.predict_proba(evaluation[features])[:, 1]
    calibrated = calibrator.predict_proba(logit(raw))[:, 1]
    print(f"Доля дефолтов на проверке: {evaluation['target'].mean():.4%}", flush=True)
    rows = {
        label: {**score(evaluation["target"], probability),
                **ranking_metrics(evaluation, probability)}
        for label, probability in (("До калибровки", raw), ("После калибровки", calibrated))
    }
    report = pd.DataFrame.from_dict(rows, orient="index").rename_axis("Режим")
    print(report.to_string(float_format="{:.6f}".format))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output)


if __name__ == "__main__":
    main()
