"""Охват дефолтов при фиксированном объёме проверки."""

from pathlib import Path

import numpy as np
import pandas as pd

from credit_scoring.bureau import add_bureau_features
from credit_scoring.selected_model import load_selected, predict_selected
from credit_scoring.temporal import CALIBRATION_END, weekly_gini
from credit_scoring.train_baselines import TRAIN_DIR
from credit_scoring.wide_static import EXCLUDED_BY_REASON, load_wide_cases

CURVE_PATH = Path("artifacts/review-curve.csv")


def main():
    model, metadata = load_selected()
    cases, _, _ = load_wide_cases(
        set().union(*EXCLUDED_BY_REASON.values()), evaluation_start=CALIBRATION_END,
    )
    evaluation = cases.loc[cases["WEEK_NUM"].ge(CALIBRATION_END)].reset_index()
    del cases
    evaluation, _ = add_bureau_features(evaluation, TRAIN_DIR)
    scores = predict_selected(model, metadata, evaluation)
    ranked = evaluation.assign(score=scores).sort_values(
        ["score", "case_id"], ascending=[False, True],
    )
    total = len(ranked)
    defaults = int(ranked["target"].sum())
    counts = np.rint(np.linspace(0, total, 101)).astype(int)
    found = np.r_[0, ranked["target"].to_numpy().cumsum()][counts]
    curve = pd.DataFrame({
        "Доля проверки, %": np.arange(101),
        "Помечено": counts, "Найдено дефолтов": found,
        "Recall": found / defaults, "Случайный отбор": counts / total,
    })
    if int(found[10]) != metadata["metrics"]["Дефолтов найдено"]:
        raise ValueError("Кривая не соответствует сохранённым метрикам модели.")
    curve.to_csv(CURVE_PATH, index=False)
    weekly_gini(evaluation, scores).rename("Gini").rename_axis("Неделя").to_csv(
        CURVE_PATH.with_name("selected-weekly-gini.csv")
    )


if __name__ == "__main__":
    main()
