"""Короткая проверка устойчивости выбранного набора и финальное переобучение."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from credit_scoring.boosters import make_booster
from credit_scoring.bureau import add_bureau_features
from credit_scoring.bureau_payments import add_bureau_payments
from credit_scoring.late_validation import REVIEW_RATE, compare
from credit_scoring.scoring import save_bundle
from credit_scoring.selected_model import (
    SCHEMA_VERSION, feature_schema, load_selected, predict_selected,
)
from credit_scoring.temporal import CALIBRATION_END, FOLD_BOUNDARIES, weekly_gini, stability_penalties
from credit_scoring.train_baselines import TRAIN_DIR
from credit_scoring.wide_static import EXCLUDED_BY_REASON, load_wide_cases

OUTPUT = Path("artifacts/final-checks")


def fit_catboost(train, features, categories):
    model = make_booster("CatBoost").set_params(
        cat_features=[name for name in categories if name in features], used_ram_limit="2gb",
    )
    model.fit(train[features], train["target"])
    return model


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases, features, categories = load_wide_cases(
        set().union(*EXCLUDED_BY_REASON.values()), include_all=True,
    )
    cases, bureau = add_bureau_features(cases.reset_index(), TRAIN_DIR)
    features += bureau
    training = cases["WEEK_NUM"].lt(FOLD_BOUNDARIES[-1])
    validation = cases["WEEK_NUM"].ge(CALIBRATION_END)
    target = cases.loc[validation, ["case_id", "WEEK_NUM", "target"]]
    k = round(len(target) * REVIEW_RATE)
    rows = []

    def record(name, probability):
        trend, variation = stability_penalties(weekly_gini(target, probability))
        row = {"Вариант": name, **compare(target, probability, k),
               "Штраф за тренд": trend, "Штраф за разброс": variation}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT / "comparison.csv", index=False)
        print(row, flush=True)

    control, metadata = load_selected()
    baseline = predict_selected(control, metadata, cases.loc[validation, features])
    record("CatBoost · контроль", baseline)

    without_dates = [name for name in features if not name.endswith("D")
                     and name != "prev_days_since_last"]
    model = fit_catboost(cases.loc[training, [*without_dates, "target"]], without_dates, categories)
    record("CatBoost · без дат и давности",
           model.predict_proba(cases.loc[validation, without_dates])[:, 1])
    del model

    light = make_booster("LightGBM")
    light.fit(cases.loc[training, features], cases.loc[training, "target"])
    probability = light.predict_proba(cases.loc[validation, features])[:, 1]
    record("LightGBM · 120 полей", probability)
    record("CatBoost + LightGBM · 50/50", (baseline + probability) / 2)
    del light

    screened = cases.loc[training | validation]
    extended, block = add_bureau_payments(screened, TRAIN_DIR, by_contract=True)
    del screened
    fields = features + block
    model = fit_catboost(extended.loc[training, [*fields, "target"]], fields, categories)
    record("CatBoost · агрегация по контрактам",
           model.predict_proba(extended.loc[validation, fields])[:, 1])
    del extended, model

    model = fit_catboost(cases, features, categories)
    schema = {"schema_version": SCHEMA_VERSION, "features": feature_schema(features, categories),
              "train_weeks": [int(cases["WEEK_NUM"].min()), int(cases["WEEK_NUM"].max())],
              "training_cases": len(cases)}
    probe = cases[features].iloc[:1024]
    expected = model.predict_proba(probe, thread_count=1)[:, 1]
    with TemporaryDirectory(dir=OUTPUT) as directory:
        staged = Path(directory) / "scoring-120-full.zip"
        save_bundle(model, schema, staged)
        loaded, contract = load_selected(staged)
        np.testing.assert_array_equal(expected, predict_selected(loaded, contract, probe))
        staged.replace(OUTPUT / staged.name)
    print(f"Refit: {len(cases):,} кейсов, {cases['WEEK_NUM'].nunique()} недели. "
          "Совпадение скоров проверено на 1 024 строках.", flush=True)


if __name__ == "__main__":
    main()
