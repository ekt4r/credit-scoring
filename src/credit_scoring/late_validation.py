"""Сравнение CatBoost со стажем и без него на позднем периоде."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_scoring.boosters import make_booster
from credit_scoring.bureau import add_bureau_features
from credit_scoring.feature_screen import DATED_SOURCES, dated_features
from credit_scoring.temporal import CALIBRATION_END, FOLD_BOUNDARIES, ranking_metrics
from credit_scoring.train_baselines import TRAIN_DIR
from credit_scoring.wide_static import EXCLUDED_BY_REASON, load_wide_cases

REVIEW_RATE = 0.10


def compare(target, probability, eligible_count):
    ranked = target.assign(probability=probability).sort_values(
        ["probability", "case_id"], ascending=[False, True],
    )
    flagged = ranked.head(eligible_count)
    found = int(flagged["target"].sum())
    return {
        **ranking_metrics(target, probability),
        "Помечено": len(flagged), "Дефолтов найдено": found,
        "Precision": found / len(flagged), "Recall": found / target["target"].sum(),
    }


def segment_metrics(predictions, groups):
    rows = []
    for key, group in predictions.groupby(groups, sort=True):
        keys = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, keys))
        row.update({"Кейсы": len(group), "Дефолты": int(group["target"].sum())})
        if group["target"].nunique() == 2:
            row["Gini"] = 2 * roc_auc_score(group["target"], group["probability"]) - 1
        rows.append(row)
    return rows


def main():
    excluded = set().union(*EXCLUDED_BY_REASON.values())
    cases, features, categories = load_wide_cases(excluded, evaluation_start=CALIBRATION_END)
    cases, bureau = add_bureau_features(cases.reset_index(), TRAIN_DIR)
    features += bureau
    audit = []
    employment = dated_features(cases, DATED_SOURCES["Заявитель и стаж"], audit)
    cases = cases.join(employment, on="case_id", validate="one_to_one")
    cases["Стаж известен"] = cases["person_1_size"].notna()
    train = cases.loc[cases["WEEK_NUM"] < FOLD_BOUNDARIES[-1]]
    evaluation = cases.loc[cases["WEEK_NUM"] >= CALIBRATION_END]
    assert train["date_decision"].max() < evaluation["date_decision"].min()
    target = evaluation[["case_id", "WEEK_NUM", "target", "Стаж известен"]]
    eligible_count = round(len(target) * REVIEW_RATE)
    print(f"Обучение: {len(train):,}. Проверка: {len(evaluation):,}. "
          f"Дефолты: {target['target'].sum():,}. Лимит проверки: {eligible_count:,}.", flush=True)

    results, segments, weekly = [], [], []
    for name, fields in (("Контроль 120", features),
                         ("120 + заявитель и стаж", [*features, *employment.columns])):
        model = make_booster("CatBoost").set_params(cat_features=categories, used_ram_limit="2gb")
        model.fit(train[fields], train["target"])
        probability = model.predict_proba(evaluation[fields])[:, 1]
        summary = {"Модель": name, **compare(target, probability, eligible_count)}
        results.append(summary)
        predictions = target.assign(probability=probability)
        segments.extend({"Модель": name, **row}
                        for row in segment_metrics(predictions, ["Стаж известен"]))
        weekly.extend({"Модель": name, **row}
                      for row in segment_metrics(predictions, ["WEEK_NUM", "Стаж известен"]))
        print(summary, flush=True)

    output = Path("artifacts/late-validation")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output / "models.csv", index=False)
    pd.DataFrame(segments).to_csv(output / "segments.csv", index=False)
    pd.DataFrame(weekly).to_csv(output / "weekly.csv", index=False)


if __name__ == "__main__":
    main()
