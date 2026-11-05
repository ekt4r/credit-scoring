"""Сравнение моделей на размеченных заявках Home Credit."""

from pathlib import Path
from time import perf_counter
import warnings

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.scoring import (
    ARTIFACT_PATH, CATEGORICAL, FEATURES, NUMERIC, RISK_THRESHOLD,
    prepare_features, save_bundle,
)

TRAIN_DIR = Path("data/raw/parquet_files/train")



def load_cases():
    base = pd.read_parquet(TRAIN_DIR / "train_base.parquet",
                           columns=["case_id", "WEEK_NUM", "date_decision", "target"])
    paths = sorted(TRAIN_DIR.glob("train_static_0_*.parquet"))
    static = pd.concat(
        [pd.read_parquet(path, columns=["case_id", *FEATURES]) for path in paths],
        ignore_index=True,
    )
    cases = base.merge(static, on="case_id", how="left", validate="one_to_one")
    cases[FEATURES] = prepare_features(cases)
    return cases


def build_models():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL),
    ])
    return {
        "LogReg": Pipeline([
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=500)),
        ]),
        "CatBoost": CatBoostClassifier(
            iterations=350, depth=6, learning_rate=0.08, cat_features=CATEGORICAL,
            random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
        ),
    }


def score(target, probability):
    precision, recall, _ = precision_recall_curve(target, probability)
    flagged = probability >= RISK_THRESHOLD
    roc_auc = roc_auc_score(target, probability)
    return {
        "ROC-AUC": roc_auc,
        "Gini": 2 * roc_auc - 1,
        "Precision": precision_score(target, flagged, zero_division=0),
        "Recall": recall_score(target, flagged),
        "Помечено": int(flagged.sum()),
        "Найдено дефолтов": int((flagged & (target == 1)).sum()),
        "AP": average_precision_score(target, probability),
        "PR-AUC": auc(recall, precision),
        "Log loss": log_loss(target, probability, labels=[0, 1]),
        "Brier": brier_score_loss(target, probability),
        "Средний прогноз": probability.mean(),
    }


def main():
    cases = load_cases()
    weeks = sorted(cases["WEEK_NUM"].unique())
    calibration_start = weeks[int(0.60 * len(weeks))]
    evaluation_start = weeks[int(0.80 * len(weeks))]
    development = cases.loc[cases["WEEK_NUM"] < calibration_start]
    evaluation = cases.loc[cases["WEEK_NUM"] >= evaluation_start]
    assert development["date_decision"].max() < evaluation["date_decision"].min()
    x_train, y_train = development[FEATURES], development["target"]
    x_eval, y_eval = evaluation[FEATURES], evaluation["target"]
    print(f"Обучение: {len(y_train):,}. Оценка: {len(y_eval):,}. "
          f"Доля дефолтов на оценке: {y_eval.mean():.4%}", flush=True)

    constant = pd.Series(y_train.mean(), index=y_eval.index)
    results = {"Средняя доля на обучении": score(y_eval, constant)}
    weekly_auc = {}
    evaluation_scores = {}
    for name, model in build_models().items():
        start = perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(x_train, y_train)
        probability = model.predict_proba(x_eval)[:, 1]
        results[name] = score(y_eval, probability)
        evaluation_scores[name] = probability
        if name == "CatBoost":
            scoring_model = model
        predictions = evaluation[["WEEK_NUM", "target"]].assign(probability=probability)
        weekly_auc[name] = {
            week: roc_auc_score(group["target"], group["probability"])
            for week, group in predictions.groupby("WEEK_NUM")
            if group["target"].nunique() == 2
        }
        print(f"{name}: обучение и оценка за {perf_counter() - start:.1f} с", flush=True)

    print(f"\nПорог прогнозного риска: {RISK_THRESHOLD:.0%}")
    print(pd.DataFrame(results).T.to_string(float_format="{:.6f}".format))
    print("\nROC-AUC по неделям")
    weekly = pd.DataFrame(weekly_auc)
    print(weekly.to_string(float_format="{:.4f}".format))
    print("\nСреднее по неделям без взвешивания")
    print(weekly.mean().to_string(float_format="{:.4f}".format))

    weekly_gini = 2 * weekly - 1
    stability = {}
    for name, gini in weekly_gini.items():
        slope, intercept = np.polyfit(gini.index, gini, deg=1)
        residuals = gini - (slope * gini.index + intercept)
        stability[name] = gini.mean() + 88 * min(0, slope) - 0.5 * residuals.std(ddof=0)
    print("\nСредняя недельная Gini и локальная метрика стабильности")
    print(pd.DataFrame({"Средняя Gini": weekly_gini.mean(), "Стабильность": stability})
          .to_string(float_format="{:.6f}".format))

    # Для интерфейса сохраняем только агрегаты, модель и допустимые категории.
    threshold_rows = []
    targets = y_eval.to_numpy(dtype=bool)
    for name, probability in evaluation_scores.items():
        for percent in range(101):
            flagged = probability >= percent / 100
            threshold_rows.append({
                "Модель": name, "Порог": percent,
                "Помечено": int(flagged.sum()),
                "Найдено дефолтов": int((flagged & targets).sum()),
            })

    report = {
        "cases": len(y_eval),
        "defaults": int(targets.sum()),
        "first_date": str(evaluation["date_decision"].min())[:10],
        "last_date": str(evaluation["date_decision"].max())[:10],
        "categories": {
            name: sorted(x_train[name].unique().tolist()) for name in CATEGORICAL
        },
        "metrics": {
            name: {"Gini": results[name]["Gini"], "Gini Stability": stability[name]}
            for name in evaluation_scores
        },
        "weekly_gini": weekly_gini.rename_axis("Неделя").reset_index().to_dict("records"),
        "thresholds": threshold_rows,
    }
    save_bundle(scoring_model, report)
    print(f"Модель и агрегаты оценки сохранены в {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
