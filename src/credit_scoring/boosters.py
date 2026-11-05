"""Градиентный бустинг с общей схемой входных признаков."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from credit_scoring.scoring import CATEGORICAL
from credit_scoring.train_baselines import build_models

BOOSTERS = ("CatBoost", "XGBoost", "LightGBM")


def make_booster(name):
    if name == "CatBoost":
        return build_models()[name]

    parameters = dict(n_estimators=350, learning_rate=0.08, random_state=42, n_jobs=4)
    if name == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(tree_method="hist", max_depth=6, **parameters)
    elif name == "LightGBM":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(num_leaves=31, max_depth=6, verbosity=-1, **parameters)
    else:
        raise ValueError(f"Неизвестная модель: {name}")

    # Плотные нули сохраняют смысл признаков, а числовые NaN обрабатывает модель.
    transform = ColumnTransformer(
        [("категории", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                    dtype=np.float32), CATEGORICAL)],
        remainder="passthrough",
    ).set_output(transform="pandas")
    return make_pipeline(transform, model)
