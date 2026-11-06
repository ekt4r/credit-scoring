"""Контракт и загрузка выбранной модели на 120 полях."""

from pathlib import Path

import numpy as np

from credit_scoring.history import HISTORY_FEATURES
from credit_scoring.scoring import MISSING_CATEGORY, read_bundle

SELECTED_PATH = Path("artifacts/scoring-120.zip")
SCHEMA_VERSION = 1


def feature_schema(features, categories):
    schema = []
    for name in features:
        source = "static_0"
        rule = "Исходное значение"
        missing = None
        dtype = "float32"
        if name in categories:
            dtype, missing = "category", MISSING_CATEGORY
        elif name.endswith("D"):
            rule = "Дни от даты события до решения. Будущие даты считаются пропусками."
        elif name in HISTORY_FEATURES:
            source = "applprev_1"
            rule = "Агрегат заявок, созданных строго до решения."
            if name.startswith("prev_count"):
                missing = 0
        elif name.startswith("bureau_"):
            source = "credit_bureau_a_1"
            rule = "Агрегат контрактов с обновлением строго до решения."
            if name.endswith("_rows"):
                missing = 0
        schema.append({"name": name, "dtype": dtype, "missing": missing,
                       "source": source, "rule": rule})
    return schema


def load_selected(path=SELECTED_PATH):
    model, metadata = read_bundle(path)
    schema = metadata["features"]
    names = [field["name"] for field in schema]
    category_indexes = [i for i, field in enumerate(schema) if field["dtype"] == "category"]
    if (metadata["schema_version"] != SCHEMA_VERSION or names != model.feature_names_
            or category_indexes != model.get_cat_feature_indices()):
        raise ValueError("Модель и схема входных признаков несовместимы.")
    return model, metadata


def prepare_selected(metadata, frame):
    schema = metadata["features"]
    names = [field["name"] for field in schema]
    missing = sorted(set(names) - set(frame.columns))
    if missing or not frame.columns.is_unique:
        raise ValueError(f"Неверная схема входных полей. Отсутствуют: {missing}.")
    features = frame.loc[:, names].copy()
    for field in schema:
        name = field["name"]
        if field["dtype"] == "category":
            features[name] = features[name].astype("string").fillna(field["missing"]).astype("category")
        else:
            column = features[name].astype("float32")
            if np.isinf(column).any():
                raise ValueError(f"Бесконечное значение поля {name}.")
            features[name] = column if field["missing"] is None else column.fillna(field["missing"])
    return features


def predict_selected(model, metadata, frame):
    return model.predict_proba(prepare_selected(metadata, frame), thread_count=1)[:, 1]
