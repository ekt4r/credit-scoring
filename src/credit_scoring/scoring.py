"""Схема признаков, экспорт и оценка заявки."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from catboost import CatBoostClassifier
import pandas as pd

ARTIFACT_PATH = Path("artifacts/scoring.zip")
MISSING_CATEGORY = "__MISSING__"
NUMERIC = [
    "credamount_770A", "maininc_215A", "currdebt_22A", "numactivecreds_622L",
    "applications30d_658L", "numrejects9m_859L", "lastapprcredamount_781A",
    "maxdpdlast12m_727P", "avginstallast24m_3658937A", "avgpmtlast12m_4525200A",
    "maxlnamtstart6m_4525199A",
]
CATEGORICAL = ["credtype_322L", "bankacctype_710L"]
FEATURES = NUMERIC + CATEGORICAL
RISK_THRESHOLD = 0.05


def prepare_features(frame):
    features = frame.loc[:, FEATURES].copy()
    features[NUMERIC] = features[NUMERIC].astype("float32")
    for name in CATEGORICAL:
        features[name] = (
            features[name].astype("string").fillna(MISSING_CATEGORY).astype(object)
        )
    return features


def save_bundle(model, report, path=ARTIFACT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        model_path = Path(directory) / "model.cbm"
        bundle_path = Path(directory) / "scoring.zip"
        model.save_model(str(model_path))
        with ZipFile(bundle_path, "w") as bundle:
            bundle.write(model_path, "model.cbm")
            bundle.writestr("report.json", json.dumps(report, ensure_ascii=False, allow_nan=False))
        bundle_path.replace(path)


def read_bundle(path):
    with ZipFile(path) as bundle:
        model = CatBoostClassifier()
        model.load_model(blob=bundle.read("model.cbm"))
        report = json.loads(bundle.read("report.json"))
    return model, report


def load_bundle(path=ARTIFACT_PATH):
    model, report = read_bundle(path)
    if model.feature_names_ != FEATURES:
        raise ValueError("Схема модели устарела. Запустите обучение заново.")
    return model, report


def predict_risk(model, application):
    features = prepare_features(pd.DataFrame([application]))
    return float(model.predict_proba(features, thread_count=1)[0, 1])
