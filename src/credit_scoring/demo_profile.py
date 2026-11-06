"""Синтетические заявки для демонстрации выбранной модели."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from catboost import Pool

from credit_scoring.bureau import CONTRACT_TYPES, add_bureau_features
from credit_scoring.history import add_application_history
from credit_scoring.selected_model import prepare_selected

PROFILES = {
    "Умеренный долг": (24000, 4000, 0),
    "Высокий долг и просрочка": (16000, 32000, 45),
    "Без кредитной истории": (19000, 0, None),
}
LABELS = {
    "twobodfilling_608L": "Тип процесса оформления",
    "bureau_closed_rows": "Записи по закрытым контрактам",
    "bureau_closed_contracts": "Число закрытых контрактов по данным бюро",
    "totalsettled_863A": "Общая сумма платежей",
    "credtype_322L": "Тип кредита",
    "numincomingpmts_3546848L": "Число входящих платежей",
    "numrejects9m_859L": "Отказы за 9 месяцев",
    "maininc_215A": "Основной доход",
    "currdebt_22A": "Текущая задолженность",
    "numactivecreds_622L": "Активные кредиты",
    "maxdpdlast12m_727P": "Максимальная просрочка за 12 месяцев",
    "prev_count": "Предыдущие заявки",
    "prev_days_since_last": "Дней с предыдущей заявки",
    "bureau_active_dpd": "Максимальная просрочка по активному кредиту",
    "bureau_active_overdue": "Просроченная сумма по активному кредиту",
}


def build_profile(name, metadata):
    income, debt, dpd = PROFILES[name]
    decision = pd.Timestamp("2020-08-01")
    base = pd.DataFrame([{"case_id": 1, "date_decision": decision}])
    applications = pd.DataFrame({
        "case_id": [1, 1], "num_group1": [0, 1],
        "creationdate_885D": ["2020-04-12", "2020-08-02"],
        "credamount_590A": [12000, 9000],
    })
    bureau = pd.DataFrame({
        "case_id": [1, 1],
        "lastupdate_1112D": ["2020-07-10", "2020-08-03"],
        "numberofcontrsvalue_258L": [1, 2],
        "dpdmax_139P": [dpd, 90],
        "overdueamount_659A": [0 if not dpd else 1500, 2000],
        **{field: [None, None] for field in CONTRACT_TYPES["closed"]},
    })
    if dpd is None:
        applications = applications.iloc[1:]
        bureau = bureau.iloc[1:]
    with TemporaryDirectory() as directory:
        folder = Path(directory)
        applications.to_parquet(folder / "train_applprev_1_0.parquet", index=False)
        bureau.to_parquet(folder / "train_credit_bureau_a_1_0.parquet", index=False)
        cases = add_application_history(base, folder)
        cases, _ = add_bureau_features(cases, folder)

    static = {field["name"]: None for field in metadata["features"] if field["source"] == "static_0"}
    static.update(maininc_215A=income, currdebt_22A=debt, credtype_322L="CAL",
                  numactivecreds_622L=int(dpd is not None), maxdpdlast12m_727P=dpd)
    cases = pd.concat([cases, pd.DataFrame([static])], axis=1)
    applications = applications.assign(
        Учтена=pd.to_datetime(applications["creationdate_885D"]).lt(decision)
    )
    bureau = bureau.assign(
        Учтён=pd.to_datetime(bureau["lastupdate_1112D"]).lt(decision)
    )
    return cases, static, applications, bureau


def explain_profile(model, metadata, cases):
    features = prepare_selected(metadata, cases)
    pool = Pool(features, cat_features=model.get_cat_feature_indices())
    probability = float(model.predict_proba(pool, thread_count=1)[0, 1])
    contributions = model.get_feature_importance(pool, type="ShapValues", thread_count=1)[0]
    explanation = pd.DataFrame({
        "Признак": [LABELS.get(field["name"], field["name"]) for field in metadata["features"]],
        "Код": list(features.columns),
        "Источник": [field["source"] for field in metadata["features"]],
        "Значение": features.iloc[0].astype("string").fillna("Пропуск").replace("__MISSING__", "Пропуск").tolist(),
        "Вклад в логарифм шансов": contributions[:-1],
    })
    top = explanation["Вклад в логарифм шансов"].abs().nlargest(8).index
    return probability, explanation.loc[top], float(contributions[-1])
