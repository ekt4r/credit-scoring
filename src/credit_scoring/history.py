"""Агрегаты предыдущих заявок, созданных до текущего решения."""

import pandas as pd

HISTORY_FEATURES = [
    "prev_count", "prev_count_30d", "prev_count_365d",
    "prev_days_since_last", "prev_amount_mean", "prev_amount_max",
]


def add_application_history(cases, train_dir):
    paths = sorted(train_dir.glob("train_applprev_1_*.parquet"))
    if not paths:
        raise FileNotFoundError("Распакуйте train_applprev_1_*.parquet из архива Home Credit.")
    previous = pd.concat(
        [pd.read_parquet(path, columns=[
            "case_id", "num_group1", "creationdate_885D", "credamount_590A",
        ]) for path in paths],
        ignore_index=True,
    )
    if previous.duplicated(["case_id", "num_group1"]).any():
        raise ValueError("Исторические заявки имеют повторяющиеся ключи.")
    previous = previous.merge(cases[["case_id", "date_decision"]], on="case_id",
                              how="inner", validate="many_to_one")
    age = (pd.to_datetime(previous["date_decision"])
           - pd.to_datetime(previous["creationdate_885D"])).dt.days
    history = previous.loc[age > 0, ["case_id", "credamount_590A"]].assign(days=age[age > 0])
    print(f"История: {len(history):,} записей до решения из {len(previous):,} связанных. "
          f"Без даты: {age.isna().sum():,}, в день решения или позже: {age.le(0).sum():,}.",
          flush=True)
    history["within_30d"] = history["days"] <= 30
    history["within_365d"] = history["days"] <= 365
    aggregates = history.groupby("case_id", sort=False).agg(
        prev_count=("days", "size"),
        prev_count_30d=("within_30d", "sum"),
        prev_count_365d=("within_365d", "sum"),
        prev_days_since_last=("days", "min"),
        prev_amount_mean=("credamount_590A", "mean"),
        prev_amount_max=("credamount_590A", "max"),
    )
    enriched = cases.join(aggregates, on="case_id", validate="one_to_one")
    counts = HISTORY_FEATURES[:3]
    enriched[counts] = enriched[counts].fillna(0)
    enriched[HISTORY_FEATURES] = enriched[HISTORY_FEATURES].astype("float32")
    return enriched
