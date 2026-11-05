"""Счётчики запросов в кредитное бюро за фиксированные окна."""

import pandas as pd

QUERY_FEATURES = [
    "days30_165L", "days90_310L", "days120_123L", "days180_256L", "days360_512L",
]


def add_bureau_queries(cases, train_dir):
    bureau = pd.read_parquet(
        train_dir / "train_static_cb_0.parquet", columns=["case_id", *QUERY_FEATURES],
    ).set_index("case_id", verify_integrity=True)
    queries = bureau.astype("float32")
    if queries.lt(0).any().any():
        raise ValueError("В счётчиках запросов обнаружены отрицательные значения.")
    joined = cases.join(queries, on="case_id", validate="one_to_one")
    coverage = joined[QUERY_FEATURES].notna().all(axis=1).mean()
    print(f"Полностью заполненные счётчики запросов: {coverage:.2%}.", flush=True)
    return joined
