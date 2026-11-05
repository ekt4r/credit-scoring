"""Признаки бюро по контрактам с обновлением до даты решения."""

import pandas as pd
from pyarrow.parquet import ParquetFile

CONTRACT_TYPES = {
    "active": ("lastupdate_1112D", "numberofcontrsvalue_258L", "dpdmax_139P", "overdueamount_659A"),
    "closed": ("lastupdate_388D", "numberofcontrsvalue_358L", "dpdmax_757P", "overdueamount_31A"),
}


def add_bureau_features(cases, train_dir):
    paths = sorted(train_dir.glob("train_credit_bureau_a_1_*.parquet"))
    if not paths:
        raise FileNotFoundError("Распакуйте train_credit_bureau_a_1_*.parquet.")
    decisions = cases.set_index("case_id")["date_decision"].pipe(pd.to_datetime)
    columns = ["case_id", *(column for group in CONTRACT_TYPES.values() for column in group)]
    partial = {kind: [] for kind in CONTRACT_TYPES}
    audit = {kind: {"допущено": 0, "без даты": 0, "в день решения или позже": 0}
             for kind in CONTRACT_TYPES}

    for path in paths:
        for batch in ParquetFile(path).iter_batches(batch_size=500_000, columns=columns):
            frame = batch.to_pandas()
            decision = frame["case_id"].map(decisions)
            for kind, (update, count, dpd, overdue) in CONTRACT_TYPES.items():
                age = (decision - pd.to_datetime(frame[update])).dt.days
                eligible = age.gt(0)
                audit[kind]["допущено"] += int(eligible.sum())
                audit[kind]["без даты"] += int(frame[update].isna().sum())
                audit[kind]["в день решения или позже"] += int(age.le(0).sum())
                valid = frame.loc[eligible, ["case_id", count, dpd, overdue]]
                grouped = valid.groupby("case_id", sort=False).agg(
                    rows=(count, "size"), contracts=(count, "max"),
                    dpd=(dpd, "max"), overdue=(overdue, "max"),
                )
                partial[kind].append(grouped)

    blocks = []
    for kind, pieces in partial.items():
        group = pd.concat(pieces).groupby(level=0, sort=False)
        metrics = group.agg({"rows": "sum", "contracts": "max", "dpd": "max", "overdue": "max"})
        blocks.append(metrics.add_prefix(f"bureau_{kind}_"))
        print(f"Бюро, {kind}: {audit[kind]}; кейсов с записью: {len(metrics):,}", flush=True)

    bureau = pd.concat(blocks, axis=1).astype("float32")
    enriched = cases.join(bureau, on="case_id", validate="one_to_one")
    for kind in CONTRACT_TYPES:
        column = f"bureau_{kind}_rows"
        enriched[column] = enriched[column].fillna(0)
    return enriched, bureau.columns.tolist()
