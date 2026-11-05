"""Агрегаты помесячных платежей бюро по допустимым снимкам контрактов."""

import pandas as pd
from pyarrow.parquet import ParquetFile

from credit_scoring.bureau import CONTRACT_TYPES

KEYS = ["case_id", "num_group1"]
PAYMENTS = {
    "active": ("pmts_year_1139T", "pmts_month_158T", "pmts_dpd_1073P", "pmts_overdue_1140A"),
    "closed": ("pmts_year_507T", "pmts_month_706T", "pmts_dpd_303P", "pmts_overdue_1152A"),
}
REDUCTIONS = {
    "rows": "sum", "known_dpd": "sum", "late": "sum",
    "max_dpd": "max", "max_overdue": "max", "months_since_late": "min",
}


def eligible_contracts(decisions, paths):
    updates = {kind: columns[0] for kind, columns in CONTRACT_TYPES.items()}
    keys = {kind: [] for kind in updates}
    for path in paths:
        for batch in ParquetFile(path).iter_batches(
            batch_size=500_000, columns=[*KEYS, *updates.values()],
        ):
            frame = batch.to_pandas()
            decision = frame["case_id"].map(decisions)
            for kind, update in updates.items():
                eligible = pd.to_datetime(frame[update]).lt(decision)
                keys[kind].append(frame.loc[eligible, KEYS])
    return {kind: pd.MultiIndex.from_frame(pd.concat(parts, ignore_index=True)).unique()
            for kind, parts in keys.items()}


def combine(parts):
    return pd.concat(parts).groupby(level=0, sort=False).agg(REDUCTIONS)


def add_bureau_payments(cases, train_dir):
    contract_paths = sorted(train_dir.glob("train_credit_bureau_a_1_*.parquet"))
    payment_paths = sorted(train_dir.glob("train_credit_bureau_a_2_*.parquet"))
    if not contract_paths or not payment_paths:
        raise FileNotFoundError("Распакуйте все части train_credit_bureau_a_1 и train_credit_bureau_a_2.")
    decisions = pd.to_datetime(cases.set_index("case_id", verify_integrity=True)["date_decision"])
    decision_months = decisions.dt.year * 12 + decisions.dt.month
    parents = eligible_contracts(decisions, contract_paths)
    columns = [*KEYS, *(column for group in PAYMENTS.values() for column in group)]
    shards = {kind: [] for kind in PAYMENTS}
    for path in payment_paths:
        partial = {kind: [] for kind in PAYMENTS}
        for batch in ParquetFile(path).iter_batches(batch_size=500_000, columns=columns):
            frame = batch.to_pandas()
            decision_month = frame["case_id"].map(decision_months)
            for kind, (year, month, dpd, overdue) in PAYMENTS.items():
                years, months = frame[year], frame[month]
                age = decision_month - (years * 12 + months)
                calendar = (years.between(1900, 2100) & years.mod(1).eq(0)
                            & months.between(1, 12) & months.mod(1).eq(0))
                valid = frame.loc[calendar & age.gt(0), [*KEYS, dpd, overdue]]
                valid = valid.loc[pd.MultiIndex.from_frame(valid[KEYS]).isin(parents[kind])]
                late = valid[dpd].gt(0)
                grouped = valid.assign(
                    late=late, months_since_late=age.loc[valid.index].where(late),
                ).groupby("case_id", sort=False).agg(
                    rows=(dpd, "size"), known_dpd=(dpd, "count"), late=("late", "sum"),
                    max_dpd=(dpd, "max"), max_overdue=(overdue, "max"),
                    months_since_late=("months_since_late", "min"),
                )
                partial[kind].append(grouped)
        for kind, parts in partial.items():
            shards[kind].append(combine(parts))
        print(f"Платежи: обработан {path.name}.", flush=True)

    blocks = []
    for kind, parts in shards.items():
        metrics = combine(parts)
        metrics["late_share"] = metrics["late"].div(metrics["known_dpd"].where(metrics["known_dpd"] > 0))
        metrics = metrics.drop(columns="known_dpd")
        print(f"Платежи, {kind}: {int(metrics['rows'].sum()):,} строк, "
              f"{len(metrics):,} кейсов.", flush=True)
        blocks.append(metrics.add_prefix(f"payments_{kind}_"))

    payments = pd.concat(blocks, axis=1).astype("float32")
    joined = cases.join(payments, on="case_id", validate="one_to_one")
    for kind in PAYMENTS:
        count = f"payments_{kind}_rows"
        joined[count] = joined[count].fillna(0)
    return joined, payments.columns.tolist()
