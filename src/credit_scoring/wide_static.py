"""Эксперимент со всеми полями static_0 и историей заявок."""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.parquet import ParquetFile

from credit_scoring.boosters import make_booster
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import MISSING_CATEGORY
from credit_scoring.temporal import FOLD_BOUNDARIES, evaluate
from credit_scoring.train_baselines import TRAIN_DIR


def load_wide_cases():
    cases = pd.read_parquet(TRAIN_DIR / "train_base.parquet",
                            columns=["case_id", "date_decision", "WEEK_NUM", "target"])
    cases = cases.loc[cases["WEEK_NUM"] < FOLD_BOUNDARIES[-1]]
    cases = add_application_history(cases, TRAIN_DIR).set_index("case_id", verify_integrity=True)
    decision = pd.to_datetime(cases["date_decision"])
    case_ids = pa.array(cases.index)
    paths = sorted(TRAIN_DIR.glob("train_static_0_*.parquet"))
    if not paths:
        raise FileNotFoundError("Распакуйте train_static_0_*.parquet.")

    schema = ParquetFile(paths[0]).schema_arrow
    columns = [name for name in schema.names if name != "case_id"]
    dates = [name for name in columns if name.endswith("D")]
    categories = [name for name in columns if pa.types.is_string(schema.field(name).type)
                  and name not in dates]
    compact_schema = pa.schema([
        field if field.name in {"case_id", *dates, *categories}
        else pa.field(field.name, pa.float32())
        for field in schema
    ])

    batches = []
    for path in paths:
        for batch in ParquetFile(path).iter_batches(batch_size=65_536):
            mask = pc.is_in(batch.column("case_id"), value_set=case_ids)
            batches.append(batch.filter(mask).cast(compact_schema, safe=False))
    static = pa.Table.from_batches(batches, schema=compact_schema).to_pandas(
        categories=categories, split_blocks=True,
    ).set_index("case_id", verify_integrity=True)
    del batches
    static = static.reindex(cases.index)

    for name in dates:
        days = (decision - pd.to_datetime(static[name], errors="raise")).dt.days
        static[name] = days.where(days >= 0).astype("float32")
    for name in categories:
        column = static[name]
        if MISSING_CATEGORY not in column.cat.categories:
            column = column.cat.add_categories([MISSING_CATEGORY])
        static[name] = column.fillna(MISSING_CATEGORY)

    cases = pd.concat([cases, static], axis=1, copy=False)
    size_mib = cases.memory_usage(index=True, deep=True).sum() / 1024**2
    print(f"{len(cases):,} кейсов, {len(columns)} статических полей "
          f"({len(categories)} категорий, {len(dates)} дат), таблица {size_mib:.0f} МиБ.",
          flush=True)
    return cases, columns + HISTORY_FEATURES, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ram-limit", default="2gb",
                        help="Мягкий лимит памяти CatBoost при расчёте статистик категорий.")
    args = parser.parse_args()
    cases, features, categories = load_wide_cases()
    rows = evaluate(
        cases, "CatBoost · все static_0 + история", features,
        lambda: make_booster("CatBoost").set_params(
            cat_features=categories, used_ram_limit=args.ram_limit,
        ),
    )
    output = Path("artifacts/features-wide-static.csv")
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
