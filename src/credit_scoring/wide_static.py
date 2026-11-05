"""Эксперимент со всеми полями static_0 и историей заявок."""

import argparse
import resource
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.parquet import ParquetFile

from credit_scoring.boosters import make_booster
from credit_scoring.bureau import add_bureau_features
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import MISSING_CATEGORY
from credit_scoring.static_cb import QUERY_FEATURES, add_bureau_queries
from credit_scoring.temporal import FOLD_BOUNDARIES, evaluate
from credit_scoring.train_baselines import TRAIN_DIR


# Группы риска по описаниям полей. Это фильтр по смыслу, доступность
# оставшихся полей требует подтверждения на источнике данных.
EXCLUDED_BY_REASON = {
    "условия и события текущей выдачи": {
        "annuity_780A", "annuitynextmonth_57A", "credamount_770A",
        "deferredmnthsnum_166L", "disbursedcredamount_1113A", "disbursementtype_67L",
        "downpmt_116A", "eir_270L", "firstdatedue_489D", "inittransactionamount_650A",
        "inittransactioncode_186L", "interestrate_311L", "interestrategrace_34L",
        "maxannuity_4075009A", "monthsannuity_845L", "numinstls_657L",
        "paytype1st_925L", "paytype_783L", "price_1097A",
    },
    "срез результатов по торговой точке с неопределённым созреванием": {
        "posfpd10lastmonth_333P", "posfpd30lastmonth_3976960P", "posfstqpd30lastmonth_3976962P",
    },
    "сводка последней заявки без подтверждённой даты снимка": {
        "lastapplicationdate_877D", "lastactivateddate_801D", "lastapprdate_640D",
        "lastapprcommoditycat_1041M", "lastapprcommoditytypec_5251766M",
        "lastapprcredamount_781A", "lastcancelreason_561M", "lastst_736L",
        "lastrejectdate_50D", "lastrejectcommoditycat_161M", "lastrejectcommodtypec_5251769M",
        "lastrejectcredamount_222A", "lastrejectreason_759M", "lastrejectreasonclient_4145040M",
        "previouscontdistrict_112M", "lastdependentsnum_448L", "lastotherinc_902A",
        "lastotherlnsexpense_631A", "opencred_647L", "numinstpaidlastcontr_4325080L",
    },
    "межклиентские счётчики без описанной границы времени": {
        "applicationcnt_361L", "applicationscnt_1086L", "applicationscnt_867L",
        "clientscnt_100L", "clientscnt_1022L", "clientscnt_1071L", "clientscnt_1130L",
        "clientscnt_136L", "clientscnt_157L", "clientscnt_257L", "clientscnt_304L",
        "clientscnt_360L", "clientscnt_493L", "clientscnt_533L", "clientscnt_887L",
        "clientscnt_946L", "mobilephncnt_593L", "sellerplacecnt_915L", "sellerplacescnt_216L",
    },
}


def load_wide_cases(excluded=()):
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
    unknown = set(excluded) - set(schema.names)
    if unknown:
        raise ValueError(f"Поля из списка исключений отсутствуют: {sorted(unknown)}")
    columns = [name for name in schema.names if name != "case_id" and name not in excluded]
    selected = ["case_id", *columns]
    dates = [name for name in columns if name.endswith("D")]
    categories = [name for name in columns if pa.types.is_string(schema.field(name).type)
                  and name not in dates]
    compact_schema = pa.schema([
        field if field.name in {"case_id", *dates, *categories}
        else pa.field(field.name, pa.float32())
        for field in (schema.field(name) for name in selected)
    ])

    batches = []
    for path in paths:
        for batch in ParquetFile(path).iter_batches(batch_size=65_536, columns=selected):
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
    parser.add_argument("--variant", choices=("full", "screened", "screened-bureau", "screened-bureau-queries"), default="full")
    args = parser.parse_args()
    excluded = set().union(*EXCLUDED_BY_REASON.values()) if args.variant != "full" else ()
    cases, features, categories = load_wide_cases(excluded)
    coverage_column = None
    if args.variant in {"screened-bureau", "screened-bureau-queries"}:
        cases, bureau_features = add_bureau_features(cases.reset_index(), TRAIN_DIR)
        features += bureau_features
        coverage_column = "bureau_covered"
        cases[coverage_column] = cases["bureau_active_rows"].gt(0) | cases["bureau_closed_rows"].gt(0)
    if args.variant == "screened-bureau-queries":
        cases = add_bureau_queries(cases, TRAIN_DIR)
        features += QUERY_FEATURES
    print(f"Вариант {args.variant}, входных полей: {len(features)}.", flush=True)
    weekly = [] if args.variant != "full" else None
    rows = evaluate(
        cases, f"CatBoost · {args.variant}", features,
        lambda: make_booster("CatBoost").set_params(
            cat_features=categories, used_ram_limit=args.ram_limit,
        ),
        coverage_column=coverage_column, weekly_rows=weekly,
    )
    suffix = "" if args.variant == "full" else f"-{args.variant}"
    output = Path(f"artifacts/features-wide-static{suffix}.csv")
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    if weekly is not None:
        pd.DataFrame(weekly).to_csv(output.with_name(f"{output.stem}-weekly.csv"), index=False)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Пиковый RSS: {rss / (1024**2 if sys.platform == 'darwin' else 1024):.0f} МиБ.", flush=True)


if __name__ == "__main__":
    main()
