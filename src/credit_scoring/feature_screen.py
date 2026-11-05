"""Последовательная проверка блоков признаков на одинаковых временных фолдах."""

import argparse
from pathlib import Path

import pandas as pd

from credit_scoring.boosters import make_booster
from credit_scoring.bureau import add_bureau_features
from credit_scoring.bureau_payments import add_bureau_payments
from credit_scoring.temporal import evaluate
from credit_scoring.train_baselines import TRAIN_DIR
from credit_scoring.wide_static import EXCLUDED_BY_REASON, load_wide_cases

DATED_SOURCES = {
    "Бюро B · контракты": [
        ("credit_bureau_b_1", "lastupdate_260D", ["dpd_550P", "debtpastduevalue_732A"]),
    ],
    "Заявитель и стаж": [
        ("person_1", "empl_employedfrom_271D", ["mainoccupationinc_384A"]),
        ("person_2", "empls_employedfrom_796D", []),
    ],
    "Налоговые записи": [
        ("tax_registry_a_1", "recorddate_4527225D", ["amount_4527230A"]),
        ("tax_registry_b_1", "deductiondate_4917603D", ["amount_4917619A"]),
        ("tax_registry_c_1", "processingdate_168D", ["pmtamount_36A"]),
    ],
    "Дебетовые карты": [("debitcard_1", "openingdate_857D", [])],
    "Депозиты": [("deposit_1", "openingdate_313D", [])],
}


def read_table(table, columns):
    return pd.read_parquet(TRAIN_DIR / f"train_{table}.parquet", columns=columns)


def dated_features(cases, sources, audit):
    decisions = pd.to_datetime(cases.set_index("case_id")["date_decision"])
    parts = []
    for table, date, amounts in sources:
        columns = ["case_id", date, *amounts]
        if table.startswith("person"):
            columns.append("num_group1")
        source = read_table(table, columns)
        if table.startswith("person"):
            source = source.loc[source["num_group1"].eq(0)]
        linked = source["case_id"].isin(decisions.index)
        source = source.loc[linked].copy()
        age = (source["case_id"].map(decisions) - pd.to_datetime(source[date])).dt.days
        eligible = source.loc[age.gt(0)].assign(age=age.loc[age.gt(0)])
        group = eligible.groupby("case_id", sort=False)
        metrics = group["age"].agg(["size", "min", "max"])
        if amounts:
            metrics = metrics.join(group[amounts].max())
        parts.append(metrics.add_prefix(f"{table}_"))
        audit.append({"Источник": table, "Связано": len(source), "Допущено": len(eligible),
                      "Кейсов": len(metrics), "Без даты": int(age.isna().sum()),
                      "В день решения или позже": int(age.le(0).sum())})
    return pd.concat(parts, axis=1).astype("float32")


def bureau_b_payments(cases, audit):
    parents = read_table("credit_bureau_b_1", ["case_id", "num_group1", "lastupdate_260D"])
    decisions = pd.to_datetime(cases.set_index("case_id")["date_decision"])
    parents = parents.loc[pd.to_datetime(parents["lastupdate_260D"]).lt(parents["case_id"].map(decisions))]
    payments = read_table("credit_bureau_b_2", ["case_id", "num_group1", "pmts_date_1107D", "pmts_dpdvalue_108P"])
    linked = payments["case_id"].isin(decisions.index)
    age = (payments["case_id"].map(decisions) - pd.to_datetime(payments["pmts_date_1107D"])).dt.days
    keys = pd.MultiIndex.from_frame(payments[["case_id", "num_group1"]])
    valid = age.gt(0) & keys.isin(pd.MultiIndex.from_frame(parents[["case_id", "num_group1"]]))
    group = payments.loc[valid].groupby("case_id", sort=False)["pmts_dpdvalue_108P"]
    result = group.agg(["size", "max", "mean"]).add_prefix("bureau_b_payment_")
    audit.append({"Источник": "credit_bureau_b_2", "Связано": int(linked.sum()),
                  "Допущено": int(valid.sum()), "Кейсов": len(result)})
    return result.astype("float32")


def previous_approvals(cases, audit):
    decisions = pd.to_datetime(cases.set_index("case_id")["date_decision"])
    columns = ["case_id", "creationdate_885D", "approvaldate_319D"]
    parts = [pd.read_parquet(path, columns=columns)
             for path in sorted(TRAIN_DIR.glob("train_applprev_1_*.parquet"))]
    rows = pd.concat(parts, ignore_index=True)
    creation = pd.to_datetime(rows["creationdate_885D"])
    approval = pd.to_datetime(rows["approvaldate_319D"])
    decision = rows["case_id"].map(decisions)
    valid = creation.le(approval) & approval.lt(decision)
    age = (decision - approval).dt.days
    result = rows.loc[valid].assign(age=age.loc[valid]).groupby("case_id", sort=False)["age"].agg(["size", "min"])
    audit.append({"Источник": "applprev_1 · одобрения", "Связано": int(decision.notna().sum()),
                  "Допущено": int(valid.sum()), "Кейсов": len(result)})
    return result.astype("float32").add_prefix("approval_")


def screen_model():
    return make_booster("LightGBM").set_params(
        lgbmclassifier__n_estimators=100, lgbmclassifier__max_bin=63,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamics-only", action="store_true")
    args = parser.parse_args()
    cases, features, _ = load_wide_cases(set().union(*EXCLUDED_BY_REASON.values()))
    cases, bureau = add_bureau_features(cases.reset_index(), TRAIN_DIR)
    features += bureau
    bureau_a = cases["bureau_active_rows"].gt(0) | cases["bureau_closed_rows"].gt(0)
    output = Path("artifacts/feature-screen")
    output.mkdir(parents=True, exist_ok=True)
    if args.dynamics_only:
        frame, names = add_bureau_payments(cases, TRAIN_DIR, windows=(3, 6, 12), dynamics=True)
        rows = evaluate(frame, "Окна и динамика просрочек", [*features, *names], screen_model)
        pd.DataFrame(rows).to_csv(output / "dynamics.csv", index=False)
        return

    audit = []
    for table in ("applprev_2", "other_1"):
        keys = read_table(table, ["case_id"])["case_id"]
        linked = keys.isin(cases["case_id"])
        audit.append({"Источник": table, "Связано": int(linked.sum()), "Кейсов": keys[linked].nunique()})
    results = []

    def run(name, block=None):
        if block is None:
            frame, fields = cases, features
        else:
            frame = cases.join(block, on="case_id", validate="one_to_one")
            fields = [*features, *block.columns]
        results.extend(evaluate(frame, name, fields, screen_model))
        pd.DataFrame(results).to_csv(output / "folds.csv", index=False)
        pd.DataFrame(audit).to_csv(output / "sources.csv", index=False)

    run("Контроль 120")
    for name, sources in DATED_SOURCES.items():
        block = dated_features(cases, sources, audit)
        if name == "Бюро B · контракты":
            covered = cases["case_id"].isin(block.index)
            audit[-1]["Только B"] = int((covered & ~bureau_a).sum())
            audit[-1]["A и B"] = int((covered & bureau_a).sum())
        if not block.empty:
            run(name, block)
    block = bureau_b_payments(cases, audit)
    if not block.empty:
        run("Бюро B · платежи", block)

    income = cases["maininc_215A"].where(cases["maininc_215A"].gt(0))
    ratios = cases[["currdebt_22A", "avgpmtlast12m_4525200A"]].div(income, axis=0)
    ratios.columns = ["debt_income_proxy", "payment_income_proxy"]
    run("Отношения к доходу", ratios.set_axis(cases["case_id"]))
    run("Предыдущие одобрения", previous_approvals(cases, audit))
    enriched, names = add_bureau_payments(cases, TRAIN_DIR, windows=(3, 6, 12))
    run("Окна просрочек 3/6/12", enriched.set_index("case_id")[names])


if __name__ == "__main__":
    main()
