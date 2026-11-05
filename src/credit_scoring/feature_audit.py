"""Проверка доступности полей static_0 по схеме, описаниям и датам."""

from pathlib import Path

import pandas as pd
from pyarrow.parquet import ParquetFile

from credit_scoring.train_baselines import TRAIN_DIR
from credit_scoring.wide_static import EXCLUDED_BY_REASON

APPLICATION_FIELDS = {
    "bankacctype_710L": "Тип счёта может быть получен при заполнении заявки.",
    "maininc_215A": "Основной доход может быть получен при заполнении заявки.",
    "typesuite_864L": "Сопровождающие лица известны при заполнении заявки.",
    "credtype_322L": "Тип кредита доступен, если это запрошенный продукт из анкеты.",
}


def audit_static():
    paths = sorted(TRAIN_DIR.glob("train_static_0_*.parquet"))
    schema = ParquetFile(paths[0]).schema_arrow
    fields = [name for name in schema.names if name != "case_id"]
    dates = [name for name in fields if name.endswith("D")]
    decisions = pd.read_parquet(TRAIN_DIR / "train_base.parquet",
                                columns=["case_id", "date_decision"]).set_index("case_id")
    decisions = pd.to_datetime(decisions["date_decision"])
    excluded = {name: reason for reason, names in EXCLUDED_BY_REASON.items() for name in names}
    definitions = pd.read_csv(TRAIN_DIR.parents[1] / "feature_definitions.csv")
    if missing := set(fields) - set(definitions["Variable"]):
        raise ValueError(f"Поля отсутствуют в словаре: {sorted(missing)}")

    date_metrics = ("Заполнено", "До решения", "В день решения", "После решения", "Ошибка разбора")
    counts = {name: dict.fromkeys(date_metrics, 0) for name in dates}
    matched = 0
    for path in paths:
        for batch in ParquetFile(path).iter_batches(batch_size=65_536, columns=["case_id", *dates]):
            frame = batch.to_pandas()
            decision = frame["case_id"].map(decisions)
            linked = decision.notna()
            matched += int(linked.sum())
            for name in dates:
                raw = frame[name].loc[linked]
                parsed = pd.to_datetime(raw, errors="coerce")
                cutoff = decision.loc[linked]
                count = counts[name]
                count["Заполнено"] += int(raw.notna().sum())
                count["До решения"] += int(parsed.lt(cutoff).sum())
                count["В день решения"] += int(parsed.eq(cutoff).sum())
                count["После решения"] += int(parsed.gt(cutoff).sum())
                count["Ошибка разбора"] += int((raw.notna() & parsed.isna()).sum())

    rows = []
    for name in fields:
        if name in excluded:
            status, reason = "Исключить", excluded[name]
        elif name in APPLICATION_FIELDS:
            status, reason = "Кандидат из анкеты", APPLICATION_FIELDS[name]
        elif name in dates and (counts[name]["После решения"] or counts[name]["В день решения"]):
            status = "Отложить до проверки дат"
            reason = "Есть даты в день решения или позже. Нужны правила преобразования и время загрузки."
        elif name in dates:
            status = "Проверить момент события"
            reason = "Для исторического события нужна дата строго до решения и подтверждение даты загрузки."
        else:
            status = "Проверить срез источника"
            reason = "Словарь не подтверждает время загрузки или границу расчёта значения."
        rows.append({"Поле": name, "Статус": status, "Причина": reason, **counts.get(name, {})})
    report = pd.DataFrame(rows).convert_dtypes()
    print(f"Проверено строк с известной датой решения: {matched:,}")
    print(report["Статус"].value_counts().to_string())
    print(report.loc[report["Заполнено"].notna(), ["Поле", *date_metrics]]
          .to_string(index=False))
    return report


def main():
    output = Path("docs/feature-availability.csv")
    output.parent.mkdir(exist_ok=True)
    audit_static().to_csv(output, index=False)


if __name__ == "__main__":
    main()
