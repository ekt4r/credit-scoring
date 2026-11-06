"""Интерфейс демонстрационного кредитного скоринга."""

from pathlib import Path
from zipfile import BadZipFile

from catboost import CatBoostError
import pandas as pd
import streamlit as st

from credit_scoring.demo_profile import LABELS, PROFILES, build_profile, explain_profile
from credit_scoring.selected_model import SELECTED_PATH, load_selected
from credit_scoring.scoring import (
    ARTIFACT_PATH, MISSING_CATEGORY, RISK_THRESHOLD, load_bundle, predict_risk,
)

APPLICATION_FIELDS = {
    "credamount_770A": "Сумма кредита или лимит",
    "maininc_215A": "Основной доход",
    "currdebt_22A": "Текущая задолженность",
}
HISTORY_FIELDS = {
    "numactivecreds_622L": "Активные кредиты",
    "applications30d_658L": "Заявки за 30 дней",
    "numrejects9m_859L": "Отказы за 9 месяцев",
    "lastapprcredamount_781A": "Сумма последней одобренной заявки",
    "maxdpdlast12m_727P": "Максимальная просрочка за 12 месяцев, дней",
    "avginstallast24m_3658937A": "Средний взнос за 24 месяца",
    "avgpmtlast12m_4525200A": "Средний платёж за 12 месяцев",
    "maxlnamtstart6m_4525199A": "Максимальная сумма кредита, открытого за 6 месяцев",
}
CATEGORY_FIELDS = {"credtype_322L": "Тип кредита", "bankacctype_710L": "Тип банковского счёта"}
COUNT_FIELDS = {"numactivecreds_622L", "applications30d_658L", "numrejects9m_859L", "maxdpdlast12m_727P"}
EXAMPLE = {
    "credamount_770A": 20000.0,
    "maininc_215A": 18000.0,
    "currdebt_22A": 5000.0,
    "numactivecreds_622L": 1,
    "applications30d_658L": 1,
    "numrejects9m_859L": 0,
    "lastapprcredamount_781A": 12000.0,
    "maxdpdlast12m_727P": 3,
    "avginstallast24m_3658937A": 1100.0,
    "avgpmtlast12m_4525200A": 1000.0,
    "maxlnamtstart6m_4525199A": 12000.0,
}


@st.cache_resource
def cached_bundle(path, version):
    return load_bundle(path)


def number_fields(fields):
    for name, label in fields.items():
        step = 1 if name in COUNT_FIELDS else 0.01
        st.number_input(label, value=None, min_value=0 * step, step=step, key=name)


def show_application(model, report):
    st.header("Оценить заявку")
    st.caption("Демонстрация на синтетических данных. Денежные поля вводятся в единицах набора Home Credit.")
    if st.button("Заполнить пример"):
        st.session_state.update(EXAMPLE)
        for name in CATEGORY_FIELDS:
            options = [value for value in report["categories"][name] if value != MISSING_CATEGORY]
            st.session_state[name] = options[0] if options else None
        st.session_state.pop("last_risk", None)

    with st.form("application"):
        st.subheader("Заявка и финансы")
        number_fields(APPLICATION_FIELDS)
        for name, label in CATEGORY_FIELDS.items():
            options = [value for value in report["categories"][name] if value != MISSING_CATEGORY]
            st.selectbox(label, [None, *options],
                         format_func=lambda value: "Неизвестно" if value is None else value,
                         key=name)
        st.caption("Коды категорий взяты из обучающей выборки. Пустое числовое поле означает пропуск.")
        with st.expander("Кредитная история"):
            number_fields(HISTORY_FIELDS)
        submitted = st.form_submit_button("Оценить риск", type="primary")

    if submitted:
        application = {
            name: st.session_state[name]
            for name in APPLICATION_FIELDS | HISTORY_FIELDS | CATEGORY_FIELDS
        }
        if all(value is None for value in application.values()):
            st.session_state.pop("last_risk", None)
            st.warning("Заполните хотя бы одно поле заявки.")
        else:
            st.session_state["last_risk"] = predict_risk(model, application)

    if "last_risk" in st.session_state:
        risk = st.session_state["last_risk"]
        st.metric("Последняя оценка риска CatBoost", f"{risk:.2%}")
        if risk >= RISK_THRESHOLD:
            st.warning(f"Заявка помечена при демонстрационном пороге {RISK_THRESHOLD:.0%}.")
        else:
            st.info(f"Оценка ниже демонстрационного порога {RISK_THRESHOLD:.0%}.")
        st.caption("Вероятности без дополнительной калибровки. Результат предназначен для демонстрации.")


def show_quality(report):
    st.header("Качество модели")
    st.caption(f"Отложенный период {report['first_date']} — {report['last_date']}. "
               f"{report['cases']:,} кейсов, из них {report['defaults']:,} дефолтов.")
    st.dataframe(pd.DataFrame.from_dict(report["metrics"], orient="index")
                 .rename_axis("Модель").round(4))
    st.subheader("Gini по неделям")
    st.line_chart(pd.DataFrame(report["weekly_gini"]).set_index("Неделя"),
                  x_label="Неделя принятия решения", y_label="Gini")

    st.subheader("Что меняется при выборе порога")
    threshold = st.slider("Порог прогнозного риска, %", 0, 100, round(100 * RISK_THRESHOLD))
    threshold_table = pd.DataFrame(report["thresholds"])
    selected = threshold_table.loc[threshold_table["Порог"] == threshold].copy()
    selected["Precision"] = selected["Найдено дефолтов"].div(
        selected["Помечено"].where(selected["Помечено"] > 0)
    )
    selected["Recall"] = selected["Найдено дефолтов"] / report["defaults"]
    st.dataframe(
        selected.drop(columns="Порог").set_index("Модель"),
        column_config={
            "Precision": st.column_config.NumberColumn(format="percent"),
            "Recall": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.caption("Precision — доля дефолтов среди помеченных. "
               "Recall — доля найденных среди всех дефолтов. "
               "Если модель не пометила ни одной заявки, precision не определён.")


@st.cache_resource
def cached_selected(path, version):
    return load_selected(path)


def show_profile(model, metadata):
    st.header("Синтетический профиль")
    st.caption("CatBoost на 120 полях. Пример показывает расчёт на дату решения. "
               "Незаполненные поля анкеты передаются как пропуски.")
    name = st.selectbox("Профиль", list(PROFILES))
    cases, static, applications, bureau = build_profile(name, metadata)
    st.write("Дата решения — 1 августа 2020 года.")
    with st.expander("Анкета и история", expanded=True):
        st.write({LABELS.get(key, key): value for key, value in static.items() if value is not None})
        st.write("Предыдущие заявки")
        st.dataframe(applications.drop(columns=["case_id", "num_group1"]))
        st.write("Снимки бюро")
        st.dataframe(bureau[["lastupdate_1112D", "dpdmax_139P", "Учтён"]])
    with st.expander("Агрегаты, переданные в модель"):
        fields = [field["name"] for field in metadata["features"] if field["source"] != "static_0"]
        st.dataframe(cases[fields].rename(columns=LABELS).T.rename(columns={0: "Значение"}))
    if st.button("Оценить профиль", type="primary"):
        risk, explanation, base = explain_profile(model, metadata, cases)
        st.metric("Исходный скор CatBoost", f"{risk:.2%}")
        st.subheader("Что повлияло на скор")
        st.bar_chart(explanation, x="Признак", y="Вклад в логарифм шансов", horizontal=True)
        st.dataframe(explanation, hide_index=True)
        st.caption(f"Базовый логарифм шансов: {base:.3f}. Положительный вклад повышает скор. "
                   "Показаны восемь крупнейших вкладов SHAP для этого примера.")


def show_review_curve(metadata):
    st.header("Качество модели на 120 полях")
    st.caption("Поздняя локальная проверка, недели 73–91. Объём проверки выбирается "
               "ретроспективно по заявкам с наивысшим скором.")
    st.metric("Gini Stability", f"{metadata['metrics']['Gini Stability']:.4f}")
    curve_path = Path("artifacts/review-curve.csv")
    weekly_path = curve_path.with_name("selected-weekly-gini.csv")
    if not curve_path.exists() or not weekly_path.exists():
        st.info("Сначала подготовьте агрегаты качества.")
        st.code("uv run --locked python -m credit_scoring.review_curve", language="bash")
        return
    curve = pd.read_csv(curve_path).set_index("Доля проверки, %")
    st.subheader("Охват дефолтов при заданном объёме проверки")
    st.line_chart(curve[["Recall", "Случайный отбор"]] * 100, y_label="Найдено дефолтов, %")
    share = st.slider("Доля заявок на проверку, %", 0, 100, 10)
    row = curve.loc[share]
    flagged, found = int(row["Помечено"]), int(row["Найдено дефолтов"])
    columns = st.columns(4)
    columns[0].metric("Проверить", f"{flagged:,}")
    columns[1].metric("Найдено дефолтов", f"{found:,}")
    columns[2].metric("Recall", f"{row['Recall']:.2%}")
    columns[3].metric("Precision", f"{found / flagged:.2%}" if flagged else "—")
    st.subheader("Gini по неделям")
    st.line_chart(pd.read_csv(weekly_path).set_index("Неделя"), y_label="Gini")


st.set_page_config(page_title="Кредитный скоринг", layout="wide")
st.title("Кредитный скоринг")
page = st.sidebar.radio("Страница", [
    "Синтетический профиль", "Охват дефолтов", "Оценить заявку", "Качество модели",
])
try:
    if page in {"Синтетический профиль", "Охват дефолтов"}:
        selected, metadata = cached_selected(SELECTED_PATH, SELECTED_PATH.stat().st_mtime_ns)
    else:
        model, report = cached_bundle(ARTIFACT_PATH, ARTIFACT_PATH.stat().st_mtime_ns)
except FileNotFoundError:
    st.info("Сначала подготовьте артефакт модели по инструкции в README.")
    st.stop()
except (BadZipFile, CatBoostError, KeyError, ValueError):
    st.error("Не удалось загрузить артефакт модели. Повторите обучение.")
    st.stop()

if page == "Синтетический профиль":
    show_profile(selected, metadata)
elif page == "Охват дефолтов":
    show_review_curve(metadata)
elif page == "Оценить заявку":
    show_application(model, report)
else:
    show_quality(report)
