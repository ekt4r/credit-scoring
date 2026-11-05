"""Сравнение дополнительных блоков признаков на прежних фолдах."""

import argparse
from pathlib import Path

import pandas as pd

from credit_scoring.boosters import make_booster
from credit_scoring.bureau import add_bureau_features
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import FEATURES
from credit_scoring.temporal import evaluate, temporal_folds
from credit_scoring.train_baselines import TRAIN_DIR, load_cases

STATIC_FEATURES = (
    "avgmaxdpdlast9m_3716943P", "avgoutstandbalancel6m_4187114A",
    "maxoutstandbalancel12m_4187113A", "totaldebt_9A", "cntincpaycont9m_3716944L",
    "numinstlswithdpd10_728L", "pctinstlsallpaidlate1d_3546856L",
    "pctinstlsallpaidlat10d_839L",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=["static", "bureau"])
    args = parser.parse_args()

    static = STATIC_FEATURES if args.block == "static" else ()
    cases = add_application_history(load_cases(static), TRAIN_DIR)
    features = [*FEATURES, *HISTORY_FEATURES, *static]
    coverage_column = None
    if args.block == "bureau":
        cases, bureau_features = add_bureau_features(cases, TRAIN_DIR)
        features.extend(bureau_features)
        coverage_column = "bureau_covered"
        cases[coverage_column] = cases["bureau_active_rows"].gt(0) | cases["bureau_closed_rows"].gt(0)
        for start, end, _, validation in temporal_folds(cases):
            print(f"Недели {start}–{end - 1}: покрытие бюро {validation[coverage_column].mean():.2%}",
                  flush=True)
    rows = evaluate(cases, f"CatBoost · history + {args.block}", features,
                    lambda: make_booster("CatBoost"), coverage_column=coverage_column)
    output = Path("artifacts") / f"features-{args.block}.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
