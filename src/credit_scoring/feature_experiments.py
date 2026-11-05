"""Сравнение дополнительных блоков признаков на прежних фолдах."""

import argparse
from pathlib import Path

import pandas as pd

from credit_scoring.boosters import make_booster
from credit_scoring.history import HISTORY_FEATURES, add_application_history
from credit_scoring.scoring import FEATURES
from credit_scoring.temporal import evaluate
from credit_scoring.train_baselines import TRAIN_DIR, load_cases

STATIC_FEATURES = (
    "avgmaxdpdlast9m_3716943P", "avgoutstandbalancel6m_4187114A",
    "maxoutstandbalancel12m_4187113A", "totaldebt_9A", "cntincpaycont9m_3716944L",
    "numinstlswithdpd10_728L", "pctinstlsallpaidlate1d_3546856L",
    "pctinstlsallpaidlat10d_839L",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("block", choices=["static"])
    args = parser.parse_args()

    cases = add_application_history(load_cases(STATIC_FEATURES), TRAIN_DIR)
    features = [*FEATURES, *HISTORY_FEATURES, *STATIC_FEATURES]
    rows = evaluate(cases, f"CatBoost · history + {args.block}", features,
                    lambda: make_booster("CatBoost"))
    output = Path("artifacts") / f"features-{args.block}.csv"
    output.parent.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


if __name__ == "__main__":
    main()
