# Credit Scoring

Portfolio project on default prediction at loan application time using
[Home Credit – Credit Risk Model Stability](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/data).

The first stage explores labeled applications and depth-zero features. It audits
join coverage, missingness and weekly default rates, then proposes chronological
development and evaluation windows.

## Reproduce the EDA

Install Python 3.11 or later. Sign in to Kaggle, accept the competition rules,
and configure the Kaggle CLI before downloading the data.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt kaggle
mkdir -p data/raw
kaggle competitions download -c home-credit-credit-risk-model-stability -p data/raw
unzip -n data/raw/home-credit-credit-risk-model-stability.zip 'parquet_files/train/train_base.parquet' 'parquet_files/train/train_static*.parquet' 'feature_definitions.csv' -d data/raw
jupyter lab notebooks/01_eda.ipynb
```

Run all cells in order. The notebook uses only the labeled competition training
files. Raw data stays local.
