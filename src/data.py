"""Titanic の生データ読み込み。ローカル・Kaggle どちらでも同じ呼び方で動く。"""

from __future__ import annotations

import pandas as pd

from . import config


def load_train() -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW / "train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW / "test.csv")


def load_sample_submission() -> pd.DataFrame:
    return pd.read_csv(config.DATA_RAW / "gender_submission.csv")
