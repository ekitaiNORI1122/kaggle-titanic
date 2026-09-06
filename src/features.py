"""特徴量エンジニアリング。

train と test をまとめて処理することで、カテゴリのエンコードや欠損補完が
両者でズレないようにしている。目的変数は一切使わないのでリークはしない。
"""

from __future__ import annotations

import pandas as pd

from . import config

# Name から取れる敬称。まれなものは Rare にまとめる。
_TITLE_MAP = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Lady": "Rare",
    "Countess": "Rare",
    "Capt": "Rare",
    "Col": "Rare",
    "Don": "Rare",
    "Dona": "Rare",
    "Dr": "Rare",
    "Major": "Rare",
    "Rev": "Rare",
    "Sir": "Rare",
    "Jonkheer": "Rare",
}

CATEGORICAL = ["Sex", "Embarked", "Title", "Deck"]

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked",
    "Title",
    "Deck",
    "FamilySize",
    "IsAlone",
    "TicketGroupSize",
]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Title"] = (
        df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().replace(_TITLE_MAP)
    )
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    # 同じチケット番号＝同行者。家族以外の同伴も拾える。
    df["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("size")
    return df


def _impute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Embarked"] = df["Embarked"].fillna("S")
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    # 敬称と客室等級ごとの中央値で埋めると、素の中央値より実態に近い
    df["Age"] = df["Age"].fillna(df.groupby(["Title", "Pclass"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


def build(train: pd.DataFrame, test: pd.DataFrame):
    """(X_train, y_train, X_test) を返す。X は LightGBM にそのまま渡せる形。"""
    n_train = len(train)
    y = train[config.TARGET]

    combined = pd.concat(
        [train.drop(columns=[config.TARGET]), test], axis=0, ignore_index=True
    )
    combined = _impute(_add_features(combined))

    for col in CATEGORICAL:
        combined[col] = combined[col].astype("category")

    X = combined[FEATURES]
    return X.iloc[:n_train].reset_index(drop=True), y, X.iloc[n_train:].reset_index(drop=True)
