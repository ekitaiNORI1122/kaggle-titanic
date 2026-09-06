"""ローカルと Kaggle Notebook のどちらで動いているかを判定し、パスを解決する。

このモジュールがあるおかげで、src/ 配下の他のコードは実行環境を意識しなくてよい。
"""

from __future__ import annotations

from pathlib import Path

# Kaggle Notebook 上では /kaggle/input が必ず存在する
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    DATA_RAW = Path("/kaggle/input/titanic")
    # Kaggle では /kaggle/working に置いたものだけが出力として保存される
    OUTPUT = Path("/kaggle/working")
    DATA_PROCESSED = OUTPUT / "processed"
    SUBMISSIONS = OUTPUT
else:
    ROOT = Path(__file__).resolve().parents[1]
    DATA_RAW = ROOT / "data" / "raw"
    OUTPUT = ROOT
    DATA_PROCESSED = ROOT / "data" / "processed"
    SUBMISSIONS = ROOT / "submissions"

TARGET = "Survived"
ID_COL = "PassengerId"
SEED = 42
N_SPLITS = 5


def describe() -> str:
    """今どこで動いているかを一目で確認するための文字列。"""
    env = "Kaggle Notebook" if IS_KAGGLE else "ローカル"
    return f"env={env}  DATA_RAW={DATA_RAW}  SUBMISSIONS={SUBMISSIONS}"
