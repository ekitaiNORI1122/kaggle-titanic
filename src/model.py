"""LightGBM による交差検証・予測・提出ファイル作成。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from . import config

DEFAULT_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 16,
    "min_child_samples": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "reg_lambda": 1.0,
    "n_estimators": 1000,
    "verbosity": -1,
    "random_state": config.SEED,
}


def run_cv(X: pd.DataFrame, y: pd.Series, params: dict | None = None):
    """層化 K 分割で学習し、(oof確率, foldごとのモデル) を返す。"""
    params = {**DEFAULT_PARAMS, **(params or {})}
    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)

    oof = np.zeros(len(X))
    models = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr_idx],
            y.iloc[tr_idx],
            # eval_X/eval_y ではなく eval_set を使う。新しい LightGBM では非推奨警告が
            # 出るが、Kaggle Notebook 側の古いバージョンでも動く方を優先している。
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        models.append(model)
        fold_acc = accuracy_score(y.iloc[va_idx], (oof[va_idx] > 0.5).astype(int))
        print(f"  fold {fold}: accuracy = {fold_acc:.4f}  (best_iter={model.best_iteration_})")

    cv_acc = accuracy_score(y, (oof > 0.5).astype(int))
    print(f"CV accuracy = {cv_acc:.4f}")
    return oof, models


def predict(models, X_test: pd.DataFrame) -> np.ndarray:
    """fold ごとのモデルの予測確率を平均する。"""
    return np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)


def make_submission(
    test_ids: pd.Series, proba: np.ndarray, name: str = "sub", threshold: float = 0.5
) -> Path:
    """提出用 CSV を書き出してパスを返す。ファイル名にタイムスタンプを付ける。"""
    config.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = config.SUBMISSIONS / f"{name}_{datetime.now():%Y%m%d_%H%M}.csv"
    pd.DataFrame(
        {config.ID_COL: test_ids, config.TARGET: (proba > threshold).astype(int)}
    ).to_csv(path, index=False)
    print(f"saved: {path}")
    return path


def feature_importance(models) -> pd.DataFrame:
    """fold 平均の重要度を降順で返す。"""
    imp = np.mean([m.feature_importances_ for m in models], axis=0)
    return (
        pd.DataFrame({"feature": models[0].feature_name_, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
