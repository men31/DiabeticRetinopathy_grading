"""
Pipeline that
▪ splits data
▪ oversamples the *training folds only* with SMOTE
▪ tunes Logistic Regression, Random Forest & HistGradientBoosting via GridSearchCV
▪ evaluates on the held‑out test set
▪ *returns* every model’s best estimator/params **and** the single overall best model
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from imblearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    get_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)

from ._utils import ModelResult, save_model
from ._piplines import build_sklearn_pipelines


def train_evaluate_models(
    X: Union[pd.DataFrame, np.ndarray, None] = None,
    y: Union[pd.Series, np.ndarray, None] = None,
    X_train: Union[pd.DataFrame, np.ndarray, None] = None,
    y_train: Union[pd.DataFrame, np.ndarray, None] = None,
    X_test: Union[pd.DataFrame, np.ndarray, None] = None,
    y_test: Union[pd.DataFrame, np.ndarray, None] = None,
    *,
    pipeline: Optional[Pipeline] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    n_splits: int = 5,
    scoring: str = "f1_macro",
    save_model_dir: Optional[str] = None,
) -> Tuple[Dict[str, ModelResult], ModelResult]:
    """
    Full workflow: split, oversample, CV‑tune, evaluate & return results.

    Returns
    -------
    all_results : dict[str, ModelResult]
        Each key is the model label; value holds best estimator, params & metrics.
    best_overall : ModelResult
        The single model with the highest CV F1 score (tie‑broken by test F1).
    """
    # 1) train / test split
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scorer = get_scorer(scoring)

    all_results: Dict[str, ModelResult] = {}

    # 2) build pipelines & run GridSearchCV
    if pipeline is None:
        pipeline = build_sklearn_pipelines(random_state)
    pbar = tqdm(
        total=len(pipeline), desc="Training models", unit="model", leave=False
    )
    for label, (pipe, grid) in pipeline.items():
        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        y_pred = gs.predict(X_test)

        all_results[label] = ModelResult(
            best_estimator=gs.best_estimator_,
            best_params=gs.best_params_,
            cv_f1=gs.best_score_,
            test_f1=f1_score(y_test, y_pred, average="macro"),
            test_acc=accuracy_score(y_test, y_pred),
            report=classification_report(y_test, y_pred, zero_division=0),
        )
        # Save the model
        if save_model_dir is not None:
            save_model(all_results[label], os.path.join(save_model_dir, f"model_{label}.joblib"))
        else:
            # Save the model in the current directory
            save_model(all_results[label], f"model_{label}.joblib")
        pbar.update(1)
        pbar.set_postfix_str(f"{label}: CV: {all_results[label].cv_f1:.3f}, F1 macro: {f1_score(y_test, y_pred, average="macro"):.4f}")

    # 3) pick the overall best model (highest CV F1, then test F1)
    best_overall = max(
        all_results.values(),
        key=lambda res: (res.cv_f1, res.test_f1),
    )

    return all_results, best_overall
