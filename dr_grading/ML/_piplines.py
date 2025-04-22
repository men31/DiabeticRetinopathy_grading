"""
Pipeline that
▪ splits data
▪ oversamples the *training folds only* with SMOTE
▪ tunes Logistic Regression, Random Forest & HistGradientBoosting via GridSearchCV
▪ evaluates on the held‑out test set
▪ *returns* every model’s best estimator/params **and** the single overall best model

Author : ChatGPT  (2025‑04‑18)
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Literal, Union

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier  # pip install lightgbm
from xgboost import XGBClassifier  # pip install xgboost
from catboost import CatBoostClassifier  # pip install catboost
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def build_sklearn_pipelines(random_state: int) -> Dict[str, Tuple[Pipeline, Any]]:
    """
    Create Pipelines and parameter grids for all five algorithms.
    Grids are deliberately compact; enlarge them for deeper tuning.
    """
    return {
        # ───────── Logistic Regression ─────────
        "LogisticRegression": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=500,
                            solver="saga",
                            n_jobs=-1,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            [
                {"clf__penalty": ["l1"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
                {"clf__penalty": ["l2"], "clf__C": [0.01, 0.1, 1.0, 10.0]},
                {
                    "clf__penalty": ["elasticnet"],
                    "clf__C": [0.01, 0.1, 1.0, 10.0],
                    "clf__l1_ratio": [0.1, 0.5, 0.9],
                },
            ],
        ),
        # ───────── Decision Tree ─────────
        "DecisionTree": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=42)),
                    ("classifier", DecisionTreeClassifier(random_state=42)),
                ]
            ),
            {
                "classifier__max_depth": [None, 5, 10, 20],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__criterion": ["gini", "entropy"],
            },
        ),
        # ───────── SVC ─────────
        "SVC": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=42)),
                    ("scaler", StandardScaler()),
                    ("classifier", SVC(probability=True, random_state=42)),
                ]
            ),
            {
                "classifier__C": [0.1, 1, 10],
                "classifier__kernel": ["linear", "rbf"],
                "classifier__gamma": ["scale", "auto"],
            },
        ),
        # ───────── Random Forest ─────────
        "RandomForest": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    (
                        "clf",
                        RandomForestClassifier(random_state=random_state, n_jobs=-1),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [300, 600, 900],
                "clf__max_depth": [None, 10, 20],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        # ───────── HistGradientBoosting ─────────
        "HistGradientBoosting": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    (
                        "clf",
                        HistGradientBoostingClassifier(random_state=random_state),
                    ),
                ]
            ),
            {
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_iter": [100, 200],
                "clf__max_depth": [None, 3, 7],
                "clf__l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
    }


def build_gbm_pipelines(
    random_state: int, mode: Literal["binary", "multi"]="binary"
) -> Dict[str, Tuple[Pipeline, Any]]:
    """
    Create Pipelines and parameter grids for all five algorithms.
    Grids are deliberately compact; enlarge them for deeper tuning.
    """
    if mode == "binary":
        xgb_objective = "binary:logistic"
        lgbm_objective = "binary"
        catboost_loss_function = "Logloss"
    elif mode == "multi":
        xgb_objective = "multi:softmax"
        lgbm_objective = "multiclass"
        catboost_loss_function = "MultiClass"
    else:
        raise ValueError("[-] Model must be either 'binary' or 'multi'")
    
    return{
        # ───────── XGBoost ─────────
        "XGBoost": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    (
                        "clf",
                        XGBClassifier(
                            objective=xgb_objective,
                            eval_metric="logloss",
                            # tree_method="hist",
                            n_jobs=-1,
                            random_state=random_state,
                            # device="cuda",
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
        # ───────── LightGBM ─────────
        "LightGBM": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    (
                        "clf",
                        LGBMClassifier(
                            objective=lgbm_objective,
                            n_jobs=-1,
                            random_state=random_state,
                            # eval_metric="logloss",
                            device="gpu",
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [200, 400, 600],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [-1, 7, 15],
                "clf__num_leaves": [31, 63, 127],
                "clf__subsample": [0.8, 1.0],
                "clf__colsample_bytree": [0.8, 1.0],
            },
        ),
        "CatBoost": (
            Pipeline(
                steps=[
                    ("smote", SMOTE(random_state=random_state)),
                    (
                        "clf",
                        CatBoostClassifier(
                            random_seed=random_state,
                            verbose=0,
                            loss_function=catboost_loss_function,
                            task_type="GPU",
                            devices="0",
                        ),
                    ),
                ]
            ),
            {
                "classifier__iterations": [250, 500, 1000],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__depth": [4, 6, 8],
                "classifier__l2_leaf_reg": [1, 3, 5],
                "classifier__border_count": [32, 64],
            },
        ),
    }
