from typing import Dict, Any, Union
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from joblib import dump, load


@dataclass(frozen=True)
class ModelResult:
    """Container for a model’s CV & test‑set performance."""

    best_estimator: Pipeline
    best_params: Dict[str, Any]
    cv_f1: float
    test_f1: float
    test_acc: float
    report: str

def save_model(model: Union[ModelResult, Pipeline], filename: str) -> None:
    """Save the model to a file using joblib."""
    dump(model, filename)

def load_model(filename: str) -> Union[ModelResult, Pipeline]:
    """Load the model from a file using joblib."""
    return load(filename)