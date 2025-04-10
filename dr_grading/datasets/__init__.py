import importlib as _importlib

from .STL10 import STL10DataModule

_submodules = []

_functions = [
    "STL10DataModule",
]

__all__ = _submodules + _functions


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"dr_grading.datasets.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'dr_grading.datasets' has no attribute '{name}'"
            )
