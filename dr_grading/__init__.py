import importlib as _importlib

_submodules = [
    "datasets",
    "models",
]

_functions = [
]

__all__ = _submodules + _functions


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"dr_grading.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'dr_grading' has no attribute '{name}'"
            )
