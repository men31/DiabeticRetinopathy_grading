import importlib as _importlib

from .LightningModelWrapper import LightningModelWrapper

from .Classifier_module import ClassifierHead, ClassiferHead_MoE
from .DenseNet_module import DenseNet161Lightning, DenseNet161, DenseNet161_MoE
from .Swin_module import Swin_S, Swin_V2_B
from .VGG_module import VGG19
from .ResNet_module import ResNet50
from .Inception_module import Inception_V3
from .ViT_module import ViT16
from .Encoder_module import ConvEncoder
from .Unet_module import UNet
from .__utils import load_state_from_ckpt

_submodules = []

_functions = [
    "ClassifierHead",
    "ClassiferHead_MoE",
    "DenseNet161",
    "DenseNet161_MoE",
    "Swin_S",
    "Swin_V2_B",
    "VGG19",
    "ResNet50",
    "Inception_V3",
    "ViT16",
    "ConvEncoder",
    "load_state_from_ckpt",
]

_lighning_functions = [
    "DenseNet161Lightning",
    "LightningModelWrapper",
]

__all__ = _submodules + _functions + _lighning_functions


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"dr_grading.models.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(
                f"Module 'dr_grading.models' has no attribute '{name}'"
            )
