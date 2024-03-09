import importlib
from .base_model import BaseModel
from torch import nn
import sys

sys.path.append("..")

MODEL_REGISTRY = {}


def register_model(name):
    """
    New models types can be added to cogdl with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the models
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate models ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError("Model ({}: {}) must extend BaseModel".format(name, cls.__name__))
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} models.")
            return False
    return True


def build_model(model):
    if isinstance(model, nn.Module):
        if not hasattr(model, 'build_model_from_args'):
            def build_model_from_args(args, hg):
                return model

            model.build_model_from_args = build_model_from_args
        return model
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]


SUPPORTED_MODELS = {
    "CompGCN": "openhgnn.models.CompGCN",
    'RGCN': 'openhgnn.models.RGCN',
    "RGAT": 'openhgnn.models.RGAT',
    'HAN': 'openhgnn.models.HAN',
    'HGT': 'openhgnn.models.HGT',
    'HPN': 'openhgnn.models.HPN',
    'SimpleHGN': 'openhgnn.models.SimpleHGN',
    'HetSANN': 'openhgnn.models.HetSANN',
    'ieHGCN': 'openhgnn.models.ieHGCN',
}

from .CompGCN import CompGCN
from .RGCN import RGCN
from .RGAT import RGAT
from .HAN import HAN
from .HGT import HGT
from .HPN import HPN
from .SimpleHGN import SimpleHGN
from .HetSANN import HetSANN
from .ieHGCN import ieHGCN

