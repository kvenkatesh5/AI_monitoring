"""
Package `__init__` file with `load_model` function
"""
from .base import Model
from .conv_autoencoder import ConvAutoEncoder
from .supervised_cnn import SupervisedCNN
from .supervised_ctr import SupervisedCTR


def load_model(options: dict) -> Model:
    if options["method"] == ConvAutoEncoder._key():
        model = ConvAutoEncoder(options)
    elif options["method"] == SupervisedCNN._key():
        model = SupervisedCNN(options)
    elif options["method"] == SupervisedCTR._key():
        model = SupervisedCTR(options)
    else:
        raise NotImplementedError(f'requested model not implemented: {options["method"]}')

    return model