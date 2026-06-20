# PyTorch ResNet exports (primary implementation)
from chapters.cv.resnet.code.model import (
    BasicBlock,
    Bottleneck,
    ResNet,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

__all__ = [
    'BasicBlock',
    'Bottleneck',
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
    # Legacy TF functions (import on demand to avoid hard TF dependency)
    'identity_block_2',
    'identity_block_3',
    'convolution_block_2',
    'convolution_block_3',
    'init_model',
]


def __getattr__(name):
    """Lazy-import legacy TF functions to avoid hard TensorFlow dependency."""
    _TF_NAMES = {
        'identity_block_2',
        'identity_block_3',
        'convolution_block_2',
        'convolution_block_3',
        'init_model',
    }
    if name in _TF_NAMES:
        from chapters.cv.resnet.code import tf_main

        return getattr(tf_main, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
