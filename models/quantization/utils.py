import torch.nn as nn

LEGAL_MODULES = [
    nn.Conv2d,
    nn.Linear
]

def check_legal(module):
    """
    Check if module is legal to quantize
    """
    for module_type in LEGAL_MODULES:
        if isinstance(module, module_type):
            return True
    return False
