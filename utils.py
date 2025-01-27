import torch
from models import *
from operator import attrgetter


def set_device(gpu=True):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if gpu:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
