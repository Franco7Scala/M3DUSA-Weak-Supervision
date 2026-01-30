import torch

from torch import nn
from torch.nn import functional as F
from src.models.gat.mixed_loss import _masked_MAELoss


class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target, mask=None):
        if mask is not None:
            ic_regression_loss = _masked_MAELoss(output[1], target[1], mask)

        else:
            ic_regression_loss = _MAELoss(output[1], target[1])

        return ic_regression_loss
