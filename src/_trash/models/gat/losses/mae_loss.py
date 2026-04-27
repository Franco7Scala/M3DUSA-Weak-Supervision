from torch import nn
from src._trash.models.gat.losses.mixed_loss import _masked_MAELoss, _MAELoss


class MAELoss(nn.Module):

    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, output, target, mask=None):
        if mask is not None:
            ic_regression_loss = _masked_MAELoss(output[1], target[1], mask)

        else:
            ic_regression_loss = _MAELoss(output[1], target[1])

        return ic_regression_loss
