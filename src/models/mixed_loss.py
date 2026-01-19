import torch

from torch import nn
from torch.nn import functional as F


class MixedLoss(nn.Module):

    def __init__(self, weight_ic_classification=0.5, weight_ic_regression=0.5, weight_proxy_regression=1.0):
        super(MixedLoss, self).__init__()
        self.weight_ic_classification = weight_ic_classification
        self.weight_ic_regression = weight_ic_regression
        self.weight_proxy_regression = weight_proxy_regression

    def forward(self, output, target, mask=None):
        if mask is not None:
            ic_classification_loss = _masked_EMDLoss(output[0], target[0], mask) * self.weight_ic_classification
            ic_regression_loss = _masked_RMSELoss(output[1], target[1], mask) * self.weight_ic_regression
            proxy_regression_loss = _masked_RMSELoss(output[2], target[2], mask) * self.weight_proxy_regression

        else:
            ic_classification_loss = _EMDLoss(output[0], target[0]) * self.weight_ic_classification
            ic_regression_loss = _RMSELoss(output[1], target[1]) * self.weight_ic_regression
            proxy_regression_loss = _RMSELoss(output[2], target[2]) * self.weight_proxy_regression

        return ic_classification_loss + ic_regression_loss + proxy_regression_loss


def _RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def _EMDLoss(y_pred, y_true):
    probs = F.softmax(y_pred, dim=1)
    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_target = torch.cumsum(y_true, dim=1)
    return F.mse_loss(cdf_pred, cdf_target, reduction="mean")


def _masked_RMSELoss(y_pred, y_true, mask):
    squared_error = (y_pred - y_true) ** 2
    masked_squared_error = squared_error * mask
    sum_squared_error = masked_squared_error.sum()
    num_valid = mask.sum()
    mse = sum_squared_error / num_valid
    return torch.sqrt(mse)


def _masked_EMDLoss(y_pred, y_true, mask):
    probs = F.softmax(y_pred, dim=1)
    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_target = torch.cumsum(y_true, dim=1)
    raw_loss = F.mse_loss(cdf_pred, cdf_target, reduction="none")
    masked_loss = raw_loss * mask
    sum_loss = masked_loss.sum()
    num_valid = mask.sum()
    return sum_loss / num_valid
