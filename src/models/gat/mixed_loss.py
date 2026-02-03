import torch

from torch import nn
from torch.nn import functional as F
from src.models.gat.consistency_loss import ConsistencyLoss


class MixedLoss(nn.Module):

    def __init__(self, weight_ic_classification=1.0, weight_ic_regression=1.0, weight_proxy_regression=1.0, weight_consistency=1.0):
        super(MixedLoss, self).__init__()
        self.weight_ic_classification = weight_ic_classification
        self.weight_ic_regression = weight_ic_regression
        self.weight_proxy_regression = weight_proxy_regression
        self.weight_consistency = weight_consistency
        self.consistency_loss = ConsistencyLoss()

    def forward(self, output, target, mask=None):
        if mask is not None:
            ic_classification_loss = _masked_EMDLoss(output[0], target[0], mask) * self.weight_ic_classification
            ic_regression_loss = _masked_MAELoss(output[1], target[1], mask) * self.weight_ic_regression

        else:
            ic_classification_loss = _EMDLoss(output[0], target[0]) * self.weight_ic_classification
            ic_regression_loss = _MAELoss(output[1], target[1]) * self.weight_ic_regression

        consistency_loss = self.consistency_loss(output[0], output[1]) * self.weight_consistency
        proxy_regression_loss = _MAELoss(output[2], target[2]) * self.weight_proxy_regression
        return ic_classification_loss + ic_regression_loss + proxy_regression_loss + consistency_loss


def _RMSELoss(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))


def _MAELoss(y_pred, y_true):
    return torch.mean(torch.abs(y_pred - y_true))


def _BCELoss(y_pred, y_true):
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    loss = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    return torch.mean(loss)


def _CrossEntropyLoss(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


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


def _masked_MAELoss(y_pred, y_true, mask):
    absolute_error = torch.abs(y_pred - y_true)
    masked_absolute_error = absolute_error * mask
    sum_absolute_error = masked_absolute_error.sum()
    num_valid = mask.sum()
    return sum_absolute_error / num_valid


def _masked_BCELoss(y_pred, y_true, mask):
    eps = 1e-7
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    loss_per_element = -(y_true * torch.log(y_pred) + (1.0 - y_true) * torch.log(1.0 - y_pred))
    masked_loss = loss_per_element * mask
    sum_loss = masked_loss.sum()
    num_valid = mask.sum()
    return sum_loss / (num_valid + 1e-8)


def _masked_CrossEntropyLoss(y_pred, y_true, mask):
    element_loss = F.cross_entropy(y_pred, y_true, reduction="none")
    masked_loss = element_loss * mask
    sum_loss = masked_loss.sum()
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    return sum_loss / num_valid


def _masked_EMDLoss(y_pred, y_true, mask):
    probs = F.softmax(y_pred, dim=1)
    cdf_pred = torch.cumsum(probs, dim=1)
    cdf_target = torch.cumsum(y_true, dim=1)
    raw_loss = F.mse_loss(cdf_pred, cdf_target, reduction="none")
    masked_loss = raw_loss * mask
    sum_loss = masked_loss.sum()
    num_valid = mask.sum()
    return sum_loss / num_valid
