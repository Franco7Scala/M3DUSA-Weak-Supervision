import torch

from torch import nn
from torch.nn import functional as F
from src.mixed_loss.binary_consistency_loss import BinaryConsistencyLoss


class MixedLoss(nn.Module):

    def __init__(self, weight_main_component=1.0, weight_proxy_component=1.0, weight_consistency=1.0):
        super(MixedLoss, self).__init__()
        self.weight_main_component = weight_main_component
        self.weight_proxy_component = weight_proxy_component
        self.weight_consistency = weight_consistency
        self.consistency_loss = BinaryConsistencyLoss()

    def forward(self, output, target, mask=None):
        if mask is not None:
            main_classification_loss = _masked_cross_entropy_loss(output[0], target["ground_truth"], mask) * self.weight_main_component

        else:
            main_classification_loss = _cross_entropy_loss(output[0], target["ground_truth"]) * self.weight_main_component

        proxy_classification_loss = _cross_entropy_loss(output[1], target["ground_truth_surrogate"]) * self.weight_proxy_component
        consistency_loss = self.consistency_loss(output[0], output[1]) * self.weight_consistency
        return main_classification_loss + proxy_classification_loss + consistency_loss


def _cross_entropy_loss(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)


def _masked_cross_entropy_loss(y_pred, y_true, mask):
    element_loss = F.cross_entropy(y_pred, y_true, reduction="none")
    masked_loss = element_loss * mask
    sum_loss = masked_loss.sum()
    num_valid = mask.sum()
    if num_valid == 0:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    return sum_loss / num_valid
