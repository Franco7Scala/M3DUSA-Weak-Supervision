from torch import nn
from torch.nn import functional as F
from src.mixed_loss.binary_consistency_loss import BinaryConsistencyLoss


class MixedLoss(nn.Module):

    def __init__(self, weight_main_component=1.0, weight_proxy_component=1.0, weight_consistency=1.0, weights=None):
        super(MixedLoss, self).__init__()
        self.weight_main_component = weight_main_component
        self.weight_proxy_component = weight_proxy_component
        self.weight_consistency = weight_consistency
        self.weights = weights
        self.consistency_loss = BinaryConsistencyLoss()

    def forward(self, output, target, gt_mask=None, surrogate_mask=None):
        if gt_mask is not None:
            main_classification_loss = self._masked_cross_entropy_loss(output[0], target["ground_truth"], gt_mask) * self.weight_main_component

        else:
            main_classification_loss = self._cross_entropy_loss(output[0], target["ground_truth"]) * self.weight_main_component

        surrogate_mask = surrogate_mask & ~gt_mask
        proxy_classification_loss = self._masked_cross_entropy_loss(output[1], target["ground_truth_surrogate"], surrogate_mask) * self.weight_proxy_component
        consistency_loss = self.consistency_loss(output[0], output[1]) * self.weight_consistency
        return main_classification_loss + proxy_classification_loss + consistency_loss

    def _cross_entropy_loss(self, y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, weight=self.weights)

    def _masked_cross_entropy_loss(self, y_pred, y_true, mask):
        y_pred_masked = y_pred[mask]
        y_true_masked = y_true[mask]
        return F.cross_entropy(y_pred_masked, y_true_masked, weight=self.weights, reduction="mean")
