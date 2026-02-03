import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, class_logits, reg_output):
        probs = F.softmax(class_logits, dim=1)
        expected_class = torch.sum(probs * torch.arange(class_logits.shape[1]).float().to(class_logits.device), dim=1)
        reg_output = reg_output.reshape(-1)
        # normalizing expected class and reg_output to have mean 0 and std 1
        cls_mean = expected_class.mean()
        cls_std = expected_class.std() + self.eps
        cls_z_score = (expected_class - cls_mean) / cls_std
        reg_mean = reg_output.mean()
        reg_std = reg_output.std() + self.eps
        reg_z_score = (reg_output - reg_mean) / reg_std
        # return mean squared error between cls_z_score and reg_z_score
        return F.mse_loss(cls_z_score, reg_z_score)
