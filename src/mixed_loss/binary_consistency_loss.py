import torch.nn as nn
import torch.nn.functional as F


class BinaryConsistencyLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, main_head_output, proxy_head_output):
        main_prob = F.softmax(main_head_output, dim=-1)
        proxy_prob = F.softmax(proxy_head_output, dim=-1)
        return F.mse_loss(main_prob, proxy_prob)
