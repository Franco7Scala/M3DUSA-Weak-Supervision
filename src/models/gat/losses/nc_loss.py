import torch
import torch.nn as nn


class NCLoss(nn.Module):

    def __init__(self):
        super(NCLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="sum")

    def forward(self, output, target, mask=None):
        output = output[1]
        target = target[1].float()
        if mask is not None:
            return self.criterion(output[mask], target[mask])

        return self.criterion(output, target)


def global_importance_loss(output, target, sample_mask=None):
    if sample_mask is not None:
        gi_sample = output[sample_mask]
        sc_sample = target[sample_mask]

    else:
        gi_sample = output
        sc_sample = target

    gi_sample = gi_sample.view(-1)
    sc_sample = sc_sample.view(-1)
    return torch.sum((gi_sample - sc_sample) ** 2)
