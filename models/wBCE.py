import torch
import torch.nn as nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=None, neg_weight=None):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight

    def forward(self, outputs, targets):
        eps = 1e-7
        outputs = torch.clamp(outputs, eps, 1 - eps)

        if self.pos_weight is None:
            self.pos_weight = 1.0
        if self.neg_weight is None:
            self.neg_weight = 1.0

        loss = -(
            self.pos_weight * targets * torch.log(outputs) +
            self.neg_weight * (1 - targets) * torch.log(1 - outputs)
        )
        return loss.mean()
