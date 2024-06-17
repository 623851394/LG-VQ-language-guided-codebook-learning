import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction,target)
        return loss, {}


class BCELossWithQuant(nn.Module):
    def __init__(self, codebook_weight=1., nceloss_weight=0.1, mskloss_weight=0.1,wrsloss_weight=0.1):
        super().__init__()
        self.codebook_weight = codebook_weight

        self.nceloss_weight = nceloss_weight
        self.mskloss_weight = mskloss_weight
        self.wrsloss_weight = wrsloss_weight


    def forward(self, qloss, target, prediction, split, nceloss=None, mskloss=None, wrsloss=None, isValid=False):
        bce_loss = F.binary_cross_entropy_with_logits(prediction,target)
        loss = bce_loss + self.codebook_weight*qloss

        if isValid == False:
            loss += self.nceloss_weight * nceloss + self.mskloss_weight * mskloss + self.wrsloss_weight * wrsloss

        return loss, {"{}/total_loss".format(split): loss.clone().detach().mean(),
                      "{}/rec_loss".format(split): bce_loss.detach().mean(),
                      "{}/bce_loss".format(split): bce_loss.detach().mean(),
                      "{}/quant_loss".format(split): qloss.detach().mean()
                      }
