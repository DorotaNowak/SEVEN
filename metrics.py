import torch.nn as nn
import torch
import torch.nn.functional as F


class LossGenerative(torch.nn.Module):
    def __init__(self):
        super(LossGenerative, self).__init__()

    def forward(self, x1, x2, output1, output2):
        euclidean_distance1 = F.pairwise_distance(x1, output1, keepdim=True)
        euclidean_distance2 = F.pairwise_distance(x2, output2, keepdim=True)
        loss_generative = torch.sum(euclidean_distance1 + euclidean_distance2)

        return loss_generative


class LossDiscriminative(torch.nn.Module):
    def __init__(self):
        super(LossDiscriminative, self).__init__()

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        p = 2 / (1 + torch.exp(2 * euclidean_distance))
        loss_discriminative = torch.sum(
            -1 / 2 * label * (label + 1) * torch.log(p) - 1 / 2 * label * (label - 1) * torch.log(1 - p))

        return loss_discriminative
