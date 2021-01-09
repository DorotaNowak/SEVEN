import torch
import torch.nn.functional as F


class LossDiscriminative(torch.nn.Module):
    def __init__(self):
        super(LossDiscriminative, self).__init__()

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        label = label.squeeze()
        p = 2 / (1 + torch.exp(2 * euclidean_distance))
        loss_discriminative = torch.sum(
            -1 / 2 * label * (label + 1) * torch.log(p) - 1 / 2 * label * (label - 1) * torch.log(1 - p))
        return loss_discriminative


class LossGenerative(torch.nn.Module):
    def __init__(self):
        super(LossGenerative, self).__init__()

    def forward(self, input1, input2, output1, output2):
        input1 = input1.flatten(start_dim=1)
        input2 = input2.flatten(start_dim=1)
        output1 = output1.flatten(start_dim=1)
        output2 = output2.flatten(start_dim=1)
        distances = F.pairwise_distance(input1, output1) + F.pairwise_distance(input2, output2)
        return distances.sum()
