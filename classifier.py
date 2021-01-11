import torch
import torch.nn.functional as F
import numpy as np


def test_classification(fnet, test_data, test_targets):
    patterns = []

    for i in range(10):
        patterns.append(test_data[np.where(test_targets == i)[0][4]])

    true = 0
    for test_idx in range(800, 850):

        distances = []
        img0 = test_data[test_idx].unsqueeze(0)
        true_label = test_targets[test_idx]

        for i in range(10):
            img1 = patterns[i].unsqueeze(0)
            img0, img1 = img0.cuda(), img1.cuda()

            output_f0, output_f1 = fnet(img0, img1)

            euclidean_distance = F.pairwise_distance(output_f0, output_f1)
            distances.append(euclidean_distance[0])
        min_idx = np.argmin(distances)
        if min_idx == true_label:
            true += 1
    print(true)