import numpy as np
import random


def create_labeled_data(n_samples, train_targets):
    labeled_pos = []
    labeled_neg = []

    for i in range(10):
        idxs1 = list(np.where(train_targets == i)[0])
        idxs2 = list(np.where(train_targets != i)[0])

        sample1 = random.sample(idxs1, 2 * n_samples)
        sample2 = random.sample(idxs1, n_samples)
        sample3 = random.sample(idxs2, n_samples)

        for i in range(0, 2 * n_samples, 2):
            labeled_pos.append((sample1[i], sample1[i + 1], 1))
        for i in range(n_samples):
            labeled_neg.append((sample2[i], sample3[i], -1))

    random.shuffle(labeled_pos)
    random.shuffle(labeled_neg)
    return labeled_pos, labeled_neg


def create_unlabeled_data(n_samples):
    unlabeled = []

    for i in range(n_samples):
        a = random.randint(0, 59999)
        b = random.randint(0, 59999)
        unlabeled.append((a, b, 0))

    random.shuffle(unlabeled)
    return unlabeled


def create_testing_data(test_targets):
    test = []
    for i in range(10):
        idxs1 = list(np.where(test_targets == i)[0])
        idxs2 = list(np.where(test_targets != i)[0])

        sample1 = random.sample(idxs1, 20)
        sample2 = random.sample(idxs1, 10)
        sample3 = random.sample(idxs2, 10)

        for i in range(0, 20, 2):
            test.append((sample1[i], sample1[i + 1], 1))
        for i in range(10):
            test.append((sample2[i], sample3[i], -1))
    return test
