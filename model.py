import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_data

n_epochs = 30
batch_size = 32
alpha = 0.05
beta = 0.0001

random_seed = 0
torch.manual_seed(random_seed)
torch.backends.cudnn.enabled = True

train_loader, test_loader = load_data()
# Training data
train_set = enumerate(train_loader)
train_batch_idx, (train_data, train_targets) = next(train_set)

# Testing data
test_set = enumerate(test_loader)
test_batch_idx, (test_data, test_targets) = next(test_set)

from plots import plot_MNIST_data

plot_MNIST_data(train_data, train_targets)

from create_data import create_labeled_data, create_unlabeled_data, create_testing_data

labeled_pos, labeled_neg = create_labeled_data(100, train_targets)
unlabeled = create_unlabeled_data(2000)
test = create_testing_data(test_targets)


def get_accuracy(test, threshold, fnet):
    correct = 0
    for i in range(len(test)):
        img0 = test_data[test[i][0]].unsqueeze(0)
        img1 = test_data[test[i][1]].unsqueeze(0)
        true_label = test[i][2]
        img0, img1 = img0.cuda(), img1.cuda()

        output_f1, output_f2 = fnet(img0, img1)

        euclidean_distance = F.pairwise_distance(output_f1, output_f2)

        if (euclidean_distance > threshold and true_label == -1):
            correct += 1
        if (euclidean_distance <= threshold and true_label == 1):
            correct += 1
    return correct / len(test)


iteration = 0
counter = []
loss_history = []
loss_history_d = []
loss_history_g = []
loss_history_step = []

from networks import FNet, GNet

fnet = FNet().cuda()
gnet = GNet().cuda()

from metrics import LossDiscriminative, LossGenerative

criterion_1 = LossDiscriminative()
criterion_2 = LossGenerative()

optimizer = optim.RMSprop([
    {'params': fnet.parameters(), 'lr': 1e-3},
    {'params': gnet.parameters(), 'lr': 1e-3}])


def reset_grad():
    optimizer.zero_grad()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

print('lala')
train_acc = []
for epoch in range(0, n_epochs):
    print(epoch)
    fnet.train()
    train_loss = 0.0
    train_d_loss = 0.0
    train_g_loss = 0.0
    for i in range(int(len(unlabeled) / 16)):
        f1 = [train_data[labeled_pos[(i * 8 + j) % len(labeled_pos)][0]] for j in range(8)]
        f1 += [train_data[labeled_neg[(i * 8 + j) % len(labeled_neg)][0]] for j in range(8)]
        f1 += [train_data[unlabeled[i * 16 + j][0]] for j in range(16)]
        img0 = torch.stack(f1, 0)
        f2 = [train_data[labeled_pos[(i * 8 + j) % len(labeled_pos)][1]] for j in range(8)]
        f2 += [train_data[labeled_neg[(i * 8 + j) % len(labeled_neg)][1]] for j in range(8)]
        f2 += [train_data[unlabeled[i * 16 + j][1]] for j in range(16)]
        img1 = torch.stack(f2, 0)
        f3 = [labeled_pos[(i * 8 + j) % len(labeled_pos)][2] for j in range(8)]
        f3 += [labeled_neg[(i * 8 + j) % len(labeled_neg)][2] for j in range(8)]
        f3 += [unlabeled[i * 16 + j][2] for j in range(16)]
        label = torch.FloatTensor(f3)

        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

        reset_grad()
        output_f1, output_f2 = fnet(img0, img1)
        loss_d = criterion_1(output_f1, output_f2, label)

        output_g1, output_g2 = gnet(output_f1, output_f2)
        loss_g = criterion_2(img0, img1, output_g1, output_g2)

        reg = 0
        for p in gnet.parameters():
            reg = reg + torch.norm(p)

        for p in fnet.parameters():
            reg = reg + torch.norm(p)

        loss = loss_d + alpha * loss_g + beta * reg
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_d_loss += loss_d
        train_g_loss += loss_g

        if iteration % 100 == 0:
            counter.append(iteration)
            loss_history_step.append(loss.item())
        iteration += 1
    fnet.eval()
    train_acc.append(get_accuracy(test, 0.5, fnet))

    loss_history.append(train_loss)
    loss_history_d.append(train_d_loss)
    loss_history_g.append(train_g_loss)
    print("Epoch number {}\n Current loss {}\n".format(epoch, train_loss))
