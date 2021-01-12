import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Visualize MNIST data
def plot_MNIST_data(data, targets):
    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(10):
        idx = list(np.where(targets == i))[0][0]
        ax = axes[i // num_col, i % num_col]
        ax.imshow(data[idx][0], cmap='gray')
        ax.set_title('Label: {}'.format(targets[idx]))
    plt.tight_layout()
    plt.savefig('images/MNIST_data.png')
    plt.show()


def plot_loss(loss_history, name):
    plt.plot([i for i in range(len(loss_history))], loss_history)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    filename = name.replace(' ', '_').lower()
    path = 'images/' + filename + '.png'
    plt.savefig(path)
    plt.show()


def plot_split(positive, negative):
    pos = [p.cpu().detach().numpy()[0] for p in positive]
    neg = [n.cpu().detach().numpy()[0] for n in negative]

    p1 = sns.kdeplot(pos, shade=True, color="b")
    p2 = sns.kdeplot(neg, shade=True, color="r")
    path = 'images/split.png'
    plt.savefig(path)
    plt.show()
