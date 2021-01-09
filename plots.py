import matplotlib.pyplot as plt


# Visualize MNIST data
def plot_MNIST_data(data, targets):
    num_row = 2
    num_col = 5
    # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(data[i][0], cmap='gray')
        ax.set_title('Label: {}'.format(targets[i]))
    plt.tight_layout()
    plt.show()
