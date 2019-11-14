import numpy as np
from matplotlib import pyplot as plt


class FashionMinstConst:
    classes = {
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    }


class PlotConfig:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(embeddings[inds, 0],
                    embeddings[inds, 1], alpha=0.5, color=PlotConfig.colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(FashionMinstConst.classes)

    return fig
