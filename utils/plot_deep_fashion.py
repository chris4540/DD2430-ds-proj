import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
matplotlib.pyplot.switch_backend('Agg')

cat_to_idx = {
"Blazer":        0,
"Blouse":        1,
"Cardigan":      2,
"Dress":         3,
"Jacket":        4,
"Jeans":         5,
"Jumpsuit":      6,
"Leggings":      7,
"Romper":        8,
"Shorts":        9,
"Skirt":        10,
"Sweater":      11,
"Tank":         12,
"Tee":          13,
"Top":          14,
}

colors = {
    "Blazer":        "#1f77b4",
"Blouse":        "#ff7f0e",
"Cardigan":      "#2ca02c",
"Dress":         "#d62728",
"Jacket":        "#9467bd",
"Jeans":         "#8c564b"
"Jumpsuit":      "#e377c2",
"Leggings":      '#7f7f7f',
"Romper":        "#bcbd22",
"Shorts":        "#17becf",
"Skirt":        "#012345",
"Sweater":      "black",
"Tank":         "red",
"Tee":          "green",
"Top":          "yellow",
}
# class FashionMinstConst:
#     classes = {
#         'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
#     }


# class PlotConfig:
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
#               '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
#               '#bcbd22', '#17becf']


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
