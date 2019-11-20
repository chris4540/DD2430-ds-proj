import pickle
import pandas as pd
import matplotlib.pyplot
from matplotlib import pyplot as plt
import numpy as np
matplotlib.pyplot.switch_backend('Agg')

with open('projected_emb.pkl', 'rb') as f:
    projected_emb = pickle.load(f)

with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)


meta_df = pd.read_csv("deepfashion_data/deepfashion1_categoryData.csv", index_col=None)
df = meta_df[['category', 'label']]
df = df.set_index('label')
df = df.drop_duplicates()
df = df.sort_index()
classes = df['category'].values

fig, ax = plt.subplots(1, figsize = (14, 10))
plt.scatter(projected_emb[:,1], projected_emb[:,0], s = 0.3, c=labels, cmap='Spectral')
plt.setp(ax, xticks = [], yticks = [])
cbar = plt.colorbar(boundaries = np.arange(11)-0.5)
cbar.set_ticks(np.arange(len(classes)))
cbar.set_ticklabels(classes)
plt.title('DeepFashion Embedded')
fig.savefig('deepfashion.png', bbox_inches='tight')
