import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import time  # Import the 'time' module for time tracking

# Load training data
filename = 'A3-data.txt'
df = pd.read_csv(filename, delimiter=',', header=0)

# Define features and target
features = ['x', 'y', 'z', 't']
target = ['class']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target (class)
y = df.loc[:, target].values

# Standardizing the features using StandardScaler
x = StandardScaler().fit_transform(x)

# Record the start time for performance measurement
time_start = time.time()

# Apply t-SNE (t-Distributed Stochastic Neighbor Embedding) dimensionality reduction
tsne = TSNE(n_components=2, verbose=1)
tsne_result = tsne.fit_transform(x)

# Print t-SNE completion message and elapsed time
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# Print shape information for the original and transformed data
print("original shape:   ", x.shape)
print("transformed shape:", tsne_result.shape)

# Plot the t-SNE result with color-coded labels
tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'class': y[:,0]})
fig, ax = plt.subplots(1)
sns.scatterplot(x='tsne_1', y='tsne_2', hue='class', data=tsne_result_df, ax=ax, s=10, palette=sns.color_palette(n_colors=6))
lim = (tsne_result.min()-5, tsne_result.max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
plt.show()
