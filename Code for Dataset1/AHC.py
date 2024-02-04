import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the training data from 'A3-data.txt' CSV file
filename = 'A3-data.txt'
df = pd.read_csv(filename, delimiter=',', header=0)
print(df)

# Define the features and target variable names
features = ['x', 'y', 'z', 't']
target = ['class']

# Separating out the features
X = df.loc[:, features].values

# Create an Agglomerative Clustering model with 'average' linkage
model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='average', distance_threshold=0)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram using average linkage")

# Perform hierarchical clustering and obtain the linkage matrix
Z = linkage(X, method='average', metric='euclidean')

# Plot the dendrogram
dendrogram(Z, truncate_mode="level", p=3, no_labels=True)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()

# Create an Agglomerative Clustering model with 'complete' linkage
model = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='complete', distance_threshold=0)

model = model.fit(X)
plt.title("Hierarchical Clustering Dendrogram using complete linkage")

# Perform hierarchical clustering and obtain the linkage matrix
Z = linkage(X, method='complete', metric='euclidean')

# Plot the dendrogram
dendrogram(Z, truncate_mode="level", p=3, no_labels=True)
plt.xlabel("Number of points in node (or index of point if no parenthesis)")
plt.show()
