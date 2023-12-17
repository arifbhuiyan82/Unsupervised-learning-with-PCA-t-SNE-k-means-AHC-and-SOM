import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

# Load training data
filename = 'A3-data.txt'
df = pd.read_csv(filename, delimiter=',', header=0)

# Print the loaded dataset
print(df)

# Define the feature columns and target column
features = ['x', 'y', 'z', 't']
target = ['class']

# Separating out the features from the dataset
X = df.loc[:, features].values

# Define a function to plot a dendrogram from a hierarchical clustering model
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # Create an array to store the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # Leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create a linkage matrix
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# Create an Agglomerative Clustering model with average linkage
# and set distance_threshold=0 to compute the full tree
model = AgglomerativeClustering(affinity='euclidean', linkage='average', distance_threshold=0, n_clusters=None)

# Fit the model to the feature data
model = model.fit(X)

# Plot the dendrogram with a title
plt.title("Hierarchical Clustering Dendrogram using average linkage")

# Plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=3)

# Add labels to the plot
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

# Show the dendrogram plot
plt.show()
