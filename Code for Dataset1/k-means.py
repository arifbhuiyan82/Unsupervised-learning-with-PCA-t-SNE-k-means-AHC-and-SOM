import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns  # Fix import statement here
from sklearn.cluster import KMeans

# Load training data
filename = 'A3-data.txt'
df = pd.read_csv(filename, delimiter=',', header=0)
print(df)

# Define the features and target variable
features = ['x', 'y', 'z', 't']
target = ['class']

# Separating out the features
X = df[features]

# Print summary statistics of the features
print(X.describe())

# Separating out the target (class)
y = df[target]

# Create a scatter matrix plot using Plotly Express
fig = px.scatter_matrix(df, dimensions=['x', 'y', 'z', 't'], color='class')
fig.show()

# Standardize the features
X_norm = StandardScaler().fit_transform(X)

# Use the elbow method to find the optimal number of clusters (k) for K-means
num_clusters = range(1, 30)
kmeans = [KMeans(n_clusters=i) for i in num_clusters]
score = [kmeans[i].fit(X_norm).score(X_norm) for i in range(len(kmeans))]

# Plot the elbow curve
plt.plot(num_clusters, score)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('Elbow Curve for K-means Clustering')
plt.show()

# Determine the number of unique classes for k-means clustering
n_classes = np.unique(y).size

# Apply k-means clustering with the determined number of clusters
kmeans = KMeans(n_clusters=n_classes, init="random", n_init=10, max_iter=300, algorithm='auto')
kmeans.fit(X_norm)

# Add the K-means cluster labels to the DataFrame
df['KMeans_clusters'] = kmeans.labels_
df.head()

# Perform PCA on the data to reduce dimensions to 2 components
reduced_data = PCA(n_components=2).fit_transform(X_norm)
results = pd.DataFrame(data=reduced_data, columns=['pca1', 'pca2'])

# Visualize the ground-truth classes in 2 dimensions
sns.scatterplot(x="pca1", y="pca2", hue=y.values.flatten(), data=results)
plt.title('Ground-truth with 2 dimensions')
plt.show()

# Visualize the K-means clustering results in 2 dimensions
sns.scatterplot(x="pca1", y="pca2", hue=kmeans.labels_, data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

# Perform PCA on the data to reduce dimensions to 3 components
reduced_data = PCA(n_components=3).fit_transform(X_norm)
results = pd.DataFrame(data=reduced_data, columns=['pca1', 'pca2', 'pca3'])

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(results['pca1'], results['pca2'], results['pca3'])
plt.show()
