import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Load training data
filename = 'students- knowledge-level-data-numeric.txt'
df = pd.read_csv(filename, delimiter=',', header=0)
print(df)

features = ['STG','SCG','STR','LPR','PEG']
target = ['UNS']

# Separating out the features
X = df[features]

print(X.describe())

# Separating out the target (class)
y = df['UNS']

fig = px.scatter_matrix(df, dimensions=['STG','SCG','STR','LPR','PEG'], color='UNS')
fig.show()

# Standardizing the features
X_norm = StandardScaler().fit_transform(X)

print(X_norm)

n_classes =  np.unique(y).size

# Application of k-means with k = 6
kmeans = KMeans(n_clusters=n_classes, init="random", n_init=10, max_iter=300, algorithm='lloyd')
kmeans.fit(X_norm)

#centroids = kmeans.cluster_centers_
#print(centroids)

# Results of clustering are in labels_ inside the model
df['KMeans_clusters'] = kmeans.labels_
df.head()

# Run PCA on the data and reduce the dimensions in pca_num_components dimensions
reduced_data = PCA(n_components=2).fit_transform(X_norm)
results = pd.DataFrame(data=reduced_data, columns=['pca1','pca2'])

print(y.values.flatten())

sb.scatterplot(x="pca1", y="pca2", hue=y.values.flatten(), data=results)
plt.title('Ground-truth with 2 dimensions')
plt.show()

print(kmeans.labels_)

sb.scatterplot(x="pca1", y="pca2", hue=kmeans.labels_, data=results)
plt.title('K-means Clustering with 2 dimensions')
plt.show()
