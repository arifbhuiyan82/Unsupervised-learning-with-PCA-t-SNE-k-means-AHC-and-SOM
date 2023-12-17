import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load training data
filename = 'A3-data.txt'
df = pd.read_csv(filename, delimiter=',', header=0)
print(df)

# Define the features and target variable
features = ['x', 'y', 'z', 't']
target = ['class']

# Separating out the features
x = df.loc[:, features].values

# Separating out the target (class)
y = df.loc[:, target].values

# Standardize the features
x = StandardScaler().fit_transform(x)

# Perform PCA with 2 principal components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])

# Print shapes of original and transformed data
print("original shape:   ", x.shape)
print("transformed shape:", principal_components.shape)
print(principal_df)

# Combine principal components with the 'class' column
final_df = pd.concat([principal_df, df[['class']]], axis=1)
print(final_df)

# Create a scatter plot to visualize the 2D PCA results
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)

# Define unique class labels and corresponding colors
targets = sorted(df['class'].unique())
colors = ['r', 'g', 'b', 'y', 'm', 'c']

# Plot each class separately with a different color
for target, color in zip(targets, colors):
    indicesToKeep = final_df['class'] == target
    ax.scatter(final_df.loc[indicesToKeep, 'principal component 1'],
               final_df.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=50)
ax.legend(targets)
ax.grid()
plt.show()
