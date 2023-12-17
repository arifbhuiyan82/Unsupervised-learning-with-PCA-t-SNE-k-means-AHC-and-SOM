import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom

# Load the 'A3-data.txt' dataset
filename = 'students- knowledge-level-data-numeric.txt'
df = pd.read_csv(filename, delimiter=',', header=0)

# Assuming the last column is the target column (class)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Feature scaling using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Create and train the Self-Organizing Map (SOM)
som = MiniSom(x=10, y=10, input_len=X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# Create U-Matrix
umatrix = som.distance_map().T

# Create Component Planes (Manually compute)
n_variables = X.shape[1]
component_planes = []
for i in range(n_variables):
    component_plane = np.zeros((som.get_weights().shape[0], som.get_weights().shape[1]))
    for r in range(som.get_weights().shape[0]):
        for c in range(som.get_weights().shape[1]):
            component_plane[r, c] = som.get_weights()[r, c][i]
    component_planes.append(component_plane)

# Display U-Matrix
plt.figure(figsize=(8, 8))
plt.title("U-Matrix")
plt.pcolor(umatrix, cmap='viridis_r')
plt.colorbar()

# Display Component Planes
n_rows, n_cols = int(np.ceil(n_variables / 2)), 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
for i, var_name in enumerate(df.columns[:-1]):
    row = i // n_cols
    col = i % n_cols
    ax = axes[row, col]
    ax.set_title(f'Component Plane for {var_name}')
    ax.pcolor(component_planes[i].T, cmap='coolwarm')
    ax.axis('off')

# Remove empty subplots if there are any
if n_variables % 2 != 0:
    fig.delaxes(axes[n_rows - 1, n_cols - 1])

plt.tight_layout()
plt.suptitle("Component Planes", fontsize=16)
plt.subplots_adjust(top=0.9)

# Visualize the SOM with updated markers and colors lists
from pylab import bone, plot, show
bone()
unique_classes = np.unique(y)
markers = ['o', 's', 'D', 'P', 'X', '^', 'v', '<', '>', '1', '2', '3', '4', '8']  # Add more marker symbols if needed
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))  # Use a colormap to generate colors
class_to_marker = {cls: markers[i] for i, cls in enumerate(unique_classes)}
class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}

for i, x in enumerate(X):
    w = som.winner(x)
    cls = y[i]
    marker = class_to_marker[cls]
    color = class_to_color[cls]
    plot(w[0] + 0.5, w[1] + 0.5, marker, markeredgecolor=color, markerfacecolor='None', markersize=10, markeredgewidth=2)

show()