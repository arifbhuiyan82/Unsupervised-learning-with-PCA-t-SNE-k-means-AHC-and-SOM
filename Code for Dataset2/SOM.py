import numpy as np
import pandas as pd
from minisom import MiniSom

# Load the 'students- knowledge-level-data-numeric.txt' dataset
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

# Store winning neurons
winners = []
for x in X:
    winners.append(som.winner(x))

# Add the winning neuron coordinates to the DataFrame
winners_df = pd.DataFrame(winners, columns=['Winner_X', 'Winner_Y'])
result_df = pd.concat([df, winners_df], axis=1)

# Display the U-Matrix, Component Planes, and the updated DataFrame
print("U-Matrix:")
print(umatrix)
print("\nComponent Planes:")
for i, var_name in enumerate(df.columns[:-1]):
    print(f'Component Plane for {var_name}:')
    print(component_planes[i])

print("\nUpdated DataFrame with Winning Neurons:")
print(result_df)
