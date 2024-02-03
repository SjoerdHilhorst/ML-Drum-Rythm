import numpy as np


data = np.load("data.npy").transpose()

print(data.shape)

 # data shape: (number_of_timesteps, number_of_drums, number_of_examples)
number_of_examples = data.shape[2]
train_size = int(number_of_examples * 0.7)

# Initialize lists for the training and testing sets
X_train, y_train = [], []
X_test, y_test = [], []

print(f"Number of examples: {number_of_examples}")
print(f"Train size: {train_size}")
print(f"Test size: {number_of_examples - train_size}")
print(f"Number of timesteps: {data.shape[0]}")
print(f"Number of drums: {data.shape[1]}")

print(data[:,:,0].flatten())
print(data[:,:,0].flatten().shape)

example_zero_flattened = data[:,:,0].flatten()

for i in range(len(example_zero_flattened) - 16):
    print(example_zero_flattened[i:i+16])
    print(np.array(range(i, i+16)))
    print(example_zero_flattened[i+16])
    print(i+16)

# # Function to create datasets
# def create_dataset(data, start, end):
#     print(f"Creating dataset from {start} to {end}")
#     X, y = [], []
#     for i in range(start, end):
#         print(f"Creating dataset for example {i}")
#         print(f"For j in range({data.shape[0]} - {n_steps})")
#         for j in range(data.shape[0] - n_steps):
#             print(f"Appending data[{j}:{j + n_steps}, :, {i}] to X")
#             print(np.array(range(j,(j + n_steps))))
#             X.append(data[j:(j + n_steps), :, i].flatten())
#             print(f"Appending data[{j + n_steps}, :, {i}] to y")
#             print(j + n_steps)
#             y.append(data[j + n_steps, :, i])
#     return np.array(X), np.array(y)

# # Prepare training and testing datasets
# X_train, y_train = create_dataset(data, 0, train_size)
# X_test, y_test = create_dataset(data, train_size, number_of_examples)

# X_train, X_test, y_train, y_test