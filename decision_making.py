import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def graph(data, filename):
    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Plot setup
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, cmap='Greys', cbar=False)

    plt.title('Generated Signal')
    plt.xlabel('Time')
    plt.ylabel('DataType')

    plt.tight_layout()
    plt.savefig(filename)

def threshold_signal(data, threshold=0.5):
    return (data > threshold).astype(int)

def probability_signal(probabilities):    
    # Generate a uniform random matrix of the same shape as the input data
    random_matrix = np.random.rand(*probabilities.shape)
    
    # Generate a binary matrix: 1s where random value is less than the corresponding probability
    binary_matrix = (random_matrix < probabilities)
    
    return binary_matrix

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_signal(data, scaling_factor):
    return sigmoid(scaling_factor*(data - 0.5))

initial_signal = np.random.rand(3, 16)
graph(initial_signal, "initial_signal.png")
treshholded_signal = threshold_signal(initial_signal)
graph(treshholded_signal, "treshholded_signal.png")
diced_signal = probability_signal(initial_signal)
graph(diced_signal, "probability_signal.png")
transformed_signal = sigmoid_signal(initial_signal, 8)
graph(transformed_signal, "transformed_signal.png")
diced_transformed_signal = probability_signal(transformed_signal)
graph(diced_transformed_signal, "diced_transformed_signal.png")