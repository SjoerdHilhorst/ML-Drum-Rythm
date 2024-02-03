import argparse
import os

import numpy as np
import tensorflow as tf
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from decision_making import *


def load_model(model_path):
    model_path = os.path.join(settings['models_dir'], model_path)

    if model_path.endswith(".sav"):
        model = pickle.load(open(model_path, 'rb'))
    else:
        model = tf.keras.models.load_model(model_path)
    return model


def generate_beat(model, initial_slices, num_steps, decision_algorithm, instruments, window_size):
    generated_slices = initial_slices.copy()
    generated_hypothesis_vector = np.array([])

    print("Initial drums: ", generated_slices)
    for step in range(num_steps):
        window = generated_slices[- window_size * instruments:]

        # Make predictions for the current slices
        probabilities = model.predict(np.expand_dims(window, axis=0))[0]
        generated_hypothesis_vector = np.append(generated_hypothesis_vector, probabilities)

        # Apply decision algorithm to obtain binary values
        binary_predictions = decision_algorithm(probabilities).astype(int)

        # Append the new slices to the generated slices
        generated_slices = np.append(generated_slices, binary_predictions)
        
        # Print or use the generated slices as needed
        print(f"Generated Bar Slice {step + 1}: {binary_predictions}")

    return generated_slices, generated_hypothesis_vector


def visualize_bar_slices(bar_slices, img_path="generated.png"):
    """
    Visualize the bar slices as a grey-scale heatmap and save to the specified path.
    """

    # Get the instruments from the settings
    instruments = settings["midi_notes"].values()

    # Make sure that the number of bar slices is a multiple of the number of instruments
    assert(bar_slices.shape[0] % len(instruments) == 0)

    # Reshape the bar slices to a 2D array
    generated = pd.DataFrame(bar_slices.reshape(-1, len(instruments)), columns=instruments)

    # Visualize the generated bar slices
    plt.figure(figsize=(10, 5))
    sns.heatmap(generated.transpose(), annot=False, cmap="Greys", cbar=False, xticklabels=True, yticklabels=True)
    plt.title("Generated Bar Slices")
    plt.xlabel("Bars")
    plt.ylabel("Drums")
    plt.savefig(os.path.join("img", img_path))
    plt.show()


def plot(slices, figure_name="generated.png"):
    result_arrays = [[], [], [], []]

    ordered_keys = sorted(settings["midi_notes"].keys())
    for i, single_slice in enumerate(slices):
        result_arrays[i % 4].append(single_slice)

    print(result_arrays)
    # Plot each array with subtitles
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

    for i, result_array in enumerate(result_arrays):
        axs[i].stem(result_array, basefmt='b-', linefmt='b-', markerfmt='bo')
        axs[i].set_title(f'{settings["midi_notes"][ordered_keys[i]]}')

    plt.xlabel('Index')
    plt.savefig(figure_name)
    plt.close()

def get_initial_beat_from_dataset(example_index=0, window_size=16):
    # Load the dataset
    dataset = np.load('data.npy').transpose()

    # Select the example
    example = dataset[:, :, example_index] 

    # Determine the length of the sequence you want to extract for each drum
    sequence_length = window_size

    # Extract the initial sequences for every drum
    # This results in a matrix of shape (sequence_length, number_of_drums)
    return example[:sequence_length, :].flatten()

def generate_beats(model_path="my_model.keras",
                   num_steps=settings["slices_to_generate"],
                   decision_algorithm=threshold_signal,
                   save_path="linear_regression/"):
    """
    Alternative main function to generate beats using the trained model.
    """

    for model_index in range(8, 17):

        # Load the pre-trained model
        model = load_model(os.path.join(f"linear_model_{model_index}.sav"))

        # Get number of instruments/drums
        instruments = len(settings["midi_notes"])

        file_path = os.path.join(f"img/linear_regression/window_size_{model_index}")
        os.makedirs(file_path, exist_ok=True)

        for initial_beat_index in range(100):
            # Convert initial_slices to a numpy array
            initial_slices = get_initial_beat_from_dataset(initial_beat_index, window_size=model_index)

            # Generate new bar slices using the iterative process
            generated_slices, hypothesis = generate_beat(model, initial_slices, num_steps, decision_algorithm=threshold_signal, instruments=instruments, window_size=model_index)
            plot(generated_slices, os.path.join(file_path, f"generated_threshold_signal_{initial_beat_index}.png"))

            generated_slices, hypothesis = generate_beat(model, initial_slices, num_steps, decision_algorithm=probability_signal, instruments=instruments, window_size=model_index)
            plot(generated_slices, os.path.join(file_path, f"generated_probability_signal_{initial_beat_index}.png"))

            generated_slices, hypothesis = generate_beat(model, initial_slices, num_steps, decision_algorithm=combined_decision_algorithm, instruments=instruments, window_size=model_index)
            plot(generated_slices, os.path.join(file_path, f"generated_combined_signal_{initial_beat_index}.png"))

            


def main():
    parser = argparse.ArgumentParser(description="Generate beats using a trained model.")
    parser.add_argument('--model_path', default="linear_model.sav", help="Path to the trained model file (in .keras format)")
    parser.add_argument('--num_steps', type=int, default=settings["slices_to_generate"], help="Number of bar slices to generate")

    args = parser.parse_args()

    generate_beats(args.model_path, args.num_steps)

if __name__ == "__main__":
    main()
