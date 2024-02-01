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


def generate_beat(model, initial_slices, num_steps, decision_algorithm, instruments):
    generated_slices = initial_slices.copy()
    generated_hypothesis_vector = np.array([])

    print("Initial drums: ", generated_slices)
    for step in range(num_steps):
        window = generated_slices[- settings['window'] * instruments:]

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
    plt.show()


def generate_beats(model_path="my_model.keras",
                   num_steps=settings["slices_to_generate"],
                   decision_algorithm=threshold_signal,
                   save_path="linear_regression/"):
    """
    Alternative main function to generate beats using the trained model.
    """
    # Load the pre-trained model
    model = load_model(os.path.join(model_path))

    # Get number of instruments/drums
    instruments = len(settings["midi_notes"])

    # Convert initial_slices to a numpy array
    initial_slices = np.array([
                                  1, 0, 0, 1,
                                  0, 0, 0, 1,
                                  0, 1, 0, 1,
                                  0, 0, 0, 1,
                              ] * instruments)

    # Generate new bar slices using the iterative process
    final_slices, hypothesis_vector = generate_beat(model, initial_slices, num_steps,
                                                    decision_algorithm=decision_algorithm,
                                                    instruments=instruments)

    # Generate image to visualize the generated bar slices
    # visualize_bar_slices(final_slices, "generated.png")

    # Generate image to visualize the hypothesis vector
    # visualize_bar_slices(hypothesis_vector, "hypothesis.png")
    # print(final_slices)

    # Create figure name
    figure_name = save_path + "generated"

    if decision_algorithm == threshold_signal:
        figure_name += "_threshold_" + str(settings['threshold'])
    elif decision_algorithm == probability_signal:
        figure_name += "_probability"
    elif decision_algorithm == combined_decision_algorithm:
        figure_name += "_combined_" + str(settings['scaling_factor'])

    figure_name += ".png"

    # Plot the generated slices
    plot(final_slices, figure_name=os.path.join("img", "", figure_name))


def main():
    parser = argparse.ArgumentParser(description="Generate beats using a trained model.")
    parser.add_argument('--model_path', default="my_model.keras", help="Path to the trained model file (in .keras format)")
    parser.add_argument('--num_steps', type=int, default=settings["slices_to_generate"], help="Number of bar slices to generate")

    args = parser.parse_args()

    generate_beats(args.model_path, args.num_steps)

if __name__ == "__main__":
    main()
