import argparse
import numpy as np
import tensorflow as tf
from settings import settings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def generate_new_bar_slices(model, initial_slices, num_steps, instruments, threshold=0.5):
    generated_slices = initial_slices.copy()
    generated_hypothesis_vector = np.array([])

    print(generated_slices)
    for step in range(num_steps):
        window = generated_slices[- settings['window'] * instruments:]

        # Make predictions for the current slices
        probabilities = model.predict(np.expand_dims(window, axis=0))[0]
        generated_hypothesis_vector = np.append(generated_hypothesis_vector, probabilities)

        # Apply thresholding to obtain binary values
        binary_predictions = (probabilities > threshold).astype(int)

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
    plt.savefig(img_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate beats using a trained model.")
    parser.add_argument('--model_path', default="my_model.keras", help="Path to the trained model file (in .keras format)")
    parser.add_argument('--num_steps', type=int, default=settings["slices_to_generate"], help="Number of bar slices to generate")

    args = parser.parse_args()

    # Load the pre-trained model
    model = load_model(args.model_path)

    # Convert initial_slices to a numpy array
    initial_slices = np.array([1, 1, 1, 0] * settings['window'])

    # Generate new bar slices using the iterative process
    final_slices, hypothesis_vector = generate_new_bar_slices(model, initial_slices, args.num_steps, len(settings["midi_notes"]))

    # Generate image to visualize the generated bar slices
    visualize_bar_slices(final_slices, "generated.png")

    # Generate image to visualize the hypothesis vector
    visualize_bar_slices(hypothesis_vector, "hypothesis.png")


if __name__ == "__main__":
    main()
