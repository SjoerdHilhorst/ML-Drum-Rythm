import argparse
import numpy as np
import tensorflow as tf
from settings import settings
import matplotlib.pyplot as plt

def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def generate_new_bar_slices(model, initial_slices, num_steps, instruments, threshold=0.5):
    current_slices = initial_slices.copy()
    print(current_slices)
    for step in range(num_steps):
        # Make predictions for the current slices
        predictions = model.predict(np.expand_dims(current_slices, axis=0))[0]

        # Apply thresholding to obtain binary values
        binary_predictions = (predictions > threshold).astype(int)

        # Use the binary predictions as input for the next iteration
        current_slices = current_slices[instruments:]
        current_slices = np.append(current_slices, binary_predictions)
        
        # Print or use the generated slices as needed
        print(f"Generated Bar Slice {step + 1}: {binary_predictions}")

    return current_slices

def plot(slices):
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
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate beats using a trained model.")
    parser.add_argument('--model_path', default="my_model.keras", help="Path to the trained model file (in .keras format)")
    parser.add_argument('--num_steps', type=int, default=settings["slices_to_generate"], help="Number of bar slices to generate")

    args = parser.parse_args()

    # Load the pre-trained model
    model = load_model(args.model_path)

    # Convert initial_slices to a numpy array
    initial_slices = np.array([
        1, 0, 0, 1,
        0, 0, 0, 1,
        0, 1, 0, 1,
        0, 0, 0, 1,  
    ] * 4)

    # Generate new bar slices using the iterative process
    final_slices = generate_new_bar_slices(model, initial_slices, args.num_steps, len(settings["midi_notes"]), 0.5)

    print(final_slices)
    plot(final_slices)
if __name__ == "__main__":
    main()
