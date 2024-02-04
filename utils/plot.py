import matplotlib.pyplot as plt
import numpy as np


def plot(slices, figure_name="generated.png"):
    no_instruments = 4  # Number of instruments
    time_steps = len(slices) // no_instruments  # Assuming slices contain data for all instruments sequentially

    # Set font size
    font = {'size': 40}

    # Ensure we have a correct reshaping of the data
    if len(slices) % no_instruments != 0:
        print("Warning: The total number of slices is not evenly divisible by the number of instruments.")

    # Initialize a data matrix correctly shaped for the number of instruments and time steps
    data = np.array(slices).reshape((no_instruments, time_steps), order='F')

    labels = ['BD', 'S', 'HH', 'C']  # Instrument labels
    fig, axs = plt.subplots(no_instruments, 1, figsize=(8, 4), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
    plt.subplots_adjust(hspace=0.5)  # Adjust space between plots

    for i in range(no_instruments):
        axs[i].imshow(data[i:i + 1], cmap='binary', aspect='equal')  # Change aspect ratio to 'auto'
        axs[i].set_yticks([])  # Remove y-axis ticks
        axs[i].set_ylabel(labels[i], rotation=0, labelpad=20, ha='right')
        axs[i].set_xticks(np.arange(0, time_steps, 16))
        axs[i].set_xticks(np.arange(0, time_steps, 1), minor=True)
        axs[i].tick_params(axis='x', which='major', length=10, width=2)

        for spine in axs[i].spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.close()
