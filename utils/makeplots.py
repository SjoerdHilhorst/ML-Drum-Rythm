import numpy as np
import matplotlib.pyplot as plt
# Load data from .npy file
data = np.load('data.npy')

# Now you can use the 'data' variable, which contains the numpy array
print(data.shape)


data = data[8]



# Create subplots for each instrument
fig, axs = plt.subplots(4, 1, figsize=(8,4), sharex=True, sharey=True, gridspec_kw={'hspace': 0.02})
plt.subplots_adjust(hspace=10)  # You can adjust the value as needed

labels = ['BD', 'S', 'HH', 'C' ]

# Loop through each instrument and plot
for i in range(4):
    axs[i].imshow(data[i:i+1], cmap='binary', aspect='equal')  # Set aspect ratio to 'equal'
    # axs[i].set_xlabel('Time')
    axs[i].set_yticks([])  # Remove y-axis ticks
    axs[i].set_ylabel('{}'.format(labels[i]), rotation=0, labelpad=20, ha='right')  # Set ylabel to the left
    axs[i].set_xticks(np.arange(0, 65, 16))  # Set time ticks every 16 units
    axs[i].set_xticks(np.arange(0, 65, 1), minor=True)  
    axs[i].tick_params(axis='x', which='major', length=10, width=2)  # Set major tick parameters

    for spine in axs[i].spines.values():
        spine.set_visible(False)
    for x in range(data.shape[1]):
        for y in range(data.shape[0]):
            if data[y, x] == 1:
                rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, edgecolor='white', linewidth=0.5, facecolor='none')
                axs[i].add_patch(rect)


# Adjust layout
plt.tight_layout()
plt.show()





# # Plotting the array as a heatmap with square cells
# plt.figure(figsize=(10, 5))
# plt.imshow(data, cmap='binary', aspect='equal')  # Set aspect='equal' for square cells
# plt.xlabel('Time')
# plt.ylabel('Instruments')
# plt.title('Presence of Instruments Over Time')
# # plt.colorbar(label='Presence (0: Absent, 1: Present)')
# plt.show()




# probabilities = np.mean(data, axis=(0, 2))

# print("Instrument probabilities:")
# for i, prob in enumerate(probabilities):
#     print(f"Instrument {i + 1}: {prob}")