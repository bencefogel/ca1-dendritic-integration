import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

# load data
fname = 'together_ca_280_480_correct'
data_folder = 'L:/branch108/correct_time'
i = 3

with open(f'{data_folder}/{fname}.pkl', 'rb') as f:
    simulation_data = pickle.load(f)

sim_data = simulation_data[i]

intrinsic_seg = sim_data['intrinsic_data'][0]
intrinsic_array = sim_data['intrinsic_data'][1]

synaptic_seg = sim_data['synaptic_data'][0]
synaptic_array = sim_data['synaptic_data'][1]

start_idx = 0  # start plotting from index 100 (time)

# Find soma indices for each intrinsic current type
soma_indices = {}
for current_type, segments in intrinsic_seg.items():
    # Try to find a segment containing 'soma' (exact match or partial)
    soma_idx = None
    for i, seg_name in enumerate(segments):
        if 'soma(0.5)' in seg_name:
            soma_idx = i
            break
    if soma_idx is not None:
        soma_indices[current_type] = soma_idx
    else:
        # If no soma found, store None (or skip)
        soma_indices[current_type] = None

print("Soma indices found:", soma_indices)

intrinsic_labels = list(intrinsic_array.keys())
intrinsic_data = intrinsic_array

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Plot soma intrinsic currents components
for label in intrinsic_labels:
    soma_row = soma_indices.get(label, None)
    if soma_row is not None:
        axs[0].plot(intrinsic_data[label][soma_row, start_idx:], label=label)

axs[0].set_title('Soma Intrinsic Membrane Currents Components (from index=100)')
axs[0].set_ylabel('Current (nA)')
axs[0].legend(loc='upper right', fontsize='small', ncol=2)

# Plot sum of soma intrinsic currents
intrinsic_sum = np.zeros(intrinsic_data[intrinsic_labels[0]].shape[1] - start_idx)
for label in intrinsic_labels:
    soma_row = soma_indices.get(label, None)
    if soma_row is not None:
        intrinsic_sum += intrinsic_data[label][soma_row, start_idx:]

axs[1].plot(intrinsic_sum, color='black')
axs[1].set_title('Sum of Soma Intrinsic Membrane Currents (from index=100)')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Current (nA)')

fig.tight_layout()
plt.show()
