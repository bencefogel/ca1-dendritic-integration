import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fname = 'together_noca_280_480_correct'
data_folder = 'L:/branch108/correct_time'
partitioning_strategy = 'type'
i = 3

together_synapse_dict = {0: 8, 1: 10, 2: 15, 3: 20}
input_dir = os.path.join(data_folder, fname, f'preprocessed_{together_synapse_dict[i]}')
iax = os.path.join(input_dir, 'iax', 'current_values_0_999.csv')
im = os.path.join(input_dir, 'im', 'current_values_0_999.csv')

df_iax = pd.read_csv(iax, index_col=[0, 1])
df_iax.columns = df_iax.columns.astype(int)
df_im = pd.read_csv(im, index_col=[0, 1])
df_im.columns = df_im.columns.astype(int)

if partitioning_strategy == 'type':
    df_im.sort_index(axis=0, level=(0, 1), inplace=True)


# plot input data
def get_segment_iax(segment, df):
    ref_mask = df.index.get_level_values("ref") == segment
    ref_iax = -1 * df[ref_mask]
    par_mask = df.index.get_level_values("par") == segment
    par_iax = df[par_mask]
    df_iax_seg = pd.concat([ref_iax, par_iax], axis=0)
    return df_iax_seg

soma_iax = get_segment_iax('soma', df_iax)
soma_im = df_im.loc['soma']

# Calculate sums
iax_sum = soma_iax.sum(axis=0)
im_sum = soma_im.sum(axis=0)
total_sum = iax_sum + im_sum

# 1. Plot iax components and sum
fig1, axs1 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
# Extract the 'ref' level from the multiindex
ref_labels = soma_iax.index.get_level_values(0)

# Plot each trace with its corresponding label
for trace, label in zip(soma_iax.iterrows(), ref_labels):
    axs1[0].plot(trace[1].values, label=label)

axs1[0].set_title('soma_iax components')
axs1[0].set_ylabel('Current (nA)')
axs1[0].legend(loc='upper right', fontsize='small', ncol=2)  # Adjust legend position and font size as needed

axs1[1].plot(iax_sum, color='black')
axs1[1].set_title('soma_iax sum')
axs1[1].set_xlabel('Time')
axs1[1].set_ylabel('Current (nA)')

fig1.tight_layout()
plt.savefig('iax_components_vs_sum_correct.png')

# 2. Plot im components and sum with legend
fig2, axs2 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Extract the 'itype' index (channel types)
im_labels = soma_im.index

# Plot each trace with its corresponding label
for trace, label in zip(soma_im.iterrows(), im_labels):
    axs2[0].plot(trace[1].values, label=label)

axs2[0].set_title('soma_im components')
axs2[0].set_ylabel('Current (nA)')
axs2[0].legend(loc='upper right', fontsize='small', ncol=2)

# Plot the sum
im_sum = soma_im.sum(axis=0)
axs2[1].plot(im_sum, color='black')
axs2[1].set_title('soma_im sum')
axs2[1].set_xlabel('Time')
axs2[1].set_ylabel('Current (nA)')

fig2.tight_layout()
plt.savefig('im_components_vs_sum_correct.png')

# 3. Plot 3 rows: iax sum, im sum, total sum
fig3, axs3 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axs3[0].plot(iax_sum, color='blue')
axs3[0].set_title('soma_iax sum')
axs3[0].set_ylabel('Current (nA)')

axs3[1].plot(im_sum, color='green')
axs3[1].set_title('soma_im sum')
axs3[1].set_ylabel('Current (nA)')

axs3[2].plot(total_sum, color='red')
axs3[2].set_title('Total current sum')
axs3[2].set_xlabel('Time')
axs3[2].set_ylabel('Current (nA)')

fig3.tight_layout()
plt.savefig('compare_sums_correct.png')
plt.show()


soma_iax = get_segment_iax('soma', df_iax)
soma_im = df_im.loc['soma']

iax_pos = soma_iax.clip(lower=0)
im_pos = soma_im.clip(lower=0)
iax_neg = soma_iax.clip(upper=0)
im_neg = soma_im.clip(upper=0)

pos = np.sum(iax_pos, axis=0) + np.sum(im_pos, axis=0)
neg = np.sum(iax_neg, axis=0) + np.sum(im_neg, axis=0)

t = np.load('L:/branch108/correct_time/together_noca_280_480_correct/results/taxis_20.npy')

fig = plt.figure()
plt.plot(t, pos.values, color='red', label='positive current sum')
plt.plot(t, neg.values, color='blue', label='negative current sum')
plt.legend()
plt.savefig('pos_neg_sum.png')
plt.show()


