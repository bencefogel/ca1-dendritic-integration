import pickle
import os

import numpy as np

from currentscape_calculator.CurrentscapeCalculator import CurrentscapeCalculator
from datasaver.DataSaver import DataSaver
from preprocessor.Preprocessor import Preprocessor

# set parameters
target = 'soma'
partitioning_strategy = 'type'

# load data
fname = 'together_ca_280_480_correct'
data_folder = 'L:/branch108/correct_time'

# Preprocessing
# with open(f'{data_folder}/simulation_data_extension.pkl', 'rb') as f:
#     simulation_data_extension = pickle.load(f)
#
# with open(f'{data_folder}/{fname}.pkl', 'rb') as f:
#     simulation_data = pickle.load(f)
#
# for i, data in enumerate(simulation_data):
#     sim_data = data
#     sim_data['areas'] = simulation_data_extension['areas']
#     sim_data['connections'] = simulation_data_extension['connections']
#
#     together_synapse_dict = {0: 8, 1: 10, 2: 15, 3: 20}
#
#     # preprocess data
#     preprocessor = Preprocessor(sim_data)
#     im = preprocessor.preprocess_membrane_currents()
#     iax = preprocessor.preprocess_axial_currents()
#
#     # save preprocessed data
#     output_directory = os.path.join(data_folder, fname)
#     preprocessed_im_directory = f'preprocessed_{together_synapse_dict[i]}/im'
#     preprocessed_iax_directory = f'preprocessed_{together_synapse_dict[i]}/iax'
#     preprocessed_datasaver = DataSaver(columns_in_chunk=None)
#     preprocessed_datasaver.save_in_chunks(im, os.path.join(output_directory, preprocessed_im_directory), 'im')
#     preprocessed_datasaver.save_in_chunks(iax, os.path.join(output_directory, preprocessed_iax_directory), 'iax')
#     preprocessed_datasaver.save_time_axis(os.path.join(output_directory, f'taxis_{together_synapse_dict[i]}'), sim_data['taxis'])


#%%
# partition axial currents of the target (can be type-or region-specific)
## first I ran the preprocessing for all simulation datas
## then I ran the partitioning individually

# set parameters
target = 'soma'
partitioning_strategy = 'region'

# load data
fname = 'together_ca_280_480_correct'
data_folder = 'L:/branch108/correct_time'

for i in range(4):

    output_directory = os.path.join(data_folder, fname)
    input_directory = os.path.join(output_directory, fname, 'preprocessed')
    regions_list_directory = 'C:/Users/Bence/Desktop/projektek/CA1_nmda/partitioning/region_list'
    currentscape_calculator = CurrentscapeCalculator(target, partitioning_strategy, regions_list_directory)

    together_synapse_dict = {0: 8, 1: 10, 2: 15, 3: 20}
    input_dir = os.path.join(data_folder, fname, f'preprocessed_{together_synapse_dict[i]}')
    try:
        iax = os.path.join(input_dir, 'iax', 'current_values_0_999.csv')
        im = os.path.join(input_dir, 'im', 'current_values_0_999.csv')
        im_part_pos, im_part_neg = currentscape_calculator.calculate_currentscape(iax, im, timepoints=None)
    except FileNotFoundError:
        iax = os.path.join(input_dir, 'iax', 'current_values_0_1000.csv')
        im = os.path.join(input_dir, 'im', 'current_values_0_1000.csv')
        im_part_pos, im_part_neg = currentscape_calculator.calculate_currentscape(iax, im, timepoints=None)

    results_dir = os.path.join(data_folder, fname, 'results_region')
    os.makedirs(results_dir, exist_ok=True)

    im_part_pos.to_csv(os.path.join(results_dir, f'pos_{together_synapse_dict[i]}.csv'))
    im_part_neg.to_csv(os.path.join(results_dir, f'neg_{together_synapse_dict[i]}.csv'))
