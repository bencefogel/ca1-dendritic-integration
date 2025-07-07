from concurrent.futures import ProcessPoolExecutor
import glob
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

from scipy import interpolate
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from currentscape_visualization.utils import *
# import altair_saver
# alt.data_transformers.enable("vegafusion")

## we need this to make sure that the files are read in the correct order...
## glob.glob file list are not necessarily ordered, and sort does not order them in the right way.
## we need to use a human sorting or natural soring http://nedbatchelder.com/blog/200712/human_sorting.html

import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def find_csb_start(csb, sp_t, dt=0.2):
    ## detecting complex spike bursts
    # 1. find the spike closest after the csb
    ii = np.flatnonzero((sp_t - csb) > 0)[0]
    allspikes = []
    if ((sp_t[ii] - csb) < 20):
        allspikes.append(sp_t[ii])
        ## collect all spikes before within 20 ms of the start
        move = True
        while (move):
            if (ii > 0):
                if ((sp_t[ii] - sp_t[ii - 1]) < 20):
                    allspikes.append(sp_t[ii - 1])
                    ii = ii - 1
                else:
                    move = False
            else:
                move = False
        start_time = np.min(allspikes)
        return start_time
    else:
        print('no spike found within 20 ms')
        return []


def find_isolated_spikes(sp_t, delta=30, Tmax=10000):
    ## detecting isolated spikes
    ## delta: isolation time in ms
    spt = np.hstack((0, sp_t, Tmax))  # ms
    isolated_spt = []
    for i in np.arange(1, len(spt) - 1):
        if (((spt[i] - spt[i - 1]) > delta) & ((spt[i + 1] - spt[i]) > delta)):
            isolated_spt.append(spt[i])
    i_spt = np.array(isolated_spt)
    return i_spt


def read_combine_current_files(input_dir, current_type, fileprefix, unique_indices=None):
    ####################################################################
    ## reading and combining current share files
    part_neg_list = []
    part_pos_list = []

    negfiles_names = input_dir + '/' + current_type + '/' + fileprefix + '_neg*'
    negfiles = glob.glob(negfiles_names)
    negfiles.sort(key=natural_keys)
    # print(negfiles)

    posfiles_names = input_dir + '/' + current_type + '/' + fileprefix + '_pos*'
    posfiles = glob.glob(posfiles_names)
    posfiles.sort(key=natural_keys)
    # print(posfiles)

    for negfile in negfiles:
        part_neg_list.append(pd.read_csv(negfile, header=0, index_col=0))
    part_neg = pd.concat(part_neg_list, axis=1, ignore_index=True)

    for posfile in posfiles:
        part_pos_list.append(pd.read_csv(posfile, header=0, index_col=0))
    part_pos = pd.concat(part_pos_list, axis=1, ignore_index=True)

    if unique_indices is not None:
        part_neg_unique = part_neg.iloc[:, unique_indices]
        part_pos_unique = part_pos.iloc[:, unique_indices]
        return [part_neg_unique, part_pos_unique]
    else:
        return [part_neg, part_pos]


def plot_currentscape(part_pos, part_neg, vm, taxis, tmin, tmax, filename, return_segs=False, segments_preselected=True,
                      vmin=-69, vmax=-65, partitionby='type'):
    if (segments_preselected):
        part_pos_seg = part_pos
        part_neg_seg = part_neg
        t_seg = taxis
        vm_seg = vm
    else:
        segment_indexes = np.flatnonzero((taxis > tmin) & (taxis < tmax))
        part_pos_seg = part_pos.iloc[:, segment_indexes]
        part_neg_seg = part_neg.iloc[:, segment_indexes]
        t_seg = taxis[segment_indexes]
        vm_seg = vm[segment_indexes]

    # Create charts
    totalpos = create_currsum_pos_chart(part_pos_seg, t_seg)
    currshares_pos, currshares_neg = create_currshares_chart(part_pos_seg, part_neg_seg, t_seg, partitionby)
    vm_chart = create_vm_chart(vm_seg, t_seg, vmin, vmax)

    # Create currentscape
    currentscape = combine_charts(vm_chart, totalpos, currshares_pos, currshares_neg)
    currentscape.save(filename)
    if (return_segs):
        return [part_neg_seg, part_pos_seg, vm_seg]


def plot_currentscape_ave(perc_pos, perc_neg, part_pos, vm, taxis, tmin, tmax, filename, return_segs=False,
                          vmin=-70, vmax=50, partitionby='type'):
    ## part_pos and part_neg are percentages,
    ## the total current is calculated separately

    segment_indexes = np.flatnonzero((taxis >= tmin) & (taxis <= tmax))
    part_pos_seg = part_pos.iloc[:, segment_indexes]
    perc_pos_seg = perc_pos.iloc[:, segment_indexes]
    perc_neg_seg = perc_neg.iloc[:, segment_indexes]
    t_seg = taxis[segment_indexes]
    vm_seg = vm[segment_indexes]

    # Create charts
    totalpos = create_currsum_pos_chart(part_pos_seg, t_seg)
    currshares_pos, currshares_neg = create_currshares_chart(perc_pos_seg, perc_neg_seg, t_seg, partitionby)
    vm_chart = create_vm_chart(vm_seg, t_seg, vmin, vmax)

    # Create currentscape
    currentscape = combine_charts(vm_chart, totalpos, currshares_pos, currshares_neg)
    currentscape.save(filename)
    if (return_segs):
        return [part_neg_seg, part_pos_seg, vm_seg]


####################################################
## Find the indices nearest to a set of selected timepoints
####################################################

def get_nearest_index(t_target, t_orig):
    ## t_target: selected timepoints (5kHz)
    ## t_orig: original time points - the index of which will be returned

    event_index = np.searchsorted(t_orig, t_target, side="left")
    # check wether we need the one before or after...
    min_before = np.argmin(
        np.vstack((np.abs(t_orig[event_index] - t_target), np.abs(t_orig[event_index - 1] - t_target))), axis=0)
    event_index_ret = event_index - min_before
    return event_index_ret


def plot_CS_percent_ave(part_pos_list, part_neg_list, vm_list, taxis, part_neg_seg, filename, vmin=-70, vmax=50,
                        partitionby='type'):
    ########################################################
    ## back to numpy arrays from list

    nrows = part_pos_list[0].shape[0]
    ncols = part_pos_list[0].shape[1]
    arr_pos = np.zeros((nrows, ncols, len(part_pos_list)))
    arr_neg = np.zeros((nrows, ncols, len(part_pos_list)))

    for i in range(len(part_pos_list)):
        arr_pos[:, :, i] = part_pos_list[i]
        arr_neg[:, :, i] = part_neg_list[i]

    arr_Vm = np.array(vm_list)

    ######################################################
    ## averaging - total current
    ######################################################
    ave_pos = pd.DataFrame(np.mean(arr_pos, axis=2), index=part_neg_seg.index)

    ######################################################
    ## normalizing the current contributions in each event
    ######################################################
    arr_pos_perc = arr_pos.copy()
    arr_neg_perc = arr_neg.copy()

    for i in range(arr_pos.shape[2]):
        arr_pos_perc[:, :, i] = arr_pos[:, :, i] / np.sum(arr_pos[:, :, i], axis=0)
    for i in range(arr_neg.shape[2]):
        arr_neg_perc[:, :, i] = -1 * arr_neg[:, :, i] / np.sum(arr_neg[:, :, i], axis=0)

    ave_pos_perc = pd.DataFrame(np.mean(arr_pos_perc, axis=2), index=part_neg_seg.index)
    ave_neg_perc = pd.DataFrame(np.mean(arr_neg_perc, axis=2), index=part_neg_seg.index)

    ######################################################
    ## normalizing the current contributions in each event
    ######################################################

    plot_currentscape_ave(ave_pos_perc, ave_neg_perc, ave_pos, np.mean(arr_Vm, axis=0), taxis, min(taxis), max(taxis),
                          filename, return_segs=False, vmin=vmin, vmax=vmax, partitionby=partitionby)


# Input Data
fname = 'together_ca_280_480_correct'
nsyn = 20

nsyns = {8:0, 10:1, 15:2, 20:3}

target = 'soma'
input_dir = f'L:/branch108/correct_time/{fname}'
partitioned_dir = os.path.join(input_dir, 'results_region')

part_type_neg = os.path.join(partitioned_dir, f'neg_{nsyn}.csv')
part_type_pos = os.path.join(partitioned_dir, f'pos_{nsyn}.csv')

part_pos = pd.read_csv(part_type_pos, index_col=0)
part_neg = pd.read_csv(part_type_neg, index_col=0)
taxis = np.load(os.path.join(partitioned_dir, f'taxis_{nsyn}.npy'))[:part_pos.shape[1]]

with open(f'L:/branch108/correct_time/{fname}.pkl', 'rb') as f:
    sim_data = pickle.load(f)
v_idx = np.where(np.array(sim_data[nsyns[nsyn]]['membrane_potential_data'][0]).astype(str) == f'{target}(0.5)')
v_target = np.array(sim_data[nsyns[nsyn]]['membrane_potential_data'][1])[v_idx].squeeze()

tmin = 280
tmax = 380
filename = f'{fname}_{nsyn}_region.pdf'
partitionby = 'region'

# Run Visualization
plot_currentscape(part_pos, part_neg, v_target, taxis, tmin, tmax, filename, return_segs=False, segments_preselected=False, partitionby=partitionby)