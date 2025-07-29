import numpy as np

import simulator.model.simulation as simulation
from numpy import loadtxt
####################### Dendritic integration #######################

def SIM_nsynIteration(model, maxNsyn, tInterval, onset, direction='IN', tstop=300):
# direction should be either 'OUT' or 'IN' (default)
#   ---------------------------------------
#   first each synapse is activated alone
    It = None
    simulation_data_single_list = []
    simulation_data_together_list = []
    # maxNsyn = sum(maxNsyn)
    # for nsyn in np.arange(1, maxNsyn+1): # 1 - N
    for nsyn in np.arange(1, maxNsyn+1):
        if (direction=='OUT'):
            etimes = np.array([[nsyn-1, onset]]) # 0 - N-1
        else:
            etimes = np.array([[maxNsyn - nsyn, onset]]) # N-1 - 0
        fih = simulation.h.FInitializeHandler(1, lambda: initSpikes_dend(model, etimes))

        # def simulate(model, t_stop=100, NMDA=False, recDend=False, i_recDend=11, x_recDend=0.5, spines=False, i_recSpine=11):
        sim_data = simulation.simulate(model, tstop)
        sim_data['etimes'] = etimes
        simulation_data_single_list.append(sim_data)

#   ---------------------------------------
#   next, synapses are activated together
#     for nsyn in np.arange(1, maxNsyn+1):
    for nsyn in [8, 10, 15, 20]:
        etimes = genDSinput(nsyn, maxNsyn, tInterval, onset, direction)
        fih = simulation.h.FInitializeHandler(1, lambda: initSpikes_dend(model, etimes))
        sim_data = simulation.simulate(model, tstop)
        sim_data['etimes'] = etimes
        simulation_data_together_list.append(sim_data)

    return  simulation_data_single_list, simulation_data_together_list


def initSpikes_dend(model, etimes):
    if (len(etimes)>0):
        for s in etimes:
            model.ncAMPAlist[int(s[0])].event(float(s[1]))
            model.ncNMDAlist[int(s[0])].event(float(s[1]))


def genDSinput(nsyn, Nmax, tInterval, onset, direction='OUT'):
    ## a single train with nsyn inputs - either in the in or in the out direction
    times = np.zeros([nsyn, 2])
    if (direction=='OUT'):
        times[:,0] = np.arange(0, nsyn)
    else:
        times[:,0] = np.arange(Nmax-1, Nmax-nsyn-1, -1)
    # # print(times)
    # tt = np.arange(0, nsyn*tInterval, tInterval) + onset
    # # print(tt)
    times[:,1] = np.arange(0, nsyn*tInterval, tInterval)[0:nsyn] + onset
    return times
