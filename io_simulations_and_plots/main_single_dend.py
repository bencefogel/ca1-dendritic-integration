import os
import pickle

from tqdm import tqdm
from simulator.ModelSimulator import ModelSimulator

# simulation parameters
direction = 'IN'
stimulated_dend = 108
gcar = 0.006
gkslow = 0.001


# generate simulation data
def run_simulation(direction, stimulated_dend, gcar, gkslow):
    simulator = ModelSimulator()
    model = simulator.build_model(stimulated_dend, gcar, gkslow)
    simulation_data_single, simulation_data_together = simulator.run_simulation(model, direction)

    with open(f'L:/branch108/check_im/single.pkl', 'wb') as f:
        pickle.dump(simulation_data_single, f)

    with open(f'L:/branch108/check_im/together.pkl', 'wb') as f:
        pickle.dump(simulation_data_together, f)

print(f"Running simulation for dendrite {simulated_dend}")
run_simulation('IN', stimulated_dend, gcar, gkslow)
