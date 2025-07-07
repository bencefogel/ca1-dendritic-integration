import os
import pickle

from tqdm import tqdm
from simulator.ModelSimulator import ModelSimulator

# simulation parameters
direction = 'IN'
simulated_dend = 108


# generate simulation data
def run_simulation(direction, simulated_dend):
    simulator = ModelSimulator()
    model = simulator.build_model(dends=[simulated_dend])
    simulation_data_single, simulation_data_together = simulator.run_simulation(model, direction)

    with open(f'L:/branch108/check_im/single_noca.pkl', 'wb') as f:
        pickle.dump(simulation_data_single, f)

    with open(f'L:/branch108/check_im/together_noca.pkl', 'wb') as f:
        pickle.dump(simulation_data_together, f)

print(f"Running simulation for dendrite {simulated_dend}")
run_simulation('IN', simulated_dend)
