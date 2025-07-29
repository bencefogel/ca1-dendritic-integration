import os
import pickle
import numpy as np

from tqdm import tqdm
from simulator.ModelSimulator import ModelSimulator

# simulation parameters
direction = 'IN'
stimulated_dend = [108]
gcar = 0.00375
gkslow = 0.00519
gcar_trunk = 0.01486
gkslow_trunk = 0.00039


# generate simulation data
def run_simulation(direction, stimulated_dend, gcar, gkslow, gcar_trunk):
    simulator = ModelSimulator()
    model = simulator.build_model(stimulated_dend, gcar, gkslow, gcar_trunk, gkslow_trunk)
    simulation_data_single, simulation_data_together = simulator.run_simulation(model, direction)

    with open(f'L:/model_optimization/input_output/optuna/single_trial1.pkl',
              'wb') as f:
        pickle.dump(simulation_data_single, f)

    with open(
            f'L:/model_optimization/input_output/optuna/together_trial1.pkl',
            'wb') as f:
        pickle.dump(simulation_data_together, f)


print(f"Running simulation for dendrite {stimulated_dend}")
run_simulation('IN', stimulated_dend, gcar, gkslow, gcar_trunk)
