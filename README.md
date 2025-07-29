# Dendritic Integration Simulation and Visualization

This project simulates synaptic input integration of oblique dendrites in a CA1 pyramidal neuron model. 
It evaluates and visualizes the relationship between the number of synaptic inputs and the resulting membrane potential changes, specifically focusing on dendritic and somatic EPSPs.

---

## Project Overview

The simulation compares:
- **Single-synapse stimulation (independent trials)**
- **Multi-synapse simultaneous stimulation**

The main goal is to study non-linear synaptic summation, particularly how multiple synaptic inputs interact at the soma and selected dendrites.

---

## Repository Structure

- `simulator/ModelSimulator.py`  
  Builds and runs the neuron model.
  
- `main_single_dend.py`  
  Runs a simulation with preset parameters and saves results.

- `input_output_selected_nsyns.ipynb`  
  Jupyter notebook for loading results and visualizing traces and input-output curves.

---

## Simulation Parameters

You can configure:
- **Stimulated dendrites** (e.g. `stimulated_dend = [108]`)
- **Direction of input propagation** (`'IN'` or `'OUT'`)
- **Channel conductances**:
  - `gcar`, `gkslow`, `gcar_trunk`, `gkslow_trunk`

Example parameter set:

```python
direction = 'IN'
stimulated_dend = [108]
gcar = 0.00375
gkslow = 0.00519
gcar_trunk = 0.01486
gkslow_trunk = 0.00039
```

---

## How to Run the Pipeline
### 1. **Install Requirements**

Use Python 3.9+ and install dependencies with:

```bash
pip install -r requirements.txt
```

---

### 2. **Compile Mechanism Files**

Before running simulations, you need to compile NEURON's `.mod` files.
Navigate to the folder containing the `.mod` files:

```bash
cd simulator/model/density_mechs
```
And run the `nrnivmodl` tool from NEURON.

If NEURON has trouble finding compiled mechanisms, consider copying them into the root folder and re-running `nrnivmodl`.

More information about NEURON: https://neuron.yale.edu/neuron<br>
More information about working with .mod files: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3263<br>
More information about compiling .mod files: https://nrn.readthedocs.io/en/latest/guide/faq.html#how-do-i-compile-mod-files

---

### 3. Run the Simulation

Edit and run `main_single_dend.py`

This creates two `.pkl` output files containing:
- `simulation_data_single`: individual synaptic responses
- `simulation_data_together`: simultaneous synaptic input response

---

## Visualization

Open `input_output_selected_nsyns.ipynb` to:

- **Plot traces** at `soma(0.5)` and target dendrites
- **Compute EPSPs** from baseline and peak membrane potentials
- **Compare expected vs. measured EPSPs** (linear vs. non-linear summation)

---
