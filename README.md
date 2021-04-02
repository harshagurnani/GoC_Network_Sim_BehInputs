# GoC_Network_Sim_BehInputs

This repository includes scripts for generating networks as well as NeuroML descriptions used for computational modelling of biophysical-detailed models of electrically-coupled cerebellar Golgi cell circuits. This model was used in the paper __"Multidimensional population activity in an electrically coupled inhibitory circuit in the cerebellar cortex"__ by Gurnani and Silver in Neuron, 2021.

## Requirements
- Python 2.7 was used for generation and execution of scripts due to compatibility with `pyneuroml` libraries at the time. Following recent updates to the library, Python 3.6 will be tested.
- NEURON 7.3 is also needed to compile cellular mechanisms  (.mod files).

However, all network and cellular descriptions are also provided in NeuroML, which can be used to generate simulations in any format.


### Mechanisms
This contains NeuroML or LEMS descriptions of ion channels, synaptic conductances and input spike trains/generators.

### Cells
This contains NeuroML descriptions of single Golgi cells, with different ion channel densities, constrained to match spontaneous firing rates between 2-9Hz, and input/output firing rates to be 14-25Hz/nA.

### Python Utils
This contains general Python scripts for generating networks and connectivity (`network_utils.py`) and input trains.

### Network_XXX
There are separate folders for generating networks with different input structures. To generate the simulation scripts, either execute `generate_all.py` provided in each *Network_XX* folder, or follow the syntax to generate a single network model using `create_GoC_network` in `generate_beh_network_main.py`. This will generate all necessary files:
- channel description as .mod files (need to be compiled)
- GoC descriptions (morphology and cellular mechanism) as .hoc files
- Network descriptors as .nml files
- Simulation description as LEMS files
- Simulation scripts as .py files (which use `neuron` python library for simulation)

To run the relevant simulation, run `LEMS_XXX_nrn.py` files.
