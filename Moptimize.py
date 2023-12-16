# Install FrEIA amd torch via pip install
import pandas as pd
import numpy as np
import scipy as sc
import dill
from scipy.optimize import minimize as nelder_mead
import torch
import torch.nn as nn
from train_surrogate import model as surrogate
from surrogate_eval import eval, lossModal
from time import time
from StandardScaler import StandardScaler as stdScaler
from CreateData import runAnsys, extract_output, change_input, modify_input


############################################
# Model function
############################################
def model(x, dataf, cols):
    # Define the path to your APDL script
    apdl_script = '"folder/BeamEI.db"'
    # Define the output file for results
    output_file = '"folder/ANSYS/file.out"'
    # Define the results file of the ansys calculation
    results_file = 'folder/ANSYS/Output.txt'

    # Define modify file
    mod_Beam_file = 'folder/ANSYS/test2.txt'

    # Create a random matrix for modification
    mod_inp = x.reshape(-1, len(cols))

    # Modify input and create new beam data file
    change_input(dataf, cols, mod_inp, mod_Beam_file)

    # Run Ansys to calculate the Beam results
    runAnsys(apdl_script, output_file)

    # Extract the beam characteristics
    freq, uy = extract_output(results_file)

    # Combine output
    return np.array(freq + [uy])


############################################
# Fitness function
############################################

def MSE(inp, dataf, cols, y_targets):
    y_calc = model(inp, dataf, cols)
    return np.mean((y_targets - y_calc) ** 2)

#################################################
# Read Beam information
#################################################

# Original input file
input_file_path = 'folder/ANSYS/orig_beam.txt'
# Read the file into a Pandas DataFrame
df = pd.read_csv(input_file_path, delimiter=',').rename(columns=lambda x: x.strip())
df_target = df.copy()

############################################
# Initialize Parameters
############################################

variation = [0.9, 1.1]
col = ['m', 'A', 'Ixx', 'J']

#################################################
# Create a target
#################################################

# Create a random matrix for modification
mod_matrix = np.random.uniform(variation[0], variation[1], (df_target.shape[0], len(col)))
mod_input = df_target[col].multiply(mod_matrix, axis=0).to_numpy().flatten()
# calculate the target values for the output
y_target = model(mod_input, df_target, col)

############################################
# Run optimization
############################################

# initialize the random generator
rng = np.random.default_rng()

init_pos = df[col].to_numpy().flatten()


bounds_list = [(init_pos[i] *variation[0], init_pos[i] *variation[1]) for i in range(init_pos.shape[0])]

results = nelder_mead(MSE, x0=init_pos, bounds=bounds_list, method='Nelder-Mead', args=(df, col, y_target),
                              options={'maxfev': 20, 'disp': True, 'fatol': 1e-5})

print("The MSE is:", results.fun)
print("The results are:", results.x.reshape(-1, len(col)))
print("The target is:", mod_input.reshape(-1, len(col)))
print("The MSE is:", np.mean((results.x - mod_input) ** 2))