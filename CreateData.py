import subprocess
import pandas as pd
import numpy as np
import os


def runAnsys(input,output):

    # Define the path to the ANSYS executable
    ansys_executable = '"C:/Program Files/ANSYS Inc/v192/ansys/bin/winx64/MAPDL.exe"'

    # Define the working folder
    working_folder = '"folder/ANSYS"'

    # Define job name
    jobname = '"Test"'

    # Delete the results file
    file_path = os.path.join(working_folder[1:-1], 'Output.txt')

    if os.path.isfile(file_path):
        os.remove(file_path)

    # Create the command to run ANSYS
    command = [ansys_executable, '-p ansys', '-np 2', '-dir', working_folder, '-j', jobname, '-s read', '-b nolist', '-i', input, '-o', output]
    command = " ".join(command)
    # Run the command
    subprocess.run(command)

def extract_output(output):

    with open(output, 'r') as file:
        lines = file.readlines()

    # Extracting frequencies
    frequencies = []
    displacement_uy = {}

    # Flags to indicate the current section of the file
    frequency_section = True
    uy_section = False
    count = 0

    for line in lines:
        # Check for the end of the frequency section and the start of the UY section
        parts = line.split()
        if len(parts) >= 2 and parts[0] == 'SET' and parts[1] == 'TIME/FREQ':
            frequency_section = True
            count = 10
            continue
        if len(parts) >= 2 and parts[0] == 'NODE' and parts[1] == 'UY':
            uy_section = True
            count = 1
            continue

        if frequency_section and count > 0:
            count -= 1
            try:
                frequency = float(parts[1])
                frequencies.append(frequency)
            except ValueError:
                continue


        if uy_section and count > 0:
            count -= 1
            try:
                uy_value = float(parts[1])
            except ValueError:
                continue


        if count < 1:
            frequency_section = False
            uy_section = False
            continue

    return frequencies, uy_value

def modify_input(df, columns, variation, output_file_path):

    # Create a random matrix for modification
    mod_matrix = np.random.uniform(variation[0], variation[1], (df.shape[0], len(columns)))
    # Multiply only the numeric columns by the random matrix
    df[columns] = df[columns].multiply(mod_matrix, axis=0)
    # Write the modified DataFrame to a file with specified float format
    df.to_csv(output_file_path, sep=',', index=False, float_format='%19.12f')

    # Converting the DataFrame to a NumPy vector
    matrix_vector = df[columns].to_numpy().transpose().flatten()

    return matrix_vector


def change_input(df, columns, new_matrix, output_file_path):

    # Multiply only the numeric columns by the random matrix
    df[columns] = new_matrix
    # Write the modified DataFrame to a file with specified float format
    df.to_csv(output_file_path, sep=',', index=False, float_format='%19.12f')


if __name__ == "__main__":
    #################################################
    # Read Beam information
    #################################################

    # Original input file
    input_file_path = 'C:/Users/Noever/Desktop/NORVENTO TEST/ANSYS/orig_beam.txt'
    # Read the file into a Pandas DataFrame
    df = pd.read_csv(input_file_path, delimiter=',').rename(columns=lambda x: x.strip())

    #################################################
    # Create Dataset
    #################################################

    n_samp = 3

    # Define the path to your APDL script
    apdl_script = '"folder/BeamEI.db"'
    # Define the output file for results
    output_file = '"folder/ANSYS/file.out"'
    # Define the results file of the ansys calculation
    results_file = 'folder/ANSYS/Output.txt'

    # Define modify file
    mod_Beam_file = 'folder/ANSYS/test2.txt'
    # Define columns to vary
    col = ['m', 'A', 'Ixx', 'J']
    # Define random range for variation
    variation = [0.95, 1.05]

    #initialize input and output
    input = [None] * n_samp
    output = [None] * n_samp

    for i in range(n_samp):
        # Modify input and create new beam data file
        input[i] = modify_input(df, col, variation, mod_Beam_file)

        # Run Ansys to calculate the Beam results
        runAnsys(apdl_script, output_file)

        # Extract the beam characteristics
        freq, uy = extract_output(results_file)

        # Combine output
        output[i] = np.array(freq+[uy])

    inp_data = np.array(input)
    # Save to CSV file
    np.savetxt('input_data.csv', input, delimiter=';')

    out_data = np.array(output)
    # Save to CSV file
    np.savetxt('output_data.csv', output, delimiter=';')



