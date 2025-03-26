# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:32:48 2024

@author: ab978
"""

import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

file = 'volvols' #it can be vols or volvols
# Define the path to your HDF5 file
input_file_path = 'processed_data/{}_mats_taq.h5'.format(file)
output_file_path = 'processed_data/{}_mats_taq_standardized.h5'.format(file)  # Define the output file path
mean_std_csv_path = 'processed_data/{}_mean_std_scalers.csv'.format(file)  # Define the Excel file path for mean and std of scalers

# Define the number of matrices you want to include in the calculations
num_matrices_to_include = 7521  # Replace with the desired number

# Lists to store matrices and matrix names
matrices = []
matrix_names = []

# Open the input HDF5 file
with h5py.File(input_file_path, 'r') as input_file:
    # Iterate through the datasets (assuming they are stored as datasets in the HDF5 file)
    for dataset_name in input_file.keys():
        # Load the dataset into a NumPy array
        matrix = input_file[dataset_name][:]
        
        # Calculate the diagonal and off-diagonal elements
        diagonal_elements = np.diag(matrix)
        off_diagonal_elements = matrix[~np.eye(matrix.shape[0], dtype=bool)].reshape(-1, 1)
        
        # Store matrix and matrix name
        matrices.append((dataset_name, diagonal_elements, off_diagonal_elements))
        matrix_names.append(dataset_name)

# Combine all diagonal and off-diagonal elements for scaling
all_diagonal_elements = np.vstack([diagonal for i, diagonal, _ in matrices if int(i) <= num_matrices_to_include])
all_off_diagonal_elements = np.vstack([off_diag for i, _, off_diag in matrices if int(i) <= num_matrices_to_include])

# Create StandardScaler objects for variances (diagonal) and covariances (off-diagonal)
variance_scaler = StandardScaler()
covariance_scaler = StandardScaler()

# Fit the scalers to all elements within the specified range
variance_scaler.fit(all_diagonal_elements.reshape(-1,1))
covariance_scaler.fit(all_off_diagonal_elements)


# Create a new HDF5 file for writing
with h5py.File(output_file_path, 'w') as output_file:
    # Iterate through the matrices and standardize them
    for matrix_name, diagonal, off_diagonal in matrices:
        # Standardize diagonal elements
        standardized_diagonal = variance_scaler.transform(diagonal.reshape(-1, 1))
        
        # Standardize off-diagonal elements
        standardized_off_diagonal = covariance_scaler.transform(off_diagonal)
        
        # Create a mask to identify off-diagonal elements
        mask = ~np.eye(matrix.shape[0], dtype=bool)
        
        # Initialize the standardized matrix with zeros
        standardized_matrix = np.zeros_like(matrix)
        
        # Fill in the diagonal elements
        np.fill_diagonal(standardized_matrix, standardized_diagonal)
        
        # Fill in the off-diagonal elements using the mask
        standardized_matrix[mask] = standardized_off_diagonal.squeeze()
        
        # Create a dataset in the output HDF5 file with the standardized matrix
        output_file.create_dataset(matrix_name, data=standardized_matrix)

        
# Create a DataFrame for mean and scale of variances and covariances
mean_std_df = pd.DataFrame({
    'Variance': ['Mean', 'Std'],
    'Covariance': ['Mean', 'Std'],
    'Mean': [variance_scaler.mean_[0], covariance_scaler.mean_[0]],
    'Std': [variance_scaler.scale_[0], covariance_scaler.scale_[0]]
})

# Save the DataFrame to a CSV file
mean_std_df.to_csv(mean_std_csv_path, index=False)

# When splitting data into train/validation/test
# Make sure to account for the 1-hour horizon

# If there's code like:
train_data = data[:train_cutoff]
val_data = data[train_cutoff:val_cutoff]

# Change to:
forecast_horizon = 12  # 1 hour in 5-min intervals
train_data = data[:train_cutoff-forecast_horizon]  # Adjust for horizon
val_data = data[train_cutoff-forecast_horizon:val_cutoff-forecast_horizon]