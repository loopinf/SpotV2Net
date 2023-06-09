# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:54:34 2023

@author: ab978
"""
import pandas as pd
import numpy as np
import os, pdb
from tqdm import tqdm
from glob import glob
import h5py
from collections import OrderedDict


# get historical cov series saved as files with TICKER1_TICKER2 filename

symbols = []
hist_covs = {}
for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data','cov_estimation','*.csv')),desc='Getting symbols...'):
    df = pd.read_csv(filename, header = None)
    pairs = filename.split('\\')[-1].split('.csv')[0]
    symbols.extend(pairs.split('_'))
    hist_covs[pairs] = df.values
    
    
# get historical vol series saved as TICKER filename 
df_to_concat = []
for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data','vol_estimation','*_1min_1min_vol.txt')),desc='Joining RVols...'):
    df = pd.read_csv(filename, names=['placeholder'])
    symbol = filename.split('\\')[-1].split('_')[0]
    df.columns = [symbol]
    df_to_concat.append(df)
vol_df = pd.concat(df_to_concat,axis=1)
# load time index
ts = pd.read_csv('processed_data/filtered_timestamps.csv',index_col=0)
vol_df.index = ts.index
vol_df.reset_index(inplace=True)
vol_df = vol_df.T.sort_index().T

# check symbols between covs and vols files
ordered_symbols = sorted(list(set(symbols)))
assert len(ordered_symbols) == (ordered_symbols == vol_df.columns[:-1]).sum()
# get the number of periods (covariance matrices)
n_periods = list(set([len(hist_covs[k]) for k in hist_covs.keys()]))[0]

# initialize dict of empty pandas dataframe
hist_cov_mat = {}
# hist_cov_mat = OrderedDict()
for i in tqdm(iterable=range(n_periods), desc='Creating empty cov mats'):
    hist_cov_mat[str(i)] = pd.DataFrame(index=ordered_symbols, columns = ordered_symbols)


# insert diagonals (variances) into the dataframes
for i in tqdm(iterable=vol_df.index, desc='Filling Mat with Vols'):
    np.fill_diagonal(hist_cov_mat[str(i)].values,vol_df.iloc[i].values[:-1])
    

# insert all the other values
for k in tqdm(iterable=hist_covs.keys(), desc='Filling Mat with Covs...'):
    s1,s2 = k.split('_')
    values = hist_covs[str(k)].reshape(-1)
    for i,v in enumerate(values):
        hist_cov_mat[i].loc[s1,s2] = v
        hist_cov_mat[i].loc[s2,s1] = v


# save df as numpy values
hist_cov_mat_numpy = {k:v.values.astype(np.float64) for k,v in hist_cov_mat.items()}
# Save the covariance matrices in an HDF5 file
with h5py.File("processed_data/covs_mats_30min_0602.h5", "w") as f:
    for key, value in hist_cov_mat_numpy.items():
        # Create a dataset with the same name as the key and store the value
        f.create_dataset(str(key), data=value, dtype=np.float64)
        
        
# # Load back the data
# # Open the HDF5 file for reading
# with h5py.File("processed_data/covs_mats_30min2.h5", 'r') as f:
#     # Create an empty dictionary to store the loaded data
#     data_dict_loaded = {}
    
#     # Loop through each dataset in the file and add it to the dictionary
#     for key in f.keys():
#         data_dict_loaded[int(key)] = np.array(f[key])
        
        
# # Assume data_dict is the dictionary containing the arrays
# is_symmetric = {}

# # Loop through each array in the dictionary
# for key, arr in hist_cov_mat_numpy.items():
#     # Check if the array is diagonal by comparing it with its diagonal elements
#     if not np.allclose(arr, arr.T):
#         print(key, 'Not symm')