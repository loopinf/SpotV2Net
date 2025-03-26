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

def generate_matrices(source='price'):
    
    ################################### PRICES################################################
    if source == 'price':
        filenames = ['covol', 'vol']
        # filenames = ['covol202306_09', 'vol202306_09']
    elif source == 'vol':
        filenames = ['covol_of_vol', 'vol_of_vol']
        
    # get historical cov series saved as files with TICKER1_TICKER2 filename
    symbols = []
    hist_covs = {}
    for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data',filenames[0],'*.csv')),desc='Getting symbols...'):
        df = pd.read_csv(filename, header = None)
        # df = df.stack().reset_index(drop=True) FIXED MISTAKE
        df = pd.concat([df[col] for col in df], ignore_index=True)
        pairs = filename.split('\\')[-1].split('.csv')[0]
        symbols.extend(pairs.split('_'))
        hist_covs[pairs] = df.values
        

    # get historical vol series saved as TICKER filename  
    df_to_concat = []
    for filename in tqdm(glob(os.path.join(os.getcwd(),'processed_data',filenames[1],'*.csv')),desc='Joining RVols...'):
        df = pd.read_csv(filename, header=None)
        # df = df.stack().reset_index(drop=True) FIXED MISTAKE
        df = pd.concat([df[col] for col in df], ignore_index=True)
        symbol = filename.split('\\')[-1].split('.')[0]
        df.name = symbol
        df_to_concat.append(df)
        
    vol_df = pd.concat(df_to_concat,axis=1)
    sorted_columns = sorted(vol_df.columns)
    vol_df = vol_df[sorted_columns]
    

    # check symbols between covs and vols files
    ordered_symbols = sorted(list(set(symbols)))
    
    assert len(ordered_symbols) == (ordered_symbols == vol_df.columns).sum()
    # get the number of periods (covariance matrices)
    n_periods = list(set([len(hist_covs[k]) for k in hist_covs.keys()]))[0]
    print('Vol has {} obs'.format(n_periods))
    
    # initialize dict of empty pandas dataframe
    hist_cov_mat = {}
    # hist_cov_mat = OrderedDict()
    for i in tqdm(iterable=range(n_periods), desc='Creating empty cov mats'):
        hist_cov_mat[str(i)] = pd.DataFrame(index=ordered_symbols, columns = ordered_symbols)
    

    # insert diagonals (variances) into the dataframes
    for i in tqdm(iterable=vol_df.index, desc='Filling Mat with Vols'):
        np.fill_diagonal(hist_cov_mat[str(i)].values,vol_df.iloc[i].values)



    # insert all the other values
    for k in tqdm(iterable=hist_covs.keys(), desc='Filling Mat with Covs...'):
        s1,s2 = k.split('_')
        values = hist_covs[str(k)].reshape(-1)
        for i,v in enumerate(values):
            hist_cov_mat[str(i)].loc[s1,s2] = v
            hist_cov_mat[str(i)].loc[s2,s1] = v
   
    # save df as numpy values
    hist_cov_mat_numpy = {k:v.values.astype(np.float64) for k,v in hist_cov_mat.items()}
    
    return hist_cov_mat_numpy
        
def are_keys_ordered_as_numbers(dictionary):
    keys = list(dictionary.keys())
    for i in range(len(keys) - 1):
        if int(keys[i]) > int(keys[i + 1]):
            return False
    return True

def calc_vol(prices):
    # Replace daily volatility calculation with hourly
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    
    # For 1-hour window (12 5-min bars)
    hourly_window = 12
    
    # Calculate realized volatility (hourly)
    vol = log_returns.rolling(window=hourly_window).std() * np.sqrt(hourly_window)
    
    # Create 1-hour ahead targets (shift by 12 bars)
    targets = vol.shift(-12)
    
    return vol, targets

# Add this function after the existing calc_vol function
def create_hourly_indices(start_date, end_date):
    """Create proper time indices for hourly forecasting
    
    Args:
        start_date: Starting date in 'YYYY-MM-DD' format
        end_date: Ending date in 'YYYY-MM-DD' format
        
    Returns:
        DatetimeIndex with hourly timestamps during market hours
    """
    # Assuming market hours 9:30-16:00
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    
    hourly_indices = []
    for day in business_days:
        # For each hour in the trading day
        for hour in range(9, 16):
            minute = 30 if hour == 9 else 0
            
            if hour < 16 or (hour == 16 and minute == 0):  # Trading hours only
                hourly_indices.append(pd.Timestamp(
                    year=day.year, month=day.month, day=day.day,
                    hour=hour, minute=minute
                ))
    
    return pd.DatetimeIndex(hourly_indices)

# Add this function to process your 5-minute data
def process_5min_data(df_5min, start_date, end_date):
    """Process 5-minute dataframe to create volatility and vol-of-vol data
    
    Args:
        df_5min: DataFrame with 5-minute price data (rows=timestamps, columns=symbols)
        start_date: Start date for analysis
        end_date: End date for analysis
        
    Returns:
        Processed volatility and volatility-of-volatility data
    """
    # Make sure the dataframe has a proper datetime index
    if not isinstance(df_5min.index, pd.DatetimeIndex):
        df_5min.index = pd.to_datetime(df_5min.index)
    
    # Get list of symbols
    symbols = df_5min.columns.tolist()
    
    # Calculate volatility and 1-hour ahead targets using existing function
    vol_df, targets_df = calc_vol(df_5min)
    
    # Create directories if they don't exist
    os.makedirs('processed_data/vol', exist_ok=True)
    os.makedirs('processed_data/covol', exist_ok=True)
    os.makedirs('processed_data/vol_of_vol', exist_ok=True)
    os.makedirs('processed_data/covol_of_vol', exist_ok=True)
    
    # Save individual volatility files
    for symbol in symbols:
        vol_df[symbol].to_csv(f'processed_data/vol/{symbol}.csv', header=False)
    
    # Calculate and save covolatility for each pair
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:  # Upper triangle
                log_returns1 = np.log(df_5min[symbol1] / df_5min[symbol1].shift(1)).fillna(0)
                log_returns2 = np.log(df_5min[symbol2] / df_5min[symbol2].shift(1)).fillna(0)
                
                # 12 bars = 1 hour
                covol = log_returns1.rolling(window=12).cov(log_returns2) * np.sqrt(12)
                covol.to_csv(f'processed_data/covol/{symbol1}_{symbol2}.csv', header=False)
    
    # Calculate vol-of-vol
    vol_of_vol_df = vol_df.pct_change().rolling(window=12).std() * np.sqrt(12)
    
    # Save individual vol-of-vol files
    for symbol in symbols:
        vol_of_vol_df[symbol].to_csv(f'processed_data/vol_of_vol/{symbol}.csv', header=False)
    
    # Calculate and save covol-of-vol for each pair
    for i, symbol1 in enumerate(symbols):
        for j, symbol2 in enumerate(symbols):
            if i < j:  # Upper triangle
                vol_returns1 = vol_df[symbol1].pct_change().fillna(0)
                vol_returns2 = vol_df[symbol2].pct_change().fillna(0)
                
                covol_of_vol = vol_returns1.rolling(window=12).cov(vol_returns2) * np.sqrt(12)
                covol_of_vol.to_csv(f'processed_data/covol_of_vol/{symbol1}_{symbol2}.csv', header=False)
    
    return vol_df, vol_of_vol_df

# Then modify the main execution block to use your 5-minute data
if __name__=='__main__':
    # Uncomment these lines when ready to process your data
    # Load your 5-minute dataframe
    # df_5min = pd.read_csv('your_5min_data.csv', index_col=0, parse_dates=True)
    # process_5min_data(df_5min, start_date='2020-01-01', end_date='2022-12-31')
    
    # Then continue with the existing code
    vol = generate_matrices(source='price')
    assert all(int(list(vol.keys())[i]) <= int(list(vol.keys())[i + 1]) for i in range(len(list(vol.keys())) - 1))

    volvol = generate_matrices(source='vol')
    assert all(int(list(volvol.keys())[i]) <= int(list(volvol.keys())[i + 1]) for i in range(len(list(volvol.keys())) - 1))
    
    # align observation
    vol = {k: v for k, v in sorted(vol.items(), key=lambda x: int(x[0]))[:len(volvol)]}

    # Save the covariance matrices in an HDF5 file
    with h5py.File("processed_data/vols_mats_taq.h5", "w") as f:
        for key, value in vol.items():
            f.create_dataset(str(key), data=value, dtype=np.float64)
            
    # Save the covariance matrices in an HDF5 file
    with h5py.File("processed_data/volvols_mats_taq.h5", "w") as f:
        for key, value in volvol.items():
            f.create_dataset(str(key), data=value, dtype=np.float64)
            
    # Save the targets for 1-hour ahead forecasting
    # Uncomment when processing your data
    # with h5py.File("processed_data/targets_1hour.h5", "w") as f:
    #     # Create dataset for targets
    #     pass