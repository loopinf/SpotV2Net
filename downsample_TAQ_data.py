# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:04:28 2024

@author: ab978
"""

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm 
import pandas_market_calendars as mcal

# Code to load the output fo the TAQ database and filter only transactons happening on the NYSE
# See https://www.nyse.com/publicdocs/nyse/data/Daily_TAQ_Client_Spec_v4.0.pdf at page 64 for code of exchanges
#########################################################################################################
# Open the CSV file as a Dask DataFrame. Use Dask (python package) otherwise it will take long time with pandas.
# I suggest to download one year of data at a time and process it separately using this script
# as it can be become very expensive for your CPU to handle this large amount of data, even if using Dask.
filename = 'FILENAME_FROM_TAQ'
years = '20_21' # it can be 20_21, 21_22, 22_23 to cover the three years of data used in the paper
dtypes = {'SYM_SUFFIX': 'object'}
df = dd.read_csv('rawdata/taq/{}.csv'.format(filename), dtype=dtypes)

filtered_df = df[df['EX']=='N']
filtered_df = filtered_df.drop(['SYM_SUFFIX','EX','SIZE'], axis=1)

# filtered_df.to_csv('rawdata/taq/dowtick_20_21.csv', index=False, single_file=True)
filtered_df.to_csv('rawdata/taq/dowtick_{}.csv'.format(years), index=False, single_file=True)
#######################################################################################################
# load back the saved dataset and filter inbetween market hours
df = dd.read_csv('rawdata/taq/dowtick_{}.csv'.format(years))
# Convert 'TIME_M' column to timestamp
df['datetime'] = dd.to_datetime(df['DATE'].astype(str) + ' ' + df['TIME_M'].astype(str), format='%Y-%m-%d %H:%M:%S.%f')
# Filter out rows outside the desired time range
start_time = pd.to_datetime('09:30:00').time()
end_time = pd.to_datetime('16:00:00').time()

# Convert datetime to time component as string
df['time_str'] = df['datetime'].dt.strftime('%H:%M:%S')

# Filter rows within the desired time range
df_filtered = df[df['time_str'].between(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S'))]

df_opposite = df_filtered[~df_filtered['time_str'].between(start_time.strftime('%H:%M:%S'), end_time.strftime('%H:%M:%S'))]

# double check
assert df_opposite.head().shape[0] == 0

# Drop the 'time_str' column
df_filtered = df_filtered.drop(columns=['DATE','TIME_M','time_str'])
df_filtered.to_csv('rawdata/taq/dowtick_final_{}.csv'.format(years), index=False, single_file=True)
#######################################################################################################
# Load the dataset to resample
# The strategy has to be:

# take a symbol and resample to 1 sec
# get rid of the datetime outside market hours (created after resampling with pandas)
# get rid of business dates and holidays
# fill the holes with the closest datetime
# concat all the symbols for one year
# go to the next year
# https://www.nasdaq.com/market-activity/2022-stock-market-holiday-calendar

symbols = ['BA','AAPL', 'AMGN', 'AXP', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']

df_filtered = dd.read_csv('rawdata/taq/dowtick_final_{}.csv'.format(years), blocksize='200MB')
# Convert 'datetime' column to datetime type
df_filtered['datetime'] = dd.to_datetime(df_filtered['datetime'])

chunks = []
for sym_root_value in tqdm(iterable=symbols,desc='Resampling stocks...'):
    print(sym_root_value)
    chunk = df_filtered.loc[df_filtered['SYM_ROOT'] == sym_root_value].compute()
    # Fix problems cause by overlapping of different time series before resampling
    # if sym_root_value == 'GS' or sym_root_value == 'JPM':
    #     chunk = chunk[chunk['PRICE'] >= 60]   
    # resample
    res_chunk = chunk.resample('1S', on='datetime').first()
    # get rid of unnecessary columns and round timestamps
    res_chunk.index = res_chunk.index.round('1s')
    res_chunk = res_chunk.drop(columns=['SYM_ROOT'],axis=1)
    # filter relevant timestamps
    res_chunk = res_chunk.between_time('09:30:00', '16:00:00')
    
    # Create a calendar instance for the US market
    us_market = mcal.get_calendar('NYSE')
    # apply the filter on the index
    mask = us_market.valid_days(start_date=res_chunk.index.min().date(), end_date=res_chunk.index.max().date())
    # Convert the mask dates to a Pandas DatetimeIndex
    mask_dates = mask.date
    # Convert the index to a separate column
    res_chunk['date'] = res_chunk.index.date
    # Filter the original DataFrame based on the valid dates
    res_chunk_activedays = res_chunk[res_chunk['date'].isin(mask_dates)]
    # Drop the additional 'date' column
    res_chunk_activedays = res_chunk_activedays.drop('date', axis=1)
    # timestamps to add
    top = pd.Timestamp('09:30:00')
    bottom = pd.Timestamp('16:00:00')
    first_time, last_time = res_chunk_activedays.index[0].time(),res_chunk_activedays.index[-1].time()
    first_date, last_date = res_chunk_activedays.index[0].date(),res_chunk_activedays.index[-1].date()
    # Check if the time of the first timestamp in the Series matches the desired time
    if first_time != top:
        # Create a new timestamp with the desired date and time
        top_timestamp = pd.Timestamp(first_date) + pd.DateOffset(hours=top.hour, minutes=top.minute, seconds=top.second)
        # Add a row with the desired timestamp and NaN value at the top of the Series
        res_chunk_activedays.loc[top_timestamp] = pd.NA

    if last_time != bottom:
        # Create a new timestamp with the desired date and time
        bottom_timestamp = pd.Timestamp(last_date) + pd.DateOffset(hours=bottom.hour, minutes=bottom.minute, seconds=bottom.second)
        # Add a row with the desired timestamp and NaN value at the top of the Series
        res_chunk_activedays.loc[bottom_timestamp] = pd.NA

    # Sort the Series by index
    res_chunk_activedays = res_chunk_activedays.sort_index()
    
    res_chunk_activedays['ticker'] = sym_root_value
    
    print(res_chunk_activedays.shape[0])
    if sym_root_value == 'GS': 
        idx_to_reindex = res_chunk_activedays.index
    else:
        res_chunk_activedays = res_chunk_activedays.reindex(idx_to_reindex)
    
    assert res_chunk_activedays.shape[0] == 1474263, '{} has a different number of observation'.format(sym_root_value)
    
    chunks.append(res_chunk_activedays)

full_df = pd.concat(chunks)

full_df.to_csv('rawdata/taq/dowtick_final_{}_resampled.csv'.format(years))


# Useful check if needed
# given_date = pd.to_datetime('2023-07-04 15:30:02')

# # Using the 'in' operator
# if given_date in chunks[0].index:
#     print("Given date exists in the datetime index")
# else:
#     print("Given date does NOT exist in the datetime index")