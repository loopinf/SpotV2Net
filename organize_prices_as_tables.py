# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:13:38 2024

@author: ab978
"""

import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm 

tickers = ['AAPL', 'AMGN', 'AXP', 'BA', 'CAT', 'CRM', 'CSCO', 'CVX', 'DIS', 'DOW', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'V', 'VZ', 'WBA', 'WMT']
dtodrop = ['2020-11-27','2021-11-26','2022-11-25'] # Thanksgivings


df2021 = dd.read_csv('rawdata/taq/dowtick_final_20_21_resampled.csv')
df2122 = dd.read_csv('rawdata/taq/dowtick_final_21_22_resampled.csv')
df2223 = dd.read_csv('rawdata/taq/dowtick_final_22_23_resampled.csv')

for t in tqdm(iterable=tickers, desc='Creating individual files...'):
    print(t)
    chunk2021 = df2021.loc[df2021['ticker'] == t].compute()
    chunk2021.set_index('datetime',inplace=True)
    chunk2021.index = pd.to_datetime(chunk2021.index)
    
    chunk2122 = df2122.loc[df2122['ticker'] == t].compute()
    chunk2122.set_index('datetime',inplace=True)
    chunk2122.index = pd.to_datetime(chunk2122.index)
    
    chunk2223 = df2223.loc[df2223['ticker'] == t].compute()
    chunk2223.set_index('datetime',inplace=True)
    chunk2223.index = pd.to_datetime(chunk2223.index)
    
    pivot2021 = pd.pivot_table(chunk2021, values='PRICE', index=chunk2021.index.time, columns=chunk2021.index.date, dropna=False)
    assert pivot2021.shape == (23401,252), 'Wrong shape in pivot table 2021'
    pivot2122 = pd.pivot_table(chunk2122, values='PRICE', index=chunk2122.index.time, columns=chunk2122.index.date, dropna=False)
    assert pivot2122.shape == (23401,253), 'Wrong shape in pivot table 2122'
    pivot2223 = pd.pivot_table(chunk2223, values='PRICE', index=chunk2223.index.time, columns=chunk2223.index.date, dropna=False)
    assert pivot2223.shape == (23401,252), 'Wrong shape in pivot table 2223'
    
    full_pivot = pd.concat(objs=[pivot2021,pivot2122,pivot2223],axis=1)
    
    assert full_pivot.shape == (23401,252+253+252), 'Wrong shape after lateral concat'
    
    full_pivot = full_pivot.fillna(method='ffill').fillna(method='bfill')
    
    # drop some columns for Black Friday
    full_pivot.drop([pd.to_datetime(d).date() for d in dtodrop], axis=1,inplace=True)
    
    full_pivot.to_csv('rawdata/taq/bycomp/{}_20_23.csv'.format(t))