# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:27:13 2023

@author: ab978
"""


import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import pdb, os
import yaml
import math
from utils.models import GATModel
from utils.dataset import CovarianceTemporalDataset
from utils.tools import back_to_matcov
from utils.models import GATModel, RecurrentGCN
from utils.dataset import CovarianceTemporalDataset,CovarianceLaggedDataset,CovarianceSparseDataset
from utils.tools import get_mean_std
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

# TODO
# 1 compare the model forecasts to a baseline, maybe the covariance matrix of the previous day
# 2 visualize the predictions in an appealing way

if __name__ == '__main__':
    

    # Load general hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    plot_losses = p['plot_losses']
    naive_benchmark = p['naive_benchmark']
    # Define the folder path
    folder_path = 'output/{}'.format(p['model_to_load'])
    
    if 'optuna' in folder_path:
        df = pd.read_csv('{}/study.csv'.format(folder_path), index_col=0)
        df.set_index('number',inplace=True)
        idx = df.sort_values('value').index[0]
        folder_path = os.path.join(folder_path,str(idx))
        
    # Load trained model hyperparam file
    with open('{}/GNN_param.yaml'.format(folder_path), 'r') as f:
        p = yaml.safe_load(f)
    p['plot_losses'] = plot_losses
    p['naive_benchmark'] = naive_benchmark
    # fix randomness
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed'])

    # Instantiate the dataset
    if p['fully_connected']:
        dataset = CovarianceLaggedDataset(hdf5_file=p['datafile'],root='_'.join([p['root'],str(p['seq_length'])]), seq_length=p['seq_length'])
    else:
        dataset = CovarianceSparseDataset(hdf5_file=p['datafile'],root='_'.join([p['root'],str(p['seq_length'])]), seq_length=p['seq_length'])
    # train-test split data
    train_size = int(p['split_proportion'] * len(dataset))
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]
    

    pdb.set_trace()
    
    test_loss = 0
    preds = []
    actual = []
    naive_benchmark = []
    with torch.no_grad():
        for data in tqdm(iterable=test_loader, desc='Testing batches...'):
            data = data.to(device)
            if p['scale_up']:
                data.x = data.x * p['scale_up'] 
                data.edge_attr = data.edge_attr * p['scale_up'] 
                data.y_x = data.y_x * p['scale_up']
            # Forward pass
            y_x_hat = model(data)
            preds.append(y_x_hat)
            # Compute loss
            y_x = data.y_x
            actual.append(y_x)
            
            naive_benchmark.append(data.x[:,0])
            
            test_loss += criterion(y_x_hat,y_x)

    # Compute average test loss
    mse = test_loss / len(test_loader)
    rmse = math.sqrt(mse)
        

    preds = torch.concat(preds)
    actual = torch.concat(actual)
    naive_benchmark = torch.concat(naive_benchmark)


    # mse = torch.mean((preds-actual)**2)
    # rmse = math.sqrt(mse)
    # mae = torch.mean((preds-actual))
            
    print('MSE',mse)
    print('RMSE',rmse)
    
       
    fig,ax = plt.subplots(figsize=(10,8))#columnwidth))
    sns.scatterplot(x=actual.numpy(),y=preds.numpy(),ax=ax)
    ax.axline((0, 0), slope=1,color='red', linestyle='--')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_xlim(-20,200)
    ax.set_ylim(-20,200)

    if p['naive_benchmark']:
        fig,ax = plt.subplots(figsize=(10,8))#columnwidth))
        sns.scatterplot(x=actual.numpy(),y=naive_benchmark.numpy(),ax=ax)
        ax.axline((0, 0), slope=1,color='red', linestyle='--')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Naive Benchmark Values')
        ax.set_xlim(-20,200)
        ax.set_ylim(-20,200)
            
        mse = torch.mean((naive_benchmark-actual)**2)
        rmse = math.sqrt(mse)
        mae = torch.mean((naive_benchmark-actual))
                
        print('MSE Naive Benchmark',mse)
        print('RMSE Naive Benchmark',rmse)
        
    if p['plot_losses']:
        train_loss = np.load('{}/train_losses_seed_{}.npy'.format(folder_path, p['seed']))
        test_loss = np.load('{}/test_losses_seed_{}.npy'.format(folder_path, p['seed']))
        fig2,ax2 = plt.subplots(figsize=(10,8))#columnwidth))
        ax2.plot(np.arange(len(train_loss)),np.log(train_loss), label='train')
        ax2.plot(np.arange(len(train_loss)),np.log(test_loss), label='test')
        ax2.legend()
        ax2.set_ylabel('Loss Values')
        ax2.set_xlabel('Epochs')

                
