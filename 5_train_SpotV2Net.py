# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:37:17 2023

@author: ab978
"""

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import pdb, os
import yaml
import math
import sys
import torch.multiprocessing as mp
from utils.models import GATModel, RecurrentGCN
from utils.dataset import (CovarianceLaggedDataset,
                            CovarianceSparseDataset,
                            CovarianceLaggedMultiOutputDataset)
from typing import Optional, Dict, Union

def train(seed: Optional[int] = None,
          trial: Optional[object] = None,
          p: Optional[Dict[str, Union[str, int, float, bool]]] = None) -> None:
    """
    Trains a model based on the given parameters.

    :param seed: Random seed for reproducibility, defaults to None
    :param trial: Optuna trial object for hyperparameter optimization, defaults to None
    :param p: Dictionary containing hyperparameters and other configuration details, defaults to None
    """
    
    if trial and p:
        # Define the folder path
        folder_path = 'output/{}_{}/{}'.format(p['modelname'], 'optuna', trial.number)
    else:
        # Load hyperparam file
        with open('config/GNN_param.yaml', 'r') as f:
            p = yaml.safe_load(f)
        p['seed'] = seed
        # Define the folder path
        folder_path = 'output/{}_{}'.format(p['modelname'],p['seq_length'])
        
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Save the yaml file to the model folder
    with open('{}/GNN_param.yaml'.format(folder_path), 'w') as f:
        yaml.dump(p, f)

    # fix randomness
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])
    torch.cuda.manual_seed_all(p['seed'])

    # Load config
    with open('config/GNN_param.yaml') as f:
        config = yaml.safe_load(f)

    # Use the horizon and lookback parameters
    forecast_horizon = config.get('forecast_horizon', 12)  # Default to 1-hour (12 intervals)
    lookback_window = config.get('lookback_window', 24)  # Default to 2-hours (24 intervals)

    # Instantiate the dataset
    if p['fully_connected']:
        if p['output_node_channels'] == 1:
            dataset = CovarianceLaggedDataset(hdf5_file1=os.path.join(os.getcwd(),p['volfile']), 
                                              hdf5_file2=os.path.join(os.getcwd(),p['volvolfile']),
                                              root='_'.join([p['root'],str(p['seq_length'])]), 
                                              seq_length=p['seq_length'])
        else:
            dataset = CovarianceLaggedMultiOutputDataset(hdf5_file1=os.path.join(os.getcwd(),p['volfile']),
                                                         hdf5_file2=os.path.join(os.getcwd(),p['volvolfile']),
                                                         root='_'.join([p['root'],str(p['seq_length']),'moutput']), 
                                                         seq_length=p['seq_length'], future_steps=p['output_node_channels'])
    else:
        if p['threshold']:
            root = '_'.join([p['root'],'sparse','t_{}'.format(p['threshold']),str(p['seq_length'])])
        else:
            root = '_'.join([p['root'],'sparse',str(p['seq_length'])])
        dataset = CovarianceSparseDataset(hdf5_file=p['datafile'],root=root, seq_length=p['seq_length'], threshold=p['threshold'])
        p['num_edge_features'] = 1
    pdb.set_trace()
    # train-test split data
    train_size = int(p['split_proportion'] * len(dataset))
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]
    
    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
    
    
    # select dimensions from data
    NODE_FEATURES = dataset[0].x.shape[1]
    EDGE_FEATURES = dataset[0].edge_attr.shape[1]

    # Instantiate the model
    if p['modeltype'] == 'gat':
        model = GATModel(num_node_features=NODE_FEATURES, 
                         num_edge_features = EDGE_FEATURES,
                         num_heads=p['num_heads'], 
                         output_node_channels=p['output_node_channels'], 
                         dim_hidden_layers=p['dim_hidden_layers'],
                         dropout_att = p['dropout_att'],
                         dropout = p['dropout'],
                         activation = p['activation'],
                         concat_heads= p['concat_heads'],
                         negative_slope=p['negative_slope'],
                         standardize = p['standardize'])
    elif p['modeltype'] == 'rnn':
        model = RecurrentGCN(num_features=p['seq_length'], 
                         hidden_channels=p['hidden_channels'], 
                         output_node_channels=p['output_node_channels'], 
                         dropout = p['dropout'],
                         activation = p['activation'])
    
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    
  
    # Set loss function and optimizer
    criterion = torch.nn.MSELoss()
    if p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'])
    elif p['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=p['learning_rate'])
    else:
        print('Choose an available optimizer')
        sys.exit()
    
    # Train the model
    train_losses, test_losses = [], []
    prev_test_loss = float('inf')
    for epoch in range(p['num_epochs']):
        model.train()
        total_loss = 0
        for data in tqdm(iterable=train_loader, desc='Training batches...'):
            data = data.to(device)
            if p['scale_up']:
                data.x = data.x * p['scale_up'] 
                data.edge_attr = data.edge_attr * p['scale_up'] 
                data.y_x = data.y_x * p['scale_up']

            # Forward pass
            y_x_hat = model(data)
            # Compute loss
            y_x = data.y_x

            loss = criterion(y_x_hat, y_x)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Optimize
            optimizer.step()
            total_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_rmse = math.sqrt(avg_train_loss)

        # Evaluate on the test set
        model.eval()
        test_loss = 0
    
        with torch.no_grad():
            for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                data = data.to(device)
                if p['scale_up']:
                    data.x = data.x * p['scale_up'] 
                    data.edge_attr = data.edge_attr * p['scale_up'] 
                    data.y_x = data.y_x * p['scale_up']
                # Forward pass
                y_x_hat = model(data)
                # Compute loss
                y_x = data.y_x
                loss = criterion(y_x_hat, y_x)
                test_loss += loss.item()
        # Compute average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        avg_test_rmse = math.sqrt(avg_test_loss)
        
        # Check if the average training loss improved by the specified tolerance
        if epoch == 0 or avg_test_loss + float(p['tolerance']) < prev_test_loss:
            # Update the previous training loss
            prev_test_loss = avg_test_loss
            # Save the model weights
            save_path = '{}/{}_weights_seed_{}.pth'.format(folder_path,p['modelname'],p['seed'])
            torch.save(model.state_dict(), save_path)
        
        print(f"Epoch: {epoch+1}/{p['num_epochs']}, Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}, Train RMSE: {avg_train_rmse:.10f}, Test RMSE: {avg_test_rmse:.10f}")
        # Update evaluation metrics to report 1-hour forecast performance
        print(f"1-Hour Forecast Results: MAE={mae:.4f}, RMSE={rmse:.4f}")
   
    # save losses
    np.save('{}/train_losses_seed_{}.npy'.format(folder_path, p['seed']), np.array(train_losses))
    np.save('{}/test_losses_seed_{}.npy'.format(folder_path, p['seed']), np.array(test_losses))


if __name__ == '__main__':
    
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    # Set the desired seed(s)
    seeds = p['seed']  # Example seeds

    if len(seeds) > 1:
        mp.set_start_method('spawn')
        # Run the `train` function in parallel with different seeds
        with mp.Pool() as pool:
            pool.map(train, seeds)
    else:
        # Run the `train` function once with a single seed that is specified in the config file
        train(seed=seeds[0])

