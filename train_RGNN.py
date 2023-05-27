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
import torch.multiprocessing as mp
from utils.models import GATModel, RecurrentGCN
from utils.dataset import CovarianceTemporalDataset,CovarianceLaggedDataset
from utils.tools import get_mean_std

def train(seed=None, trial=None, p=None):
    
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

    # Instantiate the dataset
    dataset = CovarianceLaggedDataset(hdf5_file=p['datafile'],root='_'.join([p['root'],str(p['seq_length'])]), seq_length=p['seq_length'])
    # train-test split data
    train_size = int(p['split_proportion'] * len(dataset))
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]
    
    # normalize dataset
    if p['normalize']:
        mean,std = get_mean_std(train_dataset)

    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
    
    # Instantiate the model
    if p['modeltype'] == 'gat':
        model = GATModel(num_features=p['seq_length'], 
                         hidden_channels=p['hidden_channels'], 
                         num_heads=p['num_heads'], 
                         output_node_channels=p['output_node_channels'], 
                         seq_length=p['seq_length'],
                         num_hidden_layers=p['num_hidden_layers'],
                         dropout_att = p['dropout_att'],
                         dropout = p['dropout'],
                         activation = p['activation'],
                         concat_heads= p['concat_heads'])
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=p['learning_rate'])
    
    # Train the model
    train_losses, test_losses = [], []
    prev_train_loss = float('inf')
    for epoch in range(p['num_epochs']):
        model.train()
        total_loss = 0
        for data in tqdm(iterable=train_loader, desc='Training batches...'):
            data = data.to(device)
            if p['scale_up']:
                data.x = data.x * p['scale_up'] 
                data.edge_attr = data.edge_attr * p['scale_up'] 
            elif p['normalize']:
                pdb.set_trace()
                data.x = (data.x-mean)/std
                data.edge_attr = (data.edge_attr-mean)/std
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

        
        # Check if the average training loss improved by the specified tolerance
        if epoch == 0 or avg_train_loss + float(p['tolerance']) < prev_train_loss:
            # Update the previous training loss
            prev_train_loss = avg_train_loss
            # Save the model weights
            save_path = '{}/{}_weights_seed_{}.pth'.format(folder_path,p['modelname'],p['seed'])
            torch.save(model.state_dict(), save_path)
    
        # Evaluate on the test set
        model.eval()
        test_loss = 0
    
        with torch.no_grad():
            for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                data = data.to(device)
                if p['scale_up']:
                    data.x = data.x * p['scale_up'] 
                    data.edge_attr = data.edge_attr * p['scale_up'] 
                elif p['normalize']:
                    data.x = (data.x-mean)/std
                    data.edge_attr = (data.edge_attr-mean)/std
                # Forward pass
                y_x_hat = model(data)
                # Compute loss
                y_x = data.y_x
                loss = criterion(y_x_hat, y_x)
                test_loss += loss.item()
        # pdb.set_trace()
        # Compute average test loss
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        avg_test_rmse = math.sqrt(avg_test_loss)
        
        print(f"Epoch: {epoch+1}/{p['num_epochs']}, Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}, Train RMSE: {avg_train_rmse:.10f}, Test RMSE: {avg_test_rmse:.10f}")

    # pdb.set_trace()    
    # target_range = data.y_x.max() - data.y_x.min()
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
            
