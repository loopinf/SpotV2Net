# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:37:17 2023

@author: ab978
"""

import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import h5py
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pdb, os
import yaml


class CovarianceTemporalDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_temporal/', transform=None, pre_transform=None, seq_length=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        super(CovarianceTemporalDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data_temp.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            for i in range(len(f) - self.seq_length):
                seq_data_list = []
                for j in range(self.seq_length):
                    cov_matrix = np.array(f[keys[i+j]])
                    next_cov_matrix  = np.array(f[keys[i+j+1]])
    
                    # Create the adjacency matrix from the covariance matrix
                    adj_matrix = cov_matrix.copy()
                    np.fill_diagonal(adj_matrix, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    edge_weights = torch.tensor(adj_matrix[mask], dtype=torch.float)
    
                    # Create edge_index tensor
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    y_edge_weight = torch.tensor(adj_matrix_next[mask], dtype=torch.float).view(-1, 1)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float).view(-1, 1)
                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights,
                                y_edge_weight=y_edge_weight, y_x=y_x)
                    seq_data_list.append(data)
    
                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0),
                                edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0),
                                edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0),
                                y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0),
                                y_x=torch.stack([d.y_x for d in seq_data_list], dim=0))
                
    
                data_list.append(seq_data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])






class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, output_edge_channels, output_node_channels, seq_length):
        super(GATModel, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.gat1 = GATConv(num_features, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.lstm = nn.LSTM(hidden_channels * num_heads, hidden_channels * num_heads, batch_first=True)
        self.edge_weight_predictor = torch.nn.Linear(hidden_channels * num_heads, output_edge_channels)
        self.node_feature_predictor = torch.nn.Linear(hidden_channels * num_heads, output_node_channels)
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.edge_weight_predictor.weight)
        torch.nn.init.xavier_uniform_(self.node_feature_predictor.weight)

        
        
    def forward(self, data, batch_size):
        batches_temporal_embedding = []
        
        # reshape to merge batch dimension and node dimension
        x, edge_index, edge_weights = (data.x.reshape(self.seq_length,-1, self.num_features),
                                       data.edge_index,
                                       data.edge_weight.reshape(self.seq_length,-1))
        
        temporal_embedding = []
        for j in range(x.shape[0]):
            # GAT layers
            h = self.gat1(x[j], edge_index[j], edge_weights[j])
            h = F.relu(h)
            h = F.dropout(h, p=0.6, training=self.training)

            h = self.gat2(h, edge_index[j], edge_weights[j])
            h = F.relu(h)
            temporal_embedding.append(h)
        batches_temporal_embedding.append(temporal_embedding)
        
        batches_temporal_embedding = torch.stack([torch.cat(inner_list, dim=0) for inner_list in batches_temporal_embedding], dim=0)
        # Reshape x to (batch_size*seq_length,nodes, hidden_features)
        x = batches_temporal_embedding.reshape(batches_temporal_embedding.shape[0] * self.seq_length, 
                                                -1, 
                                                batches_temporal_embedding.shape[-1])
        x, _ = self.lstm(x)
        x = x[::self.seq_length,:,:]
        x = x.reshape(x.shape[0]*x.shape[1],-1)
        
        edge_weights_pred = self.edge_weight_predictor(x[data.edge_index[0][0]] * x[data.edge_index[0][1]])
        # Predict node features for each node separately
        node_features_pred = self.node_feature_predictor(x)

        return edge_weights_pred, node_features_pred


def back_to_matcov(y,y_x):
    # Initialize the covariance matrix with zeros
    cov = torch.diag(y_x.reshape(-1,))
    # Create a mask for the upper triangle of the covariance matrix
    mask = torch.triu(torch.ones_like(cov), diagonal=1).bool()
    # Fill the upper triangle of the covariance matrix with the covariances from edge_weights_pred
    cov[mask] = y.flatten()
    # Mirror the upper triangle to the lower triangle to make the matrix symmetric
    cov = cov + cov.T - torch.diag(cov.diag())
    
    return cov


if __name__ == '__main__':
    
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    # Define the folder path
    folder_path = 'output/{}_{}'.format(p['modelname'],p['seq_length'])
    
    if p['train'] == True:
        # Check if the folder exists, and create it if it doesn't
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Save the yaml file to the model folder
        with open('{}/GNN_param.yaml'.format(folder_path), 'w') as f:
            yaml.dump(p, f)
    else:
        # Load hyperparam file
        with open('{}/GNN_param.yaml'.format(folder_path), 'r') as f:
            p = yaml.safe_load(f)
        p['train'] = False

    # Instantiate the dataset
    dataset = CovarianceTemporalDataset(hdf5_file=p['datafile'],root='_'.join([p['root'],str(p['seq_length'])]), seq_length=p['seq_length'])
    
    # train-test split data
    train_size = int(p['split_proportion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    pdb.set_trace()
    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)
    
    # Instantiate the GATModel
    model = GATModel(num_features=p['num_features'], 
                     hidden_channels=p['hidden_channels'], 
                     num_heads=p['num_heads'], 
                     output_edge_channels=p['output_edge_channels'], 
                     output_node_channels=p['output_node_channels'], 
                     seq_length=p['seq_length'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if p['train'] == True:
        # Set loss function and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=p['learning_rate'])
        
        # Train the model
        prev_train_loss = float('inf')
        for epoch in range(p['num_epochs']):
            model.train()
            total_loss = 0
            
            for data in tqdm(iterable=train_loader, desc='Training batches...'):
                data = data.to(device)
                # Forward pass
                edge_weights_pred, node_features_pred = model(data,p['batch_size'])
                # Compute loss
                loss_edge_weights = criterion(edge_weights_pred, data.y_edge_weight[::p['seq_length'],:,:].reshape(edge_weights_pred.shape))
                loss_node_features = criterion(node_features_pred, data.y_x[::p['seq_length'],:,:].reshape(node_features_pred.shape))
                loss = loss_edge_weights + loss_node_features
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                # Optimize
                optimizer.step()
                total_loss += loss.item()
        
            # Compute average training loss
            avg_train_loss = total_loss / len(train_loader)
            
            # Check if the average training loss improved by the specified tolerance
            if epoch == 0 or avg_train_loss + float(p['tolerance']) < prev_train_loss:
                # Update the previous training loss
                prev_train_loss = avg_train_loss
    
                # Save the model weights
                save_path = 'output/{}/{}_weights.pth'.format(p['modelname'],p['modelname'])
                torch.save(model.state_dict(), save_path)
        
            # Evaluate on the test set
            model.eval()
            test_loss = 0
        
            with torch.no_grad():
                for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                    data = data.to(device)
        
                    # Forward pass
                    edge_weights_pred, node_features_pred = model(data,p['batch_size'])
                    # Compute loss
                    loss_edge_weights = criterion(edge_weights_pred, data.y_edge_weight[::p['seq_length'],:,:].reshape(edge_weights_pred.shape))
                    loss_node_features = criterion(node_features_pred, data.y_x[::p['seq_length'],:,:].reshape(node_features_pred.shape))
                    loss = loss_edge_weights + loss_node_features
        
                    test_loss += loss.item()
        
            # Compute average test loss
            avg_test_loss = test_loss / len(test_loader)
        
            print(f"Epoch: {epoch+1}/{p['num_epochs']}, Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}")
        
    else:
        
        
        # Load saved model weights
        modelweights = os.path.join(folder_path,'{}_weights.pth'.format(p['modelname']))
        model.load_state_dict(torch.load(modelweights, map_location=device))
        # Set loss function and optimizer
        # criterion = torch.nn.MSELoss()
        # Evaluate on the test set
        model.eval()

        with torch.no_grad():
            for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                data = data.to(device)
                # Forward pass
                edge_weights_pred, node_features_pred = model(data,p['batch_size'])
                edge_weights_pred, node_features_pred = (edge_weights_pred.reshape(p['batch_size'],-1,1), 
                                                         node_features_pred.reshape(p['batch_size'],-1,1))
                
                y, y_x = (data.y_edge_weight.reshape(p['batch_size'],-1,1)[:,::p['seq_length'],:], 
                          data.y_x.reshape(p['batch_size'],-1,1)[:,::p['seq_length'],:])

                for i in range(p['batch_size']):
                    cov_hat = back_to_matcov(edge_weights_pred[i],node_features_pred[i])
                    cov = back_to_matcov(y[i],y_x[i])
                    
                    mse = torch.mean((cov_hat-cov)**2)
                    mae = torch.mean((cov_hat-cov))
                    
                    print('MSE',mse)
                    print('MAe',mae)
                    print()
                    
                    # check
                    # cov.equal(cov.t())

                
