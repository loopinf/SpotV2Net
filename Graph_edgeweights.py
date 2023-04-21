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
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import pdb, os

class CovarianceDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets/', transform=None, pre_transform=None):
        self.hdf5_file = hdf5_file
        self.root = root
        super(CovarianceDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [os.path.basename(self.hdf5_file)]

    @property
    def processed_file_names(self):
        return ['data.pt']
    
    @property
    def processed_dir(self):
        return self.root
    
    def process(self):
        data_list = []
    
        # Load covariance matrices from the HDF5 file
        with h5py.File(self.hdf5_file, 'r') as f:
            keys = list(f.keys())
            for i in range(len(f)-1):
                cov_matrix = np.array(f[keys[i]])
                next_cov_matrix  = np.array(f[keys[i+1]])
    
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
                data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y_edge_weight=y_edge_weight, y_x=y_x)
                data_list.append(data)
    
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])




class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, output_edge_channels, output_node_channels):
        super(GATModel, self).__init__()
        self.gat1 = GATConv(num_features, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.edge_weight_predictor = torch.nn.Linear(hidden_channels * num_heads, output_edge_channels)
        self.node_feature_predictor = torch.nn.Linear(hidden_channels * num_heads, output_node_channels)
        
        
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.edge_weight_predictor.weight)
        torch.nn.init.xavier_uniform_(self.node_feature_predictor.weight)
        
        
    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_weight


        x = self.gat1(x, edge_index, edge_weights)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.gat2(x, edge_index, edge_weights)
        x = F.relu(x)

        # Predict edge weights
        pdb.set_trace()
        edge_weights_pred = self.edge_weight_predictor(x[edge_index[0]] * x[edge_index[1]])

        # Predict node features
        node_features_pred = self.node_feature_predictor(x)
        

        return edge_weights_pred, node_features_pred

    




if __name__ == '__main__':
    
    
    # Instantiate the dataset
    dataset = CovarianceDataset(hdf5_file='processed_data/covs_mats_30min2_reduced.h5')
    # train-test split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # Create DataLoaders for train and test datasets
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Instantiate the GATModel
    model = GATModel(num_features=1, hidden_channels=32, num_heads=4, 
                     output_edge_channels=1, output_node_channels=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Set loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for data in tqdm(iterable=train_loader, desc='Training batches...'):
            data = data.to(device)
            # Forward pass
            edge_weights_pred, node_features_pred = model(data)
            # Compute loss
            loss_edge_weights = criterion(edge_weights_pred, data.y_edge_weight)
            loss_node_features = criterion(node_features_pred, data.y_x)
            loss = loss_edge_weights + loss_node_features
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            # Optimize
            optimizer.step()
            total_loss += loss.item()
    
        # Compute average training loss
        avg_train_loss = total_loss / len(train_loader)
    
        # Evaluate on the test set
        model.eval()
        test_loss = 0
    
        with torch.no_grad():
            for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                data = data.to(device)
    
                # Forward pass
                edge_weights_pred, node_features_pred = model(data)
    
                # Compute loss
                loss_edge_weights = criterion(edge_weights_pred, data.y_edge_weight)
                loss_node_features = criterion(node_features_pred, data.y_x)
                loss = loss_edge_weights + loss_node_features
    
                test_loss += loss.item()
    
        # Compute average test loss
        avg_test_loss = test_loss / len(test_loader)
    
        print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.10f}, Test Loss: {avg_test_loss:.10f}")