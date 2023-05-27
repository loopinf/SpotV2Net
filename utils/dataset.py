# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:53:35 2023

@author: ab978
"""

from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import os
import torch
import h5py, pdb

# reshape vs transpose https://discuss.pytorch.org/t/different-between-permute-transpose-view-which-should-i-use/32916

class CovarianceTemporalDataset(InMemoryDataset):
    '''This is the first dataset I implemented where the structure of the data seems not to be the one that the 
    GATConv layer is expected. '''
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
                # seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0),
                #                 edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0),
                #                 edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0),
                #                 y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0),
                #                 y_x=torch.stack([d.y_x for d in seq_data_list], dim=0))

                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0).transpose(0,1),
                                edge_index=torch.stack([d.edge_index for d in seq_data_list], dim=0).transpose(0,1),
                                edge_weight=torch.stack([d.edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_edge_weight=torch.stack([d.y_edge_weight for d in seq_data_list], dim=0).transpose(0,1),
                                y_x=torch.stack([d.y_x for d in seq_data_list], dim=0).transpose(0,1))
                
                

                data_list.append(seq_data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
        
class CovarianceLaggedDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_lagged/', transform=None, pre_transform=None, seq_length=None):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        super(CovarianceLaggedDataset, self).__init__(root, transform, pre_transform)
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
                    edge_attr = torch.tensor(adj_matrix[mask], dtype=torch.float)
    
                    # Create edge_index tensor
                    # TODO modify if we want to kill some connection. Right now the graph is full
                    edge_index = torch.tensor(np.argwhere(mask), dtype=torch.long).t().contiguous()
    
                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal)
                    mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    y_edge = torch.tensor(adj_matrix_next[mask], dtype=torch.float)
    
                    # Extract the variances (diagonal) of the next covariance matrix as target node features
                    y_x = torch.tensor(np.diag(next_cov_matrix), dtype=torch.float)
                    
                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y_edge=y_edge, y_x=y_x)
                    
                    seq_data_list.append(data)

                # Combine the sequence of Data objects into a single object with the desired format
                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=1),
                                edge_index=seq_data_list[-1].edge_index,
                                edge_attr=torch.stack([d.edge_attr for d in seq_data_list], dim=1),
                                y_edge=seq_data_list[-1].y_edge,
                                y_x=seq_data_list[-1].y_x)

                data_list.append(seq_data)


        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])