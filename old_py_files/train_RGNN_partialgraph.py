# -*- coding: utf-8 -*-
"""
Created on Wed May 10 09:46:28 2023

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


# To modify the code to have a graph that is not fully connected, you need to change the edge index of the PyTorch dataset. 
# In particular, you need to define a new edge index tensor that contains only the edges that you want to keep in the graph.


class CovarianceTemporalDataset(InMemoryDataset):
    def __init__(self, hdf5_file, root='processed_data/cached_datasets_temporal_partial/', transform=None, pre_transform=None, seq_length=None, pad_value=-999):
        self.hdf5_file = hdf5_file
        self.root = root
        self.seq_length = seq_length
        self.pad_value = pad_value
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
    
                    # Create edge_index tensor for a sparse connected graph
                    sparsity = 0.5 # set sparsity parameter to 0.5
                    edge_index = torch.tensor(np.argwhere(np.triu(np.ones_like(cov_matrix), k=1)), dtype=torch.long).t().contiguous()

                    # Remove edges to create sparse graph
                    # rng = np.random.RandomState(42)
                    keep_mask = np.random.choice([False, True], size=edge_index.shape[1], p=[1 - sparsity, sparsity])
                    edge_index = edge_index[:, keep_mask]
    
                    # Extract only upper triangle of the adjacency matrix (excluding diagonal) for edge weights
                    mask = np.triu(np.ones_like(adj_matrix), k=1) > 0
                    edge_weights = torch.tensor(adj_matrix[mask], dtype=torch.float)
                    edge_weights = edge_weights[keep_mask]
    
                    # Extract the variances (diagonal) as node features
                    node_features = np.diag(cov_matrix)
                    x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)
    
                    # Process the next covariance matrix to create target values
                    adj_matrix_next = next_cov_matrix.copy()
                    np.fill_diagonal(adj_matrix_next, 0)  # Set the diagonal to zero
    
                    # Extract only upper triangle of the next adjacency matrix (excluding diagonal) for edge weights
                    mask = np.triu(np.ones_like(adj_matrix_next), k=1) > 0
                    y_edge_weights = torch.tensor(adj_matrix_next[mask], dtype=torch.float).view(-1, 1)
                    y_edge_weights = y_edge_weights[keep_mask]
                    
                    # this is not padded because I need boolean also where there's no connection to mark the link as absent
                    y_edge_weights_binary = torch.Tensor(keep_mask.astype(int))

                    
                    # Create PyTorch Geometric Data object
                    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weights, y=y_edge_weights,
                                y_binary=y_edge_weights_binary)
                    seq_data_list.append(data)
                    
    
                # Combine the sequence of Data objects into a single object with the desired format    
                max_len = keep_mask.shape[0]

                seq_data = Data(x=torch.stack([d.x for d in seq_data_list], dim=0),
                                edge_index=fixed_padding([d.edge_index.reshape(-1,2) for d in seq_data_list],max_len,self.pad_value).reshape(self.seq_length,2,-1), 
                                edge_weight=fixed_padding([d.edge_weight for d in seq_data_list],max_len,self.pad_value).reshape(self.seq_length,-1),
                                y=fixed_padding([d.y for d in seq_data_list],max_len,self.pad_value).reshape(self.seq_length,-1,1),
                                y_binary=torch.stack([d.y_binary for d in seq_data_list], dim=0))


                data_list.append(seq_data)
                
        
        # I need to pad again
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def fixed_padding(tensor_list, fixed_length, pad_value):
    # assuming `tensor_list` is a list of tensors with variable dimensions
    padded_tensors = []
    
    for tensor in tensor_list:
        if tensor.dim() == 1:
            # if the tensor has a single dimension, reshape it to a tensor with shape [length, 1]
            tensor = tensor.unsqueeze(1)
        # get the length of the tensor along the first dimension
        length = tensor.shape[0]
        # calculate the amount of padding required
        pad_length = fixed_length - length
        # create a tensor with the appropriate shape and the specified padding value
        pad_tensor = torch.ones((pad_length, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device) * pad_value
        # concatenate the original tensor with the padding tensor along the first dimension
        padded_tensor = torch.cat([tensor, pad_tensor], dim=0)
        # append the padded tensor to the list of padded tensors
        padded_tensors.append(padded_tensor)
    # stack the tensors
    stacked_tensor = torch.stack(padded_tensors, dim=0)
    # this will result in a tensor of shape [num_tensors, fixed_length, num_features]
    return stacked_tensor

    
# to remove padded values
# padded_indices = sequence_tensor.eq(-999)
# non_padded_sequence = sequence_tensor[~padded_indices]




class GATModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, output_edge_channels, seq_length):
        super(GATModel, self).__init__()
        self.seq_length = seq_length
        self.gat1 = GATConv(num_features, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True, dropout=0.6)
        self.lstm = nn.LSTM(hidden_channels * num_heads, hidden_channels * num_heads, batch_first=True)
        self.edge_weight_predictor = torch.nn.Linear(hidden_channels * num_heads, output_edge_channels)
        self.link_classifier = torch.nn.Linear(hidden_channels * num_heads, 2)  # 2 classes for link classification
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.edge_weight_predictor.weight)
        torch.nn.init.xavier_uniform_(self.link_classifier.weight)

        
        
    def forward(self, data):
        batches_temporal_embedding = []
        # iterating over batches
        pdb.set_trace()
        for i in range(len(data)):
            temporal_embedding = []
            x, edge_index, edge_weights = data[i].x, data[i].edge_index, data[i].edge_weight
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
        
        # Link prediction
        link_logits = self.link_classifier(x[data.edge_index[0]] * x[data.edge_index[1]])
        link_pred = torch.sigmoid(link_logits)

        return edge_weights_pred, link_pred

    


if __name__ == '__main__':
    
    # Load hyperparam file
    with open('config/GNN_param.yaml', 'r') as f:
        p = yaml.safe_load(f)
    # Define the folder path
    folder_path = 'output/{}'.format(p['modelname'])
    
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the yaml file to the model folder
    with open('{}/GNN_param.yaml'.format(folder_path), 'w') as f:
        yaml.dump(p, f)

    # Instantiate the dataset
    dataset = CovarianceTemporalDataset(hdf5_file=p['datafile'], seq_length=p['seq_length'], pad_value=p['pad_value'])
    
    # train-test split data
    train_size = int(p['split_proportion'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # Create DataLoaders for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=p['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=p['batch_size'], shuffle=False)

    # Instantiate the GATModel
    model = GATModel(num_features=p['num_features'], 
                     hidden_channels=p['hidden_channels'], 
                     num_heads=p['num_heads'], 
                     output_edge_channels=p['output_edge_channels'], 
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
                pdb.set_trace()
                # Forward pass
                edge_weights_pred, node_features_pred = model(data)
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
            if epoch == 0 or avg_train_loss + p['tolerance'] < prev_train_loss:
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
                    edge_weights_pred, node_features_pred = model(data)
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
        model.load_state_dict(torch.load(modelweights))
        # Set loss function and optimizer
        # criterion = torch.nn.MSELoss()
        
        # Evaluate on the test set
        model.eval()

        with torch.no_grad():
            for data in tqdm(iterable=test_loader, desc='Testing batches...'):
                data = data.to(device)
                # Forward pass
                edge_weights_pred, node_features_pred = model(data)
                
                # Initialize the covariance matrix with zeros
                cov = torch.diag(node_features_pred.reshape(-1,))
                # Create a mask for the upper triangle of the covariance matrix
                mask = torch.triu(torch.ones_like(cov), diagonal=1).bool()
                # Fill the upper triangle of the covariance matrix with the covariances from edge_weights_pred
                cov[mask] = edge_weights_pred.flatten()
                # Mirror the upper triangle to the lower triangle to make the matrix symmetric
                cov = cov + cov.T - torch.diag(cov.diag())
                # check
                # cov.equal(cov.t())
                pdb.set_trace()
                
        
        #         # Compute loss
        #         loss_edge_weights = criterion(edge_weights_pred, data.y_edge_weight[::p['seq_length'],:,:].reshape(edge_weights_pred.shape))
        #         loss_node_features = criterion(node_features_pred, data.y_x[::p['seq_length'],:,:].reshape(node_features_pred.shape))
        #         loss = loss_edge_weights + loss_node_features
        #         test_loss += loss.item()
        
        # # Compute average test loss
        # avg_test_loss = test_loss / len(test_loader)
        
        # print(f"Test Loss: {avg_test_loss:.10f}")