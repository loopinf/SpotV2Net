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
import torch.nn as nn
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
            for key in f.keys():
                cov_matrix = np.array(f[key])

                # Store only upper triangle of the covariance matrix (excluding diagonal)
                upper_tri = np.triu(cov_matrix, k=1)

                # Convert covariance matrix to edge_index and edge_attr
                edge_index = []
                edge_attr = []
                for i in range(upper_tri.shape[0]):
                    for j in range(i+1, upper_tri.shape[1]):
                        edge_index.append((i, j))
                        edge_attr.append(upper_tri[i, j])

                # Convert to tensors
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

                # Extract the variances (diagonal) as node features
                node_features = np.diag(cov_matrix)
                x = torch.tensor(node_features, dtype=torch.float).view(-1, 1)

                # Create PyTorch Geometric Data object
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        
def create_sequences(dataset, window_size):
    X_node, X_edge, Y = [], [], []
    
    # Get the edge index tensor for the entire dataset
    edge_index = dataset[0].edge_index

    for i in tqdm(range(len(dataset) - window_size)):
        x_node_seq = [dataset[j].x for j in range(i, i + window_size)]
        x_edge_seq = [dataset[j].edge_attr for j in range(i, i + window_size)]

        y = dataset[i + window_size].edge_attr  # Now predicting edge features (covariances)

        X_node.append(torch.stack(x_node_seq))
        X_edge.append(torch.stack(x_edge_seq))
        Y.append(y)

    return torch.stack(X_node), torch.stack(X_edge), torch.stack(Y), edge_index


class GATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, concat=False, dropout=0.6, edge_index=None):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, concat=concat, dropout=dropout)
        self.edge_index = edge_index
        
    def forward(self, x):
        pdb.set_trace()
        x = F.relu(self.gat_conv(x, self.edge_index))
        return x


class GATRNN(nn.Module):
    def __init__(self, num_nodes, node_in_channels, edge_in_channels, 
                 hidden_channels, rnn_hidden_size, num_layers, num_heads=1, 
                 dropout=0.6, edge_index=None):
        super(GATRNN, self).__init__()
        self.gat_layer = GATLayer(node_in_channels, hidden_channels, heads=num_heads, dropout=dropout, edge_index=edge_index)
        self.rnn = nn.GRU(hidden_channels * num_nodes, rnn_hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, edge_in_channels * (num_nodes * (num_nodes - 1)) // 2)

    def forward(self, x_node, x_edge):
        batch_size, seq_len, num_nodes, _ = x_node.size()
        x_edge = x_edge.view(batch_size, seq_len, num_nodes * (num_nodes - 1) // 2)
        x = torch.cat((x_node, x_edge.unsqueeze(2).expand(-1, -1, num_nodes, -1)), dim=-1)
        pdb.set_trace()
        gat_outputs = []
        for t in range(seq_len):
            batch_gat_outputs = []
            for b in range(batch_size):
                gat_out = self.gat_layer(x[b, t])
                batch_gat_outputs.append(gat_out.view(1, -1))

            batch_gat_outputs = torch.cat(batch_gat_outputs, dim=0)
            gat_outputs.append(batch_gat_outputs)

        gat_outputs = torch.stack(gat_outputs).permute(1, 0, 2)  # Reorder dimensions to (batch_size, seq_len, -1)
        rnn_out, _ = self.rnn(gat_outputs)
        out = self.fc(rnn_out[:, -1])

        return out
    
    
def train(model, criterion, optimizer, x_node_train, x_edge_train, y_train, batch_size):
    model.train()
    num_examples = x_node_train.shape[0]
    num_batches = num_examples // batch_size
    total_loss = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        x_node_batch = x_node_train[start_idx:end_idx]
        x_edge_batch = x_edge_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        optimizer.zero_grad()
        output = model(x_node_batch, x_edge_batch)
        loss = criterion(output, y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, criterion, x_node_test, x_edge_test, y_test, batch_size):
    model.eval()
    num_examples = x_node_test.shape[0]
    num_batches = num_examples // batch_size
    total_loss = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        x_node_batch = x_node_test[start_idx:end_idx]
        x_edge_batch = x_edge_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        with torch.no_grad():
            output = model(x_node_batch, x_edge_batch)
            loss = criterion(output, y_batch.view(-1))
        total_loss += loss.item()
    avg_loss = total_loss / num_batches
    return avg_loss



if __name__ == '__main__':
    
    
    # Instantiate the dataset
    dataset = CovarianceDataset(hdf5_file='processed_data/covs_mats_30min2.h5')
    print(dataset)
    
    # train-test split data
    num_data = len(dataset)
    num_train = int(num_data * 0.8)
    num_test = num_data - num_train
    
    train_mask = torch.zeros(num_data, dtype=torch.bool)
    train_mask[:num_train] = 1
    
    test_mask = torch.zeros(num_data, dtype=torch.bool)
    test_mask[num_train:] = 1
    
    dataset.train_mask = train_mask
    dataset.test_mask = test_mask
    
    window_size = 5
    X_node, X_edge, Y, edge_index = create_sequences(dataset, window_size)
    
    X_node_train, X_edge_train, Y_train = X_node[train_mask[:-window_size]], X_edge[train_mask[:-window_size]], Y[train_mask[:-window_size]]
    X_node_test, X_edge_test, Y_test = X_node[test_mask[:-window_size]], X_edge[test_mask[:-window_size]], Y[test_mask[:-window_size]]
    print(X_node_train.shape)
    print(X_edge_train.shape)
    print(Y_train.shape)
    print(edge_index.shape)
    print()
    print(X_node_test.shape)
    print(X_edge_test.shape)
    print(Y_test.shape)
    pdb.set_trace()

    

    # model
    num_nodes = dataset[0].num_nodes
    node_in_channels = dataset[0].x.size(1)
    edge_in_channels = dataset[0].edge_attr.size(1)
    hidden_channels = 64
    rnn_hidden_size = 128
    num_layers = 1
    num_heads = 2
    dropout = 0.6
    learning_rate = 0.001
    batch_size = 512
    
    model = GATRNN(num_nodes, node_in_channels, edge_in_channels, hidden_channels,
                   rnn_hidden_size, num_layers, num_heads, dropout, edge_index)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train
    num_epochs = 5
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        train_loss = train(model, criterion, optimizer, X_node_train, X_edge_train, Y_train, batch_size)
        val_loss = validate(model, criterion, X_node_test, X_edge_test, Y_test,batch_size)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
    
    # diagnostics
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    
    
    
    
    