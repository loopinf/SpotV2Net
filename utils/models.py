# -*- coding: utf-8 -*-
"""
Created on Tue May 23 14:53:24 2023

@author: ab978
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv,GATv2Conv
from torch_geometric_temporal.nn.recurrent import DCRNN,A3TGCN
import sys, pdb


class RecurrentGCN(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_channels, 
                 output_node_channels, 
                 dropout=0.0,
                 activation='relu'):
        super(RecurrentGCN, self).__init__()
        self.num_features = num_features
        self.activation = activation
        self.dropout = dropout
        self.grnn = A3TGCN(in_channels=1, 
                            out_channels=hidden_channels,
                            periods=12)
        self.linear = torch.nn.Linear(hidden_channels, output_node_channels)
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
        if self.activation == 'relu':
            self.a = F.relu
        elif self.activation == 'tanh':
            self.a = F.tanh
        elif self.activation == 'sigmoid':
            self.a = F.sigmoid        
        else:
            print('Choose an available activation function')
            sys.exit()
        

   
        
    def forward(self, data):

        x, edge_index, edge_attr = (data.x,
                                       data.edge_index,
                                       data.edge_attr)
        # pdb.set_trace()
        h = self.grnn(x, edge_index, edge_attr)
        h = self.a(h)
        if self.dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)
        y_x = self.linear(h)

        return y_x.view(-1)

class GATModel(torch.nn.Module):
    def __init__(self, 
                 num_features, 
                 hidden_channels, 
                 num_heads, 
                 output_node_channels, 
                 seq_length, 
                 num_hidden_layers=2, 
                 dropout_att=0.0, 
                 dropout=0.0,
                 activation='relu',
                 concat_heads=False):
        super(GATModel, self).__init__()
        self.seq_length = seq_length
        self.dropout = dropout
        self.num_features = num_features
        self.activation = activation
        first_gat = [GATConv(in_channels=num_features, out_channels=hidden_channels, heads=num_heads,  
                                    concat=concat_heads, dropout=dropout_att, edge_dim=num_features)]
                     
        stacked_gats = [GATConv(in_channels=hidden_channels * num_heads, out_channels=hidden_channels, heads=num_heads, 
                                    concat=concat_heads, dropout=dropout_att, edge_dim=num_features) for i in range(num_hidden_layers-1)]
        
        self.gat_layers = nn.ModuleList(first_gat + stacked_gats)
        if concat_heads:
            self.linear = torch.nn.Linear(hidden_channels * num_heads, output_node_channels)
        else:
            self.linear = torch.nn.Linear(hidden_channels, output_node_channels)
        # Apply Xavier initialization to the linear layers
        torch.nn.init.xavier_uniform_(self.linear.weight)
        
        if self.activation == 'relu':
            self.a = F.relu
        elif self.activation == 'tanh':
            self.a = F.tanh
        elif self.activation == 'sigmoid':
            self.a = F.sigmoid        
        else:
            print('Choose an available activation function')
            sys.exit()

        

        
    def forward(self, data):
        x, edge_index, edge_attr = (data.x,
                                    data.edge_index,
                                    data.edge_attr)
        
        for l in self.gat_layers[:-1]:
            x = l(x, edge_index, edge_attr)
            x = self.a(x)
            if self.dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat_layers[-1](x, edge_index, edge_attr)
        x = self.linear(x)

        return x.view(-1) # enforce positivity of the output