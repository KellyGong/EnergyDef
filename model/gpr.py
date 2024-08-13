import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from .gcn import GCNConv_dense, GCNConv_dgl


class GPR(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GPR, self).__init__()

        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)
        
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        if not sparse:
            self.gnn = GPR_dense(hidden_channels, num_layers, dropout, dropout_adj)
        else:
            self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, g, return_emb=False):
        x = self.inlinear(x)
        x = self.gnn.forward(x, g)
        if return_emb:
            x_out = self.outlinear(x)
            return x, x_out
        else:
            x = self.outlinear(x)
            return x


class GPR_dense(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_dense, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dense(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.dropout = dropout
        self.dropout_adj = nn.Dropout(p=dropout_adj)

        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))

    def forward(self, x, adj=None, g=None):

        adj = self.dropout_adj(adj)
        hidden = x * self.temp[0]
        
        for i, conv in enumerate(self.layers):
            x = conv(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden += x * self.temp[i+1]
        return hidden


class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl(hidden_channels, hidden_channels) for _ in range(num_layers)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, g=None):

        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        hidden = x * self.temp[0]

        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden += x * self.temp[i+1]
        return hidden
