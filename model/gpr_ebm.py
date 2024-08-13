import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from .gcn import GCNConv_dgl


class GPR_EBM(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GPR_EBM, self).__init__()

        self.inlinear = nn.Linear(in_channels, hidden_channels)
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, g=None):
        x = self.inlinear(x)
        energy = self.gnn.forward(x, g)
        return energy


class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.energy_layers = nn.ModuleList([nn.Linear(hidden_channels, 1) for _ in range(num_layers + 1)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, g=None):
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        energy = self.energy_layers[0](x) * self.temp[0]
        for i, conv in enumerate(self.layers):
            x = conv(x, g)
            x = F.relu(x)
            energy += self.energy_layers[i+1](x) * self.temp[i+1]
            x = F.dropout(x, p=self.dropout, training=self.training)
        return energy


class GPRConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GPRConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g, energy):
        with g.local_scope():
            g.ndata['e'] = torch.clamp(energy.clone(), 0, 10)
            g.apply_edges(fn.u_add_v('e', 'e', 'e_e'))
            g.edata['w_new'] = g.edata['w'] * (1.0 / (1 + torch.exp(g.edata['e_e'].squeeze(1))))
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w_new', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']
