import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(GAT, self).__init__()

        if not sparse:
            raise NotImplementedError
        else:
            self.gnn = GAT_sparse(in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, adj=None, g=None):
        return self.gnn.forward(x, adj, g)


class GAT_sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(GAT_sparse, self).__init__()

        self.layers = nn.ModuleList()

        NUM_HEADS = 4

        self.layers.append(GATConv(in_channels, hidden_channels, num_heads=NUM_HEADS))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_channels * NUM_HEADS, hidden_channels, num_heads=NUM_HEADS))
        self.layers.append(GATConv(hidden_channels * NUM_HEADS, out_channels, num_heads=1))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj=None, g=None):
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(g, x).view([x.shape[0], -1])
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](g, x)
        return x.squeeze(1)
