import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import APPNPConv


class APPNP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse):
        super(APPNP, self).__init__()

        if not sparse:
            raise NotImplementedError
        else:
            self.gnn = APPNP_sparse(in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj)

    def forward(self, x, adj=None, g=None):
        return self.gnn.forward(x, adj, g)


class APPNP_sparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj):
        super(APPNP_sparse, self).__init__()

        self.layers = nn.ModuleList()
        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)
        self.appnp_conv = APPNPConv(k=3, alpha=0.1)
        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj=None, g=None):
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        
        x = self.inlinear(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.appnp_conv(g, x)
        x = self.outlinear(x)

        return x
