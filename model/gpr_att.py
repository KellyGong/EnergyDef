import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import dgl.function as fn
from .utils import create_activation
from .gcn import GCNConv_dense, GCNConv_dgl

from torch import Tensor
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor
from torch_geometric.utils import sort_edge_index


class GPR_ATT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, dropout_adj, sparse, \
                       final_r=0.7, decay_interval=10, decay_r=0.1, non_linear='relu'):
        super(GPR_ATT, self).__init__()

        self.inlinear = nn.Linear(in_channels, hidden_channels)
        self.outlinear = nn.Linear(hidden_channels, out_channels)
        
        torch.nn.init.xavier_uniform_(self.inlinear.weight)
        torch.nn.init.xavier_uniform_(self.outlinear.weight)

        self.gnn = GPR_sparse(hidden_channels, num_layers, dropout, dropout_adj, non_linear)
        self.extractor = ExtractorMLP(hidden_channels, non_linear, dropout)

    def forward(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            g.edata['attn'] = self.extractor(h_gnn, g.edges())
            # g.edata['attn'] = self.sampling(self.extractor(h_gnn, g.edges()), training=self.training)
            h_gnn = self.gnn.forward(h, adj, g, edge_attn=True)
            x = self.outlinear(h_gnn)
        return x
    
    def gen_node_emb(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            h_gnn = self.extractor.feature_extractor(h_gnn)
            return h_gnn
    
    def gen_edge_attn(self, x, adj=None, g=None):
        with g.local_scope():
            h = self.inlinear(x)
            h_gnn = self.gnn.forward(h, adj, g)
            return self.extractor(h_gnn, g.edges())
    
    @staticmethod
    def sampling(att_log_logit, training, temp=1):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
    @staticmethod
    def get_r(decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    
    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att


class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, activation='relu', dropout=0.2):
        super(ExtractorMLP, self).__init__()
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(hidden_size * 2, hidden_size * 2),
        #     nn.Dropout(p=dropout),
        #     create_activation(activation),
        #     nn.Linear(hidden_size * 2, 1),
        # )
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(p=dropout),
            create_activation(activation),
            nn.Linear(hidden_size, hidden_size),
        )
        self.cos = nn.CosineSimilarity(dim=1)
        self._init_weight(self.feature_extractor)
        
    
    @staticmethod
    def _init_weight(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    @staticmethod
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]
    
    def symmetric(self, edge_index, attn_logits):
        row, col = edge_index
        trans_attn_logits = self.reorder_like(torch.stack(edge_index), torch.stack((col, row)), attn_logits)
        edge_attn = (trans_attn_logits + attn_logits) / 2
        return edge_attn

    def forward(self, emb, edge_index, batch=None):
        col, row = edge_index
        f1, f2 = emb[col], emb[row]
        # f12 = torch.cat([f1, f2], dim=-1)
        # attn_logits = self.feature_extractor(f12)
        # return self.symmetric(edge_index, attn_logits).squeeze(1)
        attn_logits = self.cos(self.feature_extractor(f1), self.feature_extractor(f2))
        return attn_logits


class GCNConv_dgl_attn(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl_attn, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        g.ndata['h'] = self.linear(x)
        g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
        return g.ndata['h']


class GPR_sparse(nn.Module):
    def __init__(self, hidden_channels, num_layers, dropout, dropout_adj, non_linear):
        super(GPR_sparse, self).__init__()

        self.layers = nn.ModuleList([GCNConv_dgl_attn(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.activations = nn.ModuleList([create_activation(non_linear) for _ in range(num_layers)])
        # GPR temprature initialize
        alpha = 0.1
        temp = alpha * (1 - alpha) ** np.arange(num_layers + 1)
        temp[-1] = (1 - alpha) ** num_layers
        self.temp = nn.Parameter(torch.from_numpy(temp))

        self.dropout = dropout
        self.dropout_adj_p = dropout_adj

    def forward(self, x, adj=None, g=None, edge_attn=False):
        if edge_attn:
            g.edata['w'] = g.edata['w'] * g.edata['attn']
        g.edata['w'] = F.dropout(g.edata['w'], p=self.dropout_adj_p, training=self.training)
        hidden = x * self.temp[0]
        for i, (conv, actication) in enumerate(zip(self.layers, self.activations)):
            x = conv(x, g)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            hidden += x * self.temp[i+1]
        return hidden
