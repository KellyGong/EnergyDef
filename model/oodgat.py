import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Linear, Parameter
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax


class OODGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=4, adjust=True, drop_edge=0, add_self_loop=True, drop_prob=0, bias=True, drop_input=0):
        super(OODGAT, self).__init__()

        self.conv1 = OODGATConv(in_dim, hidden_dim, heads, adjust, True, drop_edge, add_self_loop, bias)
        self.conv2 = OODGATConv(hidden_dim * heads, out_dim, heads, adjust, False, drop_edge, add_self_loop, bias)
        self.drop_prob = drop_prob
        self.drop_input = drop_input

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.ndata['h'], data.edges()
        edge_index = torch.stack((edge_index[0], edge_index[1]))
        if not return_attention_weights:
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x = F.elu(self.conv1(x, edge_index, False))
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x = self.conv2(x, edge_index, False)
            return x
        else:
            attention = []
            x = F.dropout(x, p=self.drop_input, training=self.training)
            x, a = self.conv1(x, edge_index, True)
            attention.append(a)
            x = F.elu(x)
            x = F.dropout(x, p=self.drop_prob, training=self.training)
            x, a = self.conv2(x, edge_index, True)
            attention.append(a)
            return (x, attention)


def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return Parameter(initial)


def glorot_init_2(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return initial


class OODGATConv(MessagePassing):
    def __init__(self, in_dim, out_dim, heads, adjust=True, concat=True, dropout=0.0,
                 add_self_loops= True, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(OODGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.adjust = adjust
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = glorot_init(in_dim, heads * out_dim)
        # The learnable parameters to compute attention coefficients:
        self.att_q = Parameter(glorot_init_2(heads, out_dim).unsqueeze(0))
        if adjust:
            self.att_v = Parameter(glorot_init_2(heads, out_dim).unsqueeze(0))
        if bias and concat:
            self.bias = Parameter(torch.zeros(heads * out_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, edge_index, return_attention_weights=False):
        H, C = self.heads, self.out_dim
        # We first transform the input node features.
        x = torch.matmul(x, self.lin).view(-1, H, C)  # x: [N, H, C]
        # Next, we compute node-level attention coefficients
        alpha = (x * self.att_q).sum(dim=-1) # alpha: [N, H]

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index, x=x, alpha=alpha) # out: [N, H, C]

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if return_attention_weights:
            return (out, alpha)
        else:
            return out

    def message(self, x_i, x_j, alpha_j, alpha_i, index):
        edge_weight_alpha = 1 - torch.abs(F.sigmoid(alpha_i) - F.sigmoid(alpha_j))
        if self.adjust:
            edge_weight_beta = (self.att_v * F.leaky_relu(x_i + x_j)).sum(-1)
            edge_weight = edge_weight_alpha * edge_weight_beta
        else:
            edge_weight = edge_weight_alpha
        edge_weight = softmax(edge_weight, index)
        edge_weight = F.dropout(edge_weight, p=self.dropout, training=self.training)

        return x_j * edge_weight.unsqueeze(-1)


