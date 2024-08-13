import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv


class PolyConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_channels = in_channels
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_channels, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_channels = in_channels
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, d=2, batch=False, **kwargs):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(hidden_channels, hidden_channels, self.thetas[i], lin=False))
            else:
                self.conv.append(PolyConvBatch(hidden_channels, hidden_channels, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels*len(self.conv), hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, x, adj=None, g=None):
        h = self.linear(x)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(x), 0]).to(x.device)
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat),0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h


# heterogeneous graph
class BWGNN_Hetero(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, d=2, **kwargs):
        super(BWGNN_Hetero, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.hidden_channels = hidden_channels
        self.conv = [PolyConv(hidden_channels, hidden_channels, theta, lin=False) for theta in self.thetas]
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels*len(self.conv), hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, out_channels)
        self.act = nn.LeakyReLU()
        # print(self.thetas)
        # for param in self.parameters():
        #     print(type(param), param.size())

    def forward(self, x, adj=None, g=None):
        h = self.linear(x)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_all = []

        for relation in g.canonical_etypes:
            # print(relation)
            h_final = torch.zeros([len(x), 0])
            for conv in self.conv:
                h0 = conv(self.g[relation], h)
                h_final = torch.cat([h_final, h0], -1)
                # print(h_final.shape)
            h = self.linear3(h_final)
            h_all.append(h)

        h_all = torch.stack(h_all).sum(0)
        h_all = self.act(h_all)
        h_all = self.linear4(h_all)
        return h_all
