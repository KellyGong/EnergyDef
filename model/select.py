from .gcn import GCN
from .gpr import GPR
from .gat import GAT
from .gin import GIN
from .appnp import APPNP
from .gpr_att import GPR_ATT
from .bwgnn import BWGNN, BWGNN_Hetero
from .gpr_ebm import GPR_EBM


def get_gnn_model(model_str, in_channels, hidden_channels, out_channels, num_layers,
                  dropout, dropout_adj, sparse):
    if model_str == 'gcn':
        Model = GCN
    elif model_str == 'gpr':
        Model = GPR
    elif model_str == 'gat':
        Model = GAT
    elif model_str == 'gin':
        Model = GIN
    elif model_str == 'appnp':
        Model = APPNP
    elif model_str == 'gpr_att':
        Model = GPR_ATT
    elif model_str == 'bwgnn':
        Model = BWGNN
    elif model_str == 'bwgnn_hetero':
        Model = BWGNN_Hetero
    elif model_str == 'gpr_ebm':
        Model = GPR_EBM
    else:
        raise NotImplementedError

    return Model(in_channels=in_channels,
                 hidden_channels=hidden_channels,
                 out_channels=out_channels,
                 num_layers=num_layers,
                 dropout=dropout,
                 dropout_adj=dropout_adj,
                 sparse=sparse)
