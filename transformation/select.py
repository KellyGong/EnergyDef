from .goad import GOAD
from .neutralAD import NeutralAD


def get_trans_model(model_str, nfeats, hidden, dropout, lr, batch_size, nclass=32):
    if model_str == 'goad':
        Model = GOAD
    elif model_str == 'neutralAD':
        Model = NeutralAD
    else:
        raise NotImplementedError

    return Model(input_channels=nfeats,
                 hidden_channels=hidden,
                 dropout=dropout,
                 lr=lr,
                 batch_size=batch_size,
                 nclass=nclass)
