import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from model import get_gnn_model
from sklearn.metrics import roc_auc_score


class DCL(nn.Module):
    def __init__(self,temperature=0.1):
        super(DCL, self).__init__()
        self.temp = temperature

    def forward(self, z):
        z = F.normalize(z, p=2, dim=-1)
        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1

        pos_sim = torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp) # n,k-1
        K = num_trans - 1
        scale = 1 / np.abs(K*np.log(1.0 / K))

        loss_tensor = (torch.log(trans_matrix) - torch.log(pos_sim)) * scale
        return loss_tensor.sum(1)


class TransformNet(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers=2):
        super(TransformNet, self).__init__()
        net = []
        input_dim = x_dim
        for _ in range(num_layers-1):
            net.append(nn.Linear(input_dim, h_dim, bias=False))
            net.append(nn.ReLU())
            input_dim = h_dim
        net.append(nn.Linear(input_dim, x_dim, bias=False))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, num_layers=2):

        super(Encoder, self).__init__()
        enc = []
        input_dim = x_dim
        for _ in range(num_layers - 1):
            enc.append(nn.Linear(input_dim, h_dim))
            enc.append(nn.ReLU())
            input_dim = h_dim

        self.enc = nn.Sequential(*enc)
        self.fc = nn.Linear(input_dim, z_dim)

    def forward(self, x):
        z = self.enc(x)
        z = self.fc(z)
        return z


class NeutralAD_Transformation(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout, lr, batch_size, nclass):
        super(NeutralAD_Transformation, self).__init__()
        self.enc = Encoder(hidden_channels, hidden_channels, hidden_channels).cuda()
        self.trans = nn.ModuleList([TransformNet(hidden_channels, hidden_channels) for _ in range(nclass)]).cuda()
        self.num_trans = nclass
        self.z_dim = hidden_channels

    def forward(self, x):
        x = x.type(torch.FloatTensor).cuda()
        x_T = torch.empty(x.shape[0], self.num_trans,x.shape[-1]).to(x)
        for i in range(self.num_trans):
            mask = self.trans[i](x)
            x_T[:, i] = mask + x
        x_cat = torch.cat([x.unsqueeze(1), x_T], 1)
        zs = self.enc(x_cat.reshape(-1, x.shape[-1]))
        zs = zs.reshape(x.shape[0], self.num_trans+1,self.z_dim)
        return zs



class NeutralAD:
    def __init__(self, input_channels, hidden_channels, dropout, lr, batch_size, nclass):
        nclass = 11
        self.neural_transformation = NeutralAD_Transformation(input_channels, hidden_channels, dropout, lr, batch_size, nclass)
        self.num_trans = nclass
        self.gnn = get_gnn_model(model_str='gpr',
                                 in_channels=input_channels,
                                 hidden_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 num_layers=2, dropout=dropout,
                                 dropout_adj=0.1,
                                 sparse=1).cuda()
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(list(self.neural_transformation.parameters()) +
                                     list(self.gnn.parameters()), lr=lr)
        self.z_dim = hidden_channels
        self.loss_func = DCL()
        self.n_epoch = 20

    def fit_trans_classifier(self, x, g, train_mask, val_mask, test_mask, label):
        train_inds = torch.nonzero(train_mask).squeeze(1)
        val_inds = torch.nonzero(val_mask).squeeze(1)
        best_val_auc = .0
        best_test_pred = None
        bad_counter = 0
        print('Training')
        for epoch in range(self.n_epoch):
            self.neural_transformation.train()
            self.gnn.train()

            train_inds_rand = torch.randperm(len(train_inds)).to(train_inds.device)

            train_inds_batches = train_inds_rand.split(self.batch_size)

            for i in range(0, len(train_inds_batches)):
                self.optimizer.zero_grad()

                hidden_output = self.gnn(x, g)[train_inds_batches[i]]
                z = self.neural_transformation(hidden_output)
                loss = self.loss_func(z)
                loss_mean = loss.mean()
                loss_mean.backward()
                self.optimizer.step()

            self.neural_transformation.eval()
            self.gnn.eval()

            with torch.no_grad():
                hidden_output = self.gnn(x, g)
                hidden_output_val = hidden_output[val_inds]
                hidden_output_test = hidden_output[test_mask]

                val_scores = self.neural_transformation(hidden_output_val)
                val_scores = self.loss_func(val_scores)

                auc = roc_auc_score(label[val_mask].cpu().numpy(), val_scores.cpu().numpy())

                print(f'AUC: {auc:.4f}')
                if auc > best_val_auc:
                    best_val_auc = auc
                    test_scores = self.neural_transformation(hidden_output_test)
                    test_scores = self.loss_func(test_scores)
                    best_test_pred = test_scores

                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter >= 5:
                        break

        return best_test_pred.cpu().numpy()
