import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from model import get_gnn_model
from sklearn.metrics import roc_auc_score


def tc_loss(zs, m):
    means = zs.mean(0).unsqueeze(0)
    res = ((zs.unsqueeze(2) - means.unsqueeze(1)) ** 2).sum(-1)
    pos = torch.diagonal(res, dim1=1, dim2=2)
    offset = torch.diagflat(torch.ones(zs.size(1))).unsqueeze(0).cuda() * 1e6
    neg = (res + offset).min(-1)[0]
    loss = torch.clamp(pos + m - neg, min=0).mean()
    return loss


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Conv') != -1:
        init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
    elif classname.find('Linear') != -1:
        init.eye_(m.weight)
    elif classname.find('Emb') != -1:
        init.normal(m.weight, mean=0, std=0.01)


class netC1(nn.Module):
    def __init__(self, d, ndf, nc):
        super(netC1, self).__init__()
        self.trunk = nn.Sequential(
        nn.Conv1d(d, ndf, kernel_size=1, bias=False),
        )
        self.head = nn.Sequential(
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(ndf, nc, kernel_size=1, bias=True),
        )

    def forward(self, input):
        tc = self.trunk(input)
        ce = self.head(tc)
        return tc, ce


class GOAD:
    def __init__(self, input_channels, hidden_channels, dropout, lr, batch_size, nclass):
        self.m = 1
        self.lmbda = 0.1
        self.batch_size = batch_size
        self.input_channels = input_channels

        hidden_channels = 128

        self.ndf = 8
        self.n_rots = nclass
        self.eps = 0

        self.n_epoch = 20
        self.gnn = get_gnn_model(model_str='gpr',
                                 in_channels=input_channels,
                                 hidden_channels=hidden_channels,
                                 out_channels=hidden_channels,
                                 num_layers=2, dropout=dropout,
                                 dropout_adj=0.1,
                                 sparse=1).cuda()
        self.netC = netC1(4, self.ndf, self.n_rots).cuda()
        weights_init(self.netC)
        self.optimizerC = optim.Adam(list(self.netC.parameters()) + list(self.gnn.parameters()), lr=lr, betas=(0.5, 0.999))
        # self.optimizerC = optim.Adam(self.netC.parameters(), lr=lr, betas=(0.5, 0.999))
        self.decision_scores_ = None

    def fit_trans_classifier(self, x, g, train_mask, val_mask, test_mask, label):
        labels = torch.arange(self.n_rots).unsqueeze(0).expand((self.batch_size, self.n_rots)).long().cuda()
        celoss = nn.CrossEntropyLoss()
        train_inds = torch.nonzero(train_mask).squeeze(1)
        val_inds = torch.nonzero(val_mask).squeeze(1)
        best_val_auc = .0
        best_test_pred = None
        bad_counter = 0
        print('Training')
        for epoch in range(self.n_epoch):
            self.netC.train()
            self.gnn.train()

            train_inds_rand = torch.randperm(len(train_inds)).to(train_inds.device)

            train_inds_batches = train_inds_rand.split(self.batch_size)

            n_batch = 0
            sum_zs = torch.zeros((self.ndf, self.n_rots)).cuda()

            for i in range(0, len(train_inds_batches)):
                self.netC.zero_grad()
                self.gnn.zero_grad()
                train_labels = labels
                if i == len(train_inds_batches) - 1:
                    train_labels = torch.arange(self.n_rots).unsqueeze(0).expand((len(train_inds_batches[i]), self.n_rots)).long().cuda()

                hidden_output = self.gnn(x, g)
                hidden_output = hidden_output.reshape((hidden_output.shape[0], 4, -1))[train_inds_batches[i]]

                tc_zs, ce_zs = self.netC(hidden_output)
                sum_zs = sum_zs + tc_zs.mean(0)
                tc_zs = tc_zs.permute(0, 2, 1)

                loss_ce = celoss(ce_zs, train_labels)
                er = self.lmbda * tc_loss(tc_zs, self.m) + loss_ce
                er.backward()
                self.optimizerC.step()
                n_batch += 1

            means = sum_zs.t() / n_batch
            means = means.unsqueeze(0)
            self.netC.eval()
            self.gnn.eval()

            with torch.no_grad():
                hidden_output = self.gnn(x, g)
                hidden_output = hidden_output.reshape((hidden_output.shape[0], 4, -1))
                hidden_output_val = hidden_output[val_inds]
                hidden_output_test = hidden_output[test_mask]

                zs, fs = self.netC(hidden_output_val)
                zs = zs.permute(0, 2, 1)
                diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)

                diffs_eps = self.eps * torch.ones_like(diffs)
                diffs = torch.max(diffs, diffs_eps)
                logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)

                val_probs_rots = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()

                val_probs_rots = val_probs_rots.sum(1)



                auc = roc_auc_score(label[val_mask].cpu().numpy(), val_probs_rots)

                print(f'AUC: {auc:.4f}')
                if auc > best_val_auc:
                    best_val_auc = auc
                    zs, fs = self.netC(hidden_output_test)
                    zs = zs.permute(0, 2, 1)
                    diffs = ((zs.unsqueeze(2) - means) ** 2).sum(-1)
                    diffs_eps = self.eps * torch.ones_like(diffs)
                    diffs = torch.max(diffs, diffs_eps)
                    logp_sz = torch.nn.functional.log_softmax(-diffs, dim=2)
                    test_probs_rots = -torch.diagonal(logp_sz, 0, 1, 2).cpu().data.numpy()
                    test_probs_rots = test_probs_rots.sum(1)
                    best_test_pred = test_probs_rots

                    bad_counter = 0
                else:
                    bad_counter += 1
                    if bad_counter >= 5:
                        break

        return best_test_pred
