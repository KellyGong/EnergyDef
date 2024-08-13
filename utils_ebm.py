import math
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, average_precision_score, \
    precision_recall_curve, roc_curve, auc
from collections import namedtuple
from pygod.metric import eval_precision_at_k


EPS = 1e-5


class Langevin_Sampler(object):
    def __init__(self, in_dim, args, edge_predictor=None, ood_samples=None, update=False):
        self.in_dim = in_dim
        self.k = ood_samples.shape[0]
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size
        self.nsteps = args.nsteps
        self.device = args.device
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        self.buffer_rate = args.buffer_rate
        self.update = update
        self.edge_predictor = edge_predictor

        self.replay_buffer = self._init_replay_buffer(ood_samples)

    def _init_replay_buffer(self, ood_samples):
        assert self.k <= self.buffer_size
        buffer = torch.FloatTensor(self.buffer_size, self.in_dim).uniform_(-1, 1)
        if self.k > 0:
            buffer[:self.k] = ood_samples
        return buffer

    def _init_random(self, n):
        return torch.FloatTensor(n, self.in_dim).uniform_(-1, 1)

    def sample_p_0(self):
        if self.k == 0:
            inds = torch.randint(0, self.buffer_size, (self.batch_size,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size)
            buffer_or_random = (torch.rand(self.batch_size) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
        else:
            assert self.batch_size > self.k
            inds = torch.randint(self.k, self.buffer_size, (self.batch_size - self.k,))
            buffer_samples = self.replay_buffer[inds]
            random_samples = self._init_random(self.batch_size - self.k)
            buffer_or_random = (torch.rand(self.batch_size - self.k) < self.buffer_rate).float().unsqueeze(1)
            sample_0 = buffer_or_random * buffer_samples + (1 - buffer_or_random) * random_samples
            sample_0 = torch.cat((sample_0, self.replay_buffer[:self.k]), dim=0)
        return sample_0, inds

    def sample_q(self, model, replace_inds, features, g):
        model.eval()
        init_sample, buffer_inds = self.sample_p_0()
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)

        # langevin dynamics
        for i in range(self.nsteps):
            replaced_features = self.generate_q_features(x_k, features, replace_inds)
            mean_energy = model(replaced_features, g)[replace_inds].mean()
            # print(f'mean_energy: {mean_energy:.4f}')
            f_prime = torch.autograd.grad(mean_energy, [x_k], retain_graph=True)[0]
            x_k.data = x_k.data - self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)

        model.train()
        final_samples = x_k.detach()
        # update replay buffer
        # not update ground truth fraud samples
        if self.update:
            update_inds = buffer_inds[buffer_inds >= self.k]
            self.replay_buffer[update_inds] = final_samples.cpu()[torch.randperm(self.batch_size)[:update_inds.shape[0]]]
        return self.generate_q_features(final_samples, features, replace_inds)

    def generate_q_features(self, q_features, features, replace_inds):
        replaced_features = features.clone()
        replaced_features[replace_inds] = q_features
        return replaced_features


def grad_out(model, mask, features, adj, g):
    model.eval()
    x = torch.autograd.Variable(features, requires_grad=True)
    mean_energy = model(x, adj, g)[mask].mean()
    f_prime = torch.autograd.grad(mean_energy, [x], retain_graph=True)[0]

    grad_norm = torch.norm(f_prime, p=2, dim=-1)
    model.train()
    return grad_norm


def best_f1_value_threshold(preds, labels):
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    return thresholds[best_f1_score_index]


Evaluation_Metrics = namedtuple('Evaluation_Metrics', ['accuracy',
                                                       'macro_F1',
                                                       'recall',
                                                       'auc',
                                                       'ap',
                                                       'gmean',
                                                       'fpr95'])


def fpr95_score(pred, label):
    # calculate false positive rate (OOD -> ID) at 95% true negative rate(ID->ID)
    pred = 1 - pred
    label = 1 - label
    fpr, tpr, thresh = roc_curve(label, pred)
    return fpr[np.where(tpr > 0.95)[0][0]]


def evaluation_ebm_prediction(energy, label, threshold=None):
    # threshold are chosen in training set !!

    energy_norm = (energy - min(energy)) / (max(energy) - min(energy) + EPS)

    if threshold is None:
        threshold = best_f1_value_threshold(energy_norm, label)

    pred_label = np.where(energy_norm > threshold, 1, 0)

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    auc_value = roc_auc_score(label, energy_norm)
    precision_aupr, recall_aupr, _ = precision_recall_curve(label, energy_norm)
    ap = auc(recall_aupr, precision_aupr)
    # ap = average_precision_score(label, energy_norm, average='macro')
    gmean_value = gmean(label, pred_label)
    fpr95 = fpr95_score(energy_norm, label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc_value, ap=ap, gmean=gmean_value,
                              fpr95=fpr95), threshold


def evaluation_model_prediction(pred_logit, label):
    pred_label = np.argmax(pred_logit, axis=1)
    pred_logit = pred_logit[:, 1]

    accuracy = accuracy_score(label, pred_label)
    f1 = f1_score(label, pred_label, average='macro')
    recall = recall_score(label, pred_label, average='macro')
    auc_value = roc_auc_score(label, pred_logit)
    precision_aupr, recall_aupr, _ = precision_recall_curve(label, pred_logit)
    ap = auc(recall_aupr, precision_aupr)
    gmean_value = gmean(label, pred_label)
    fpr95 = fpr95_score(pred_logit, label)

    return Evaluation_Metrics(accuracy=accuracy, macro_F1=f1, recall=recall, auc=auc_value, ap=ap, gmean=gmean_value,
                              fpr95=fpr95)


def gmean(y_true, y_pred):
    """binary geometric mean of  True Positive Rate (TPR) and True Negative Rate (TNR)

    Args:
            y_true (np.array): label
            y_pred (np.array): prediction
    """

    TP, TN, FP, FN = 0, 0, 0, 0
    for sample_true, sample_pred in zip(y_true, y_pred):
        TP += sample_true * sample_pred
        TN += (1 - sample_true) * (1 - sample_pred)
        FP += (1 - sample_true) * sample_pred
        FN += sample_true * (1 - sample_pred)

    return math.sqrt(TP * TN / (TP + FN) / (TN + FP))
