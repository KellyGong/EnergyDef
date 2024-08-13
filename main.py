# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'


from utils import *
from utils_ebm import *
from utils_oodgat import *
from model import get_gnn_model
from data_loader import load_data
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm
import torch
import numpy as np
import copy
import argparse
import wandb
from scipy.spatial.distance import mahalanobis
from sklearn.svm import OneClassSVM
from transformation import get_trans_model
from model.oodgat import OODGAT

# torch.autograd.set_detect_anomaly(True)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.set_num_threads(8)

EOS = 1e-10

setup_seed(2022)


class Experiment:
    def __init__(self, args):
        super(Experiment, self).__init__()
        self.args = args
        self.device = args.device

    def get_classification_loss(self, model, mask, features, labels, g=None):
        logits = model(features, g)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[mask], labels[mask].long(), reduction='mean')
        eval_res = evaluation_model_prediction(
            logp[mask].exp().detach().cpu().numpy(), labels[mask].cpu().numpy())
        return loss, eval_res

    def get_contrastive_divergence(self, model, edge_predictor, mask, features, labels, args, g=None, threshold=None):
        with g.local_scope():
            # phi = edge_predictor(g.ndata['h'], g.edges())
            # g.edata['w'] = g.edata['w'] * phi
            energy = model(features, g).squeeze(1)
            # loss = self._energy_loss(energy[mask], labels[mask], args.margin_in, args.margin_out)
            loss = self._energy_loss2(energy[mask], labels[mask])
            # grad_norm = grad_out(model, mask, features, Adj, g)
            eval_res, threshold = evaluation_ebm_prediction(energy[mask].detach().cpu().numpy(), labels[mask].cpu().numpy(),
                                                            threshold)
            return loss, eval_res, threshold

    @staticmethod
    def _energy_loss(energy, labels, margin_in, margin_out):
        pos_loss = torch.clamp(energy[labels == 0] - margin_in, 0) ** 2
        neg_loss = torch.clamp(margin_out - energy[labels == 1], 0) ** 2
        loss = pos_loss.mean() + neg_loss.mean()
        return loss

    @staticmethod
    def _energy_loss2(energy, labels, reg=0.1, **kwargs):
        # contrastive_loss = margin_rank_loss(labels, energy)
        contrastive_loss = torch.mean(energy[labels == 0]) - torch.mean(energy[labels == 1])
        reg_loss = torch.mean(energy ** 2)
        # loss = contrastive_loss + reg * reg_loss
        loss = contrastive_loss
        return loss

    def train_classification_gcn(self, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args,
                                 g, model_str):
        model = get_gnn_model(model_str=model_str, in_channels=nfeats, hidden_channels=args.hidden,
                              out_channels=nclasses, num_layers=args.nlayers,
                              dropout=args.dropout, dropout_adj=args.dropout_adj, sparse=args.sparse)

        bad_counter = 0
        best_val = None
        best_loss = 0

        model = model.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        g = g.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss, train_res = self.get_classification_loss(
                model, train_mask, features, labels, g)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                val_loss, val_res = self.get_classification_loss(
                    model, val_mask, features, labels, g)

                if args.wandb:
                    wandb_log(loss, val_loss, val_res, epoch)

                if best_val is None or val_res.auc > best_val.auc:
                    bad_counter = 0
                    best_val = val_res
                    best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                    best_loss = val_loss
                    best_train_loss = loss
                    print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val fpr95 {:.4F}".format(
                        epoch, best_loss, best_val.auc, best_val.ap, best_val.fpr95))
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

        print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
            best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
        with torch.no_grad():
            model.eval()
            model.load_state_dict(best_model_weight)
            test_loss, test_res = self.get_classification_loss(
                model, test_mask, features, labels, g)
            print("Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test fpr95 {:.4F}".format(
                test_loss, test_res.auc, test_res.ap, test_res.fpr95))
            torch.save(model, 'model.pt')
        return best_val, test_res, model

    @staticmethod
    def load_model(path='model.pt', device='cpu'):
        model = torch.load(path, map_location=device)
        return model

    def train_single_gnn(self, args):
        assert args.sparse == 1
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)
            Adj = g.adj()

        for trial in range(args.ntrials):
            val_res, test_res, best_model = self.train_classification_gcn(Adj, features, nfeats, labels, nclasses,
                                                                          train_mask[trial], val_mask[trial],
                                                                          test_mask[trial],
                                                                          args, g)
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def train_sklearn(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []

        for trial in range(args.ntrials):
            model = get_sklearn_model(args.sklearn_model)
            if args.sklearn_model != 'LocalOutlier':
                model.fit(features[train_mask[trial]].numpy())
                val_scores = -model.score_samples(features[val_mask[trial]].numpy())
                val_res, threshold = evaluation_ebm_prediction(val_scores, labels[val_mask[trial]].numpy())
                test_scores = -model.score_samples(features[test_mask[trial]].numpy())
                test_res, _ = evaluation_ebm_prediction(test_scores, labels[test_mask[trial]].numpy(), threshold)

            else:
                model.fit(features.numpy())
                val_scores = -model.negative_outlier_factor_[val_mask[trial].numpy()]
                val_res, threshold = evaluation_ebm_prediction(val_scores, labels[val_mask[trial]].numpy())
                test_scores = -model.negative_outlier_factor_[test_mask[trial].numpy()]
                test_res, _ = evaluation_ebm_prediction(test_scores, labels[test_mask[trial]].numpy(), threshold)

            print("Val Auc {:.4f}, Val FPR@95 {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                val_res.auc, val_res.fpr95, val_res.ap, val_res.macro_F1))
            print("Test Auc {:.4f}, Test FPR@95 {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                test_res.auc, test_res.fpr95, test_res.ap, test_res.macro_F1))
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def train_pygod(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []

        g = convert_dgl_graph2pyg(g)

        for trial in range(args.ntrials):

            model = select_pygod_model(args.pygod_model, args.hidden, args.dropout, args.lr, args.w_decay,
                                       args.batch_size)
            model.fit(g, labels)
            scores = model.decision_scores_

            val_res, threshold = evaluation_ebm_prediction(scores, labels.numpy())

            test_res, _ = evaluation_ebm_prediction(scores[test_mask[trial]], labels[test_mask[trial]].numpy(),
                                                    threshold)

            print("Val Auc {:.4f}, Val FPR@95 {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                val_res.auc, val_res.fpr95, val_res.ap, val_res.macro_F1))

            print("Test Auc {:.4f}, Test FPR@95 {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                test_res.auc, test_res.fpr95, test_res.ap, test_res.macro_F1))
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def train_trans(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)
        g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[1], Adj.values(), features)

        for trial in range(args.ntrials):

            best_val = None

            model = get_trans_model(args.trans_model, nfeats, args.hidden, args.dropout, args.lr, args.batch_size, nclass=32)

            train_mask_trial = train_mask[trial].clone().to(self.device)
            val_mask_trial = val_mask[trial].clone().to(self.device)
            test_mask_trial = test_mask[trial].clone().to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            g = g.to(self.device)

            scores = model.fit_trans_classifier(features, g, train_mask_trial, val_mask_trial, test_mask_trial, labels)

            test_res, _ = evaluation_ebm_prediction(scores, labels[test_mask[trial]].cpu().numpy(),
                                                    0.5)

            print("Test Auc {:.4f}, Test FPR@95 {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                test_res.auc, test_res.fpr95, test_res.ap, test_res.macro_F1))
            val_results.append(best_val)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_test_results(test_results)
        return test_results

    def kshot_cross_entropy(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)

        for trial in range(args.ntrials):
            val_res, test_res, best_model = self.train_classification_gcn(features, nfeats, labels, nclasses,
                                                                          train_mask[trial], val_mask[trial],
                                                                          test_mask[trial],
                                                                          args, g, 'gpr')
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def kshot_pronet(self, args):
        def mahalanobis_distance(emb, labels, train_mask, val_mask):
            mu_ood = torch.mean(emb[(labels == 1) & train_mask], dim=0)
            mu_ind = torch.mean(emb[(labels == 0) & train_mask], dim=0)
            sigma_ood = (emb[(labels == 1) & train_mask] - mu_ood).T @  (emb[(labels == 1) & train_mask] - mu_ood)
            sigma_ind = (emb[(labels == 0) & train_mask] - mu_ind).T @  (emb[(labels == 0) & train_mask] - mu_ind)
            # sigma = (sigma_ood + sigma_ind) / torch.sum(train_mask)
            sigma_ind_inv = torch.linalg.inv(sigma_ind)
            sigma_ood_inv = torch.linalg.inv(sigma_ood)
            eval_dis = emb[val_mask].cpu().detach().numpy()
            in_distance = [mahalanobis(eval_dis[i], mu_ind.cpu().detach().numpy(), sigma_ind_inv.cpu().detach().numpy()) for i in range(len(eval_dis))]
            ood_distance = [mahalanobis(eval_dis[i], mu_ind.cpu().detach().numpy(), sigma_ind_inv.cpu().detach().numpy()) for i in range(len(eval_dis))]
            return ood_distance

        features, nfeats, labels, nclasses, train_mask_all, val_mask_all, test_mask_all, g = load_data(args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)
        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)

        for trial in range(args.ntrials):
            model = get_gnn_model(model_str='gpr', in_channels=nfeats, hidden_channels=args.hidden,
                                  out_channels=nclasses, num_layers=args.nlayers,
                                  dropout=args.dropout, dropout_adj=args.dropout_adj, sparse=args.sparse)

            bad_counter = 0
            best_val = None
            best_loss = 0
            best_threshold = 0

            model = model.to(self.device)
            train_mask = train_mask_all[trial].to(self.device)
            val_mask = val_mask_all[trial].to(self.device)
            test_mask = test_mask_all[trial].to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            g = g.to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                _, logits = model(features, g, return_emb=True)
                logp = F.log_softmax(logits, 1)
                loss = F.nll_loss(logp[train_mask], labels[train_mask].long(), reduction='mean')
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    emb, logits = model(features, g, return_emb=True)
                    logp = F.log_softmax(logits, 1)
                    val_loss = F.nll_loss(logp[val_mask], labels[val_mask].long(), reduction='mean')
                    mah_dis = mahalanobis_distance(emb, labels, train_mask, val_mask)
                    val_res, threshold = evaluation_ebm_prediction(mah_dis.detach().cpu().numpy(), labels[val_mask].cpu().numpy())

                    if args.wandb:
                        wandb_log(loss, val_loss, val_res, epoch)

                    if best_val is None or val_res.auc > best_val.auc:
                        bad_counter = 0
                        best_val = val_res
                        best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                        best_loss = val_loss
                        best_threshold = threshold
                        print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val fpr95 {:.4F}".format(
                            epoch, best_loss, best_val.auc, best_val.ap, best_val.fpr95))
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience:
                        break

            print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
            with torch.no_grad():
                model.eval()
                model.load_state_dict(best_model_weight)
                emb, logits = model(features, g, return_emb=True)
                test_loss = F.nll_loss(logp[val_mask], labels[val_mask].long(), reduction='mean')
                mah_dis = mahalanobis_distance(emb, labels, train_mask, test_mask)
                test_res, _ = evaluation_ebm_prediction(mah_dis.detach().cpu().numpy(),
                                                        labels[test_mask].cpu().numpy(), best_threshold)
                print("Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test fpr95 {:.4F}".format(
                       test_loss, test_res.auc, test_res.ap, test_res.fpr95))
                torch.save(model, 'model.pt')
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)

    def kshot_outlier_exposure(self, args):
        features, nfeats, labels, nclasses, train_mask_all, val_mask_all, test_mask_all, g = load_data(
            args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)

        for trial in range(args.ntrials):
            model = get_gnn_model(model_str='gpr_ebm', in_channels=nfeats, hidden_channels=args.hidden,
                                  out_channels=nclasses, num_layers=args.nlayers,
                                  dropout=args.dropout, dropout_adj=args.dropout_adj, sparse=args.sparse)

            bad_counter = 0
            best_val = None
            best_loss = 0
            best_threshold = 0

            model = model.to(self.device)
            train_mask = train_mask_all[trial].to(self.device)
            val_mask = val_mask_all[trial].to(self.device)
            test_mask = test_mask_all[trial].to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            g = g.to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.w_decay)

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()
                energy = model(features, g)
                loss = margin_rank_loss(labels[train_mask].long(), energy[train_mask])
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.eval()
                    val_loss = margin_rank_loss(labels[val_mask].long(), energy[val_mask])
                    val_res, threshold = evaluation_ebm_prediction(energy[val_mask].detach().cpu().numpy(), labels[val_mask].cpu().numpy())

                    if args.wandb:
                        wandb_log(loss, val_loss, val_res, epoch)

                    if best_val is None or val_res.auc > best_val.auc:
                        bad_counter = 0
                        best_val = val_res
                        best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                        best_loss = val_loss
                        best_threshold = threshold
                        print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val fpr95 {:.4F}".format(
                            epoch, best_loss, best_val.auc, best_val.ap, best_val.fpr95))
                    else:
                        bad_counter += 1

                    if bad_counter >= args.patience:
                        break

            print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
            with torch.no_grad():
                model.eval()
                model.load_state_dict(best_model_weight)
                energy = model(features, g)
                test_loss = margin_rank_loss(labels[test_mask].long(), energy[test_mask])
                test_res, _ = evaluation_ebm_prediction(energy[test_mask].detach().cpu().numpy(),
                                                     labels[test_mask].cpu().numpy(), best_threshold)
                print("Test Loss {:.4f}, Test Auc {:.4f}, Test AP {:.4f}, Test fpr95 {:.4F}".format(
                    test_loss, test_res.auc, test_res.ap, test_res.fpr95))
                torch.save(model, 'model.pt')
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)

    def train_entropy(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []
        class_label = g.ndata['class']

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)
            g.ndata['class'] = class_label

        for trial in range(args.ntrials):
            val_res, test_res, best_model = self.train_gnnsafe(features, nfeats, labels, nclasses,
                                                               train_mask[trial], val_mask[trial],
                                                               test_mask[trial],
                                                               args, g)
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def train_oodgat_epoch(self, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args, g):
        a = torch.tensor(0.9).to(args.device)
        b = torch.tensor(0.01).to(args.device)
        g = g.to(args.device)
        train_mask = train_mask.to(args.device)
        val_mask = val_mask.to(args.device)
        test_mask = test_mask.to(args.device)
        labels = labels.to(args.device)
        best_metric = 0
        patience = 0
        xent = nn.CrossEntropyLoss()
        model = OODGAT(nfeats, args.hidden, int(g.ndata['class'][train_mask].max() + 1), 4, False, args.dropout_adj, True, args.dropout, True).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        w_discrepancy = 5e-3
        w_consistent = 2.0
        w_ent = 0.05
        ent_loss_func = EntropyLoss(reduction=False)
        best_val = None
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = torch.zeros(1).to(args.device)
            logits, att = model(g, return_attention_weights=True)
            ent_loss = ent_loss_func(logits)  # ent_loss: N-dim tensor
            cos_loss_1 = get_consistent_loss_new(att[0].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                 f1=F.sigmoid, f2=F.sigmoid)
            cos_loss_2 = get_consistent_loss_new(att[1].T, (ent_loss - ent_loss.mean()) / ent_loss.std(),
                                                 f1=F.sigmoid, f2=F.sigmoid)
            consistent_loss = 0.5 * (cos_loss_1 + cos_loss_2)
            loss += torch.pow(a, b * epoch) * w_consistent * consistent_loss

            loss -= torch.pow(a, b * epoch) * w_discrepancy * cosine_similarity(att[0].mean(axis=1),
                                                                                att[1].mean(axis=1))

            loss += torch.pow(a, b * epoch) * w_ent * local_ent_loss(logits, att, int(g.ndata['class'][train_mask].max() + 1), 0.6)

            sup_loss = xent(logits[train_mask], g.ndata['class'][train_mask])
            loss += sup_loss
            loss.backward()
            optimizer.step()
            # validate
            with torch.no_grad():
                model.eval()
                logits, att = model(g, return_attention_weights=True)

                ATT = F.sigmoid(torch.hstack([att[0].detach(), att[1].detach()]).mean(axis=1)).cpu()
                val_res, threshold = evaluation_ebm_prediction(ATT[val_mask].cpu().numpy(),
                                                               labels[val_mask].cpu().numpy(), 0.5)

                if args.wandb:
                    wandb_log(loss, 0, val_res, epoch)

                if best_val is None or val_res.auc > best_val.auc:
                    bad_counter = 0
                    best_val = val_res
                    best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                    best_loss = 0
                    best_threshold = threshold
                    best_train_loss = loss
                    print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                        epoch, best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

        print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}, Val fpr95 {:.4f}".format(
            best_loss, best_val.auc, best_val.ap, best_val.macro_F1, best_val.fpr95))
        with torch.no_grad():
            model.eval()
            model.load_state_dict(best_model_weight)
            logits = model(g, return_attention_weights=True)
            ATT = F.sigmoid(torch.hstack([att[0].detach(), att[1].detach()]).mean(axis=1)).cpu()
            test_res, _ = evaluation_ebm_prediction(ATT[test_mask].cpu().numpy(), labels[test_mask].cpu().numpy(),
                                                    best_threshold)
            print("Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}, Test fpr95 {:.4f}".format(
                test_res.auc, test_res.ap, test_res.macro_F1, test_res.fpr95))
            torch.save(model, 'model.pt')
        return best_val, test_res, model

    def train_oodgat(self, args):
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(
            args)
        val_results = []
        test_results = []
        class_label = g.ndata['class']

        Adj = normalize(g.adj(), args.normalization, args.sparse)

        if args.sparse:
            g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[
                1], Adj.values(), features)
            g.ndata['class'] = class_label

        for trial in range(args.ntrials):
            val_res, test_res, best_model = self.train_oodgat_epoch(features, nfeats, labels, nclasses,
                                                                    train_mask[trial], val_mask[trial],
                                                                    test_mask[trial],
                                                                    args, g)
            val_results.append(val_res)
            test_results.append(test_res)
            if args.wandb:
                wandb_summary(test_res)
        print_results(val_results, test_results)
        return test_results

    def train_gnnsafe(self, features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, args,
                            g=None):
        model = get_gnn_model(model_str='gpr', in_channels=nfeats, hidden_channels=args.hidden,
                              out_channels=int(g.ndata['class'][train_mask].max() + 1), num_layers=args.nlayers,
                              dropout=args.dropout, dropout_adj=args.dropout_adj, sparse=args.sparse)

        bad_counter = 0
        best_val = None
        best_loss = 0
        best_threshold = 0

        model = model.to(self.device)
        train_mask = train_mask.to(self.device)
        val_mask = val_mask.to(self.device)
        test_mask = test_mask.to(self.device)
        features = features.to(self.device)
        labels = labels.to(self.device)
        g = g.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.w_decay)

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(features, g)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp[train_mask], g.ndata['class'][train_mask], reduction='mean')

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                logits = model(features, g)
                energy = -torch.log(logits.exp().sum(dim=1) + 1e-5)
                energy = torch.where(torch.isinf(energy), torch.zeros_like(energy), energy)
                val_res, threshold = evaluation_ebm_prediction(energy[val_mask].cpu().numpy(), labels[val_mask].cpu().numpy(), 0.5)

                if args.wandb:
                    wandb_log(loss, 0, val_res, epoch)

                if best_val is None or val_res.auc > best_val.auc:
                    bad_counter = 0
                    best_val = val_res
                    best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                    best_loss = 0
                    best_threshold = threshold
                    best_train_loss = loss
                    print("Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                        epoch, best_loss, best_val.auc, best_val.ap, best_val.macro_F1))
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

        print("Val Loss {:.4f}, Val Auc {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}, Val fpr95 {:.4f}".format(
            best_loss, best_val.auc, best_val.ap, best_val.macro_F1, best_val.fpr95))
        with torch.no_grad():
            model.eval()
            model.load_state_dict(best_model_weight)
            logits = model(features, g)
            energy = -logits.exp().sum(dim=1).log()
            energy = torch.where(torch.isinf(energy), torch.zeros_like(energy), energy)
            test_res, _ = evaluation_ebm_prediction(energy[test_mask].cpu().numpy(), labels[test_mask].cpu().numpy(), best_threshold)
            print("Test Auc {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}, Test fpr95 {:.4f}".format(
                test_res.auc, test_res.ap, test_res.macro_F1, test_res.fpr95))
            torch.save(model, 'model.pt')
        return best_val, test_res, model

    def train_ebm_gnn(self, args):
        assert args.sparse == 1
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, g = load_data(args)
        val_results = []
        test_results = []

        Adj = normalize(g.adj(), args.normalization, args.sparse)
        g = gen_dgl_graph(Adj.indices()[0], Adj.indices()[1], Adj.values(), features)
        best_test_res = None

        for trial in range(args.ntrials):
            edge_predictor = train_edge_predictor(g, nfeats, args.hidden, train_mask[trial], labels, self.device)

            model = get_gnn_model(model_str=args.model, in_channels=nfeats, hidden_channels=args.hidden,
                                  out_channels=nclasses, num_layers=args.nlayers,
                                  dropout=args.dropout, dropout_adj=args.dropout_adj, sparse=args.sparse)

            bad_counter = 0
            best_val = None
            best_loss = 0
            best_threshold = None

            model = model.to(self.device)
            train_mask_trial = train_mask[trial].clone().to(self.device)
            val_mask_trial = val_mask[trial].clone().to(self.device)
            test_mask_trial = test_mask[trial].clone().to(self.device)
            features = features.to(self.device)
            labels = labels.to(self.device)
            g = g.to(self.device)

            langevin_sampler = Langevin_Sampler(nfeats, args, edge_predictor, features[(labels * train_mask_trial) == 1])

            # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            optimizer = torch.optim.Adam(list(model.parameters()) + list(edge_predictor.parameters()), lr=args.lr, weight_decay=args.w_decay)

            for epoch in range(1, args.epochs + 1):
                model.train()
                optimizer.zero_grad()

                train_inds = torch.nonzero(train_mask_trial).squeeze(1)
                selected_train_oods = train_inds[torch.randperm(train_inds.shape[0])][:args.batch_size]
                update_features = langevin_sampler.sample_q(model, selected_train_oods, features, g)
                update_labels = labels.clone()
                update_labels[selected_train_oods] = 1
                loss, train_res, threshold = self.get_contrastive_divergence(
                    model, edge_predictor, train_mask_trial, update_features, update_labels, args, g)

                loss.backward()
                clip_grad_norm(model.parameters(), max_norm=5.0)
                optimizer.step()

                # with torch.no_grad():
                model.eval()
                val_loss, val_res, _ = self.get_contrastive_divergence(
                    model, edge_predictor, val_mask_trial, features, labels, args, g, threshold)

                if args.wandb:
                    wandb_log(loss, val_loss, val_res, epoch)

                if best_val is None or val_res.auc > best_val.auc:
                    bad_counter = 0
                    best_val = val_res
                    best_threshold = threshold
                    best_model_weight = {k: v.cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
                    best_loss = val_loss
                    print(
                        "Epoch {} Val Loss {:.4f}, Val Auc {:.4f}, Val FPR@95 {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                            epoch, best_loss, best_val.auc, best_val.fpr95, best_val.ap, best_val.macro_F1))
                else:
                    bad_counter += 1

                if bad_counter >= args.patience:
                    break

            print("Val Loss {:.4f}, Val Auc {:.4f}, Val FPR@95 {:.4f}, Val AP {:.4f}, Val macro_F1 {:.4F}".format(
                best_loss, best_val.auc, best_val.fpr95, best_val.ap, best_val.macro_F1))
            # with torch.no_grad():
            model.eval()
            model.load_state_dict(best_model_weight)
            test_loss, test_res, _ = self.get_contrastive_divergence(
                model, edge_predictor, test_mask_trial, features, labels, args, g, best_threshold)
            print("Test Loss {:.4f}, Test Auc {:.4f}, Test FPR@95 {:.4f}, Test AP {:.4f}, Test macro_F1 {:.4F}".format(
                test_loss, test_res.auc, test_res.fpr95, test_res.ap, test_res.macro_F1))
            torch.save(model, 'model.pt')
            val_results.append(best_val)
            test_results.append(test_res)
            if best_test_res is None or test_res.auc > best_test_res.auc:
                best_test_res = test_res
        if args.wandb:
            wandb_summary(best_test_res)
        print_results(val_results, test_results)
        return test_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--w_decay', type=float, default=0.001,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--dropout_adj', type=float, default=0.25,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayers', type=int, default=2, help='#layers')
    parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
    parser.add_argument('--ntrials', type=int, default=5, help='Number of trials')
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--non_linearity', choices=["gelu", "prelu", "relu", "elu"], default='elu')
    parser.add_argument('--normalization', type=str, default='sym')
    parser.add_argument('--attack', type=str, default="none",
                        choices=['fraud_edge_drop', 'edge_drop', 'mask_attribute', 'gaussian', 'none'])
    parser.add_argument('--attack_ratio', type=float, default=0.3,
                        help='attack ratio to remove')
    parser.add_argument('--sparse', type=int, default=1)

    # ebm
    parser.add_argument('--margin_in', type=float, default=1.0, help='threshold for normal nodes')
    parser.add_argument('--margin_out', type=float, default=5.0, help='threshold for fraud nodes')
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--nsteps', type=int, default=20)
    parser.add_argument('--sgld_lr', type=float, default=1)
    parser.add_argument('--sgld_std', type=float, default=0.01)
    parser.add_argument('--buffer_rate', type=float, default=0.8)
    parser.add_argument('--epoch_ebm', type=int, default=500)
    parser.add_argument('--train_mode', choices=['unsupervised', 'shot'], default='unsupervised')
    parser.add_argument('--shot', type=int, default=100, help='k-shot ood samples for ood detection')
    parser.add_argument('--shot_mode', type=str, default='OE', help='See choices',
                        choices=['CE', 'pronet', 'ebm_gnn', 'OE', 'Mahalanobis', 'ebm_gnn'])

    parser.add_argument('--dataset', type=str, default='yelp', help='See choices',
                        choices=['amazon', 'yelp', 'reddit', 'wikics', 'squirrel'])
    parser.add_argument('--mode', type=str, default="ebm_gnn", help='See choices',
                        choices=['single', 'ebm_gnn', 'sklearn', 'pygod', 'trans', 'entropy', 'oodgat'])
    parser.add_argument('--model', type=str, default="gpr_ebm", help='See choices',
                        choices=['gcn', 'gat', 'gpr', 'appnp', 'gin', 'gpr_att', 'bwgnn', 'gpr_ebm'])
    parser.add_argument('--pygod_model', type=str, default='DOMINANT', help='See choices',
                        choices=['ANOMALOUS', 'CoLA', 'CONAD', 'DOMINANT', 'GCNAE', 'MLPAE', 'GAAN'])
    parser.add_argument('--trans_model', type=str, default="goad",
                        choices=['goad', 'neutralAD'])
    parser.add_argument('--sklearn_model', type=str, default="svm",
                        choices=['svm', 'IsolationForest', 'LocalOutlier'])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--wandb', action='store_true', default=False, help='if use wandb to log experiment results')
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project='energy_ood',
                   tags=[f'{args.model}', f'{args.dataset}'],
                   entity='gztql',
                   config=vars(args))

    print(f'dataset: {args.dataset}, train_mode: {args.train_mode}, shot: {args.shot} mode: {args.mode}, pygod_model: {args.pygod_model}, trans_model: {args.trans_model}, sklearn_model: {args.sklearn_model}')

    print(args)

    experiment = Experiment(args)

    if args.train_mode == 'unsupervised':
        if args.mode == "single":
            experiment.train_single_gnn(args)
        elif args.mode == 'ebm_gnn':
            experiment.train_ebm_gnn(args)
        elif args.mode == 'sklearn':
            experiment.train_sklearn(args)
        elif args.mode == 'pygod':
            experiment.train_pygod(args)
        elif args.mode == 'trans':
            experiment.train_trans(args)
        elif args.mode == 'entropy':
            experiment.train_entropy(args)
        elif args.mode == 'oodgat':
            experiment.train_oodgat(args)
        else:
            raise NotImplementedError

    else:
        if args.shot_mode == 'CE':
            experiment.kshot_cross_entropy(args)
        elif args.shot_mode == 'OE':
            experiment.kshot_outlier_exposure(args)
        elif args.shot_mode == 'pronet':
            experiment.kshot_pronet(args)
        elif args.shot_mode == 'ebm_gnn':
            experiment.train_ebm_gnn(args)
        else:
            raise NotImplementedError

    if args.wandb:
        wandb.finish()
