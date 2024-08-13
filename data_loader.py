# Copyright (c) 2020-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import warnings

import os
import dgl
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
from utils import normalize_features, attack

from dgl.data import FraudYelpDataset, FraudAmazonDataset, WikiCSDataset, SquirrelDataset
from dgl.data.utils import load_graphs, save_graphs
from pygod.utils import load_data as load_pygod_data
import pandas as pd
from collections import Counter
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.cluster import KMeans


warnings.simplefilter("ignore")


def pseudo_class_label(feat, nclass=10):
    x = feat.numpy()
    kmeans = KMeans(n_clusters=nclass, random_state=0, n_init='auto').fit(x)
    return torch.from_numpy(kmeans.labels_).long().to(feat.device)


def load_data(args, class_label=False):
    dataset_str = args.dataset

    if class_label:
        assert dataset_str in ['squirrel', 'wikics']

    if dataset_str == 'yelp':
        dataset = FraudYelpDataset()
        graph = dataset[0]

        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials, mode=args.train_mode, k=args.shot)

        x_data = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=False), dtype=torch.float)
        graph.ndata['class'] = pseudo_class_label(graph.ndata['feature'])

        return x_data, graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph
        # graph.ndata['train_mask'].bool(), graph.ndata['val_mask'].bool(), graph.ndata['test_mask'].bool()

    elif dataset_str == 'amazon':
        dataset = FraudAmazonDataset()
        graph = dataset[0]

        graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials, mode=args.train_mode, k=args.shot)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
                                              dtype=torch.float)
        graph.ndata['class'] = pseudo_class_label(graph.ndata['feature'])

        if args.attack != "none":
            graph = attack(graph, args)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    elif dataset_str == 'reddit' or dataset_str == 'weibo':
        data = load_pygod_data(dataset_str)
        graph = dgl.graph((data.edge_index[0], data.edge_index[1]))
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        graph.ndata['feature'] = data.x
        graph.ndata['label'] = data.y.type(torch.LongTensor)

        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials, mode=args.train_mode)

        graph.ndata['feature'] = torch.tensor(normalize_features(graph.ndata['feature'], norm_row=True),
                                              dtype=torch.float)
        graph.ndata['class'] = pseudo_class_label(graph.ndata['feature'])

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    elif dataset_str == 'elliptic':
        graph = load_elliptic('dataset/elliptic/')
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials, mode=args.train_mode, k=args.shot)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    elif dataset_str == 'dgraph':
        graph = load_dgraph('dataset/dgraph/dgraphfin.npz')
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)
        train_mask, val_mask, test_mask = graph_split(dataset_str, graph.ndata['label'], train_ratio=args.train_ratio,
                                                      folds=args.ntrials, mode=args.train_mode, k=args.shot)

        return graph.ndata['feature'], graph.ndata['feature'].size()[-1], graph.ndata['label'], 2, \
            train_mask, val_mask, test_mask, graph

    elif dataset_str == 'ogbn-arxiv':
        graph = PygNodePropPredDataset(name='ogbn-arxiv')
        g = dgl.graph((graph.edge_index[0], graph.edge_index[1]))
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g.ndata['feature'] = torch.tensor(normalize_features(graph.x, norm_row=False), dtype=torch.float)
        g.ndata['year'] = graph.node_year
        g.ndata['label'] = graph.y
        train_mask, val_mask, test_mask, ood_label = load_arxiv_dataset(g, mode=args.train_mode, folds=args.ntrials)
        g.ndata['label'] = ood_label
        return g.ndata['feature'], g.ndata['feature'].size()[-1], g.ndata['label'], 2, \
            train_mask, val_mask, test_mask, g

    elif dataset_str == 'wikics':
        g = WikiCSDataset()[0]
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g.ndata['feature'] = g.ndata['feat']
        del g.ndata['feat']
        train_mask, val_mask, test_mask, ood_label, new_class_label = load_wikics_dataset(g, mode=args.train_mode, folds=args.ntrials,
                                                                                          k=args.shot)
        g.ndata['class'] = new_class_label
        # g.ndata['class'] = pseudo_class_label(g.ndata['feature'])
        g.ndata['label'] = ood_label

        return g.ndata['feature'], g.ndata['feature'].size()[-1], g.ndata['label'], 2, \
            train_mask, val_mask, test_mask, g

    elif dataset_str == 'squirrel':
        g = SquirrelDataset()[0]
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        g.ndata['feature'] = g.ndata['feat']
        del g.ndata['feat']
        train_mask, val_mask, test_mask, ood_label, new_class_label = load_wikics_dataset(g, mode=args.train_mode, folds=args.ntrials,
                                                                         k=args.shot, ood_class=[1])
        g.ndata['class'] = new_class_label
        # g.ndata['class'] = pseudo_class_label(g.ndata['feature'])
        g.ndata['label'] = ood_label

        return g.ndata['feature'], g.ndata['feature'].size()[-1], g.ndata['label'], 2, \
            train_mask, val_mask, test_mask, g

    else:
        raise NotImplementedError


def load_dgraph(path):
    """
    0: 1210092    normal
    1: 15509    fraud
    2: 1620851    background
    3: 854098   background
    """
    dgraph_data = np.load(path)
    edge_index = torch.IntTensor(dgraph_data['edge_index']).transpose(0, 1)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g.ndata['feature'] = torch.FloatTensor(dgraph_data['x'])
    g.ndata['label'] = torch.tensor(dgraph_data['y'])
    return g


def load_elliptic(path):
    """
    # fraud nodes 4545
    # normal nodes 42019
    # unknown nodes 157205
    """

    if os.path.exists(path + 'data.pt'):
        g = load_graphs(path + 'data.pt')[0][0]
        return g

    raw_id2new_id = {}
    id_class = pd.read_csv(path + 'elliptic_txs_classes.csv').values
    unknown_ids, normal_ids, fraud_ids = [], [], []
    for raw_id, node_class in id_class:
        if node_class == 'unknown':
            unknown_ids.append(raw_id)
        elif node_class == '1':
            fraud_ids.append(raw_id)
        elif node_class == '2':
            normal_ids.append(raw_id)
        else:
            raise NotImplementedError

    labels = [1 for _ in range(len(fraud_ids))] + [0 for _ in range(len(normal_ids))] + [-1 for _ in
                                                                                         range(len(unknown_ids))]
    raw_id2new_id.update({raw_id: i for i, raw_id in enumerate(fraud_ids)})
    raw_id2new_id.update({raw_id: (i + len(fraud_ids)) for i, raw_id in enumerate(normal_ids)})
    raw_id2new_id.update({raw_id: (i + len(fraud_ids) + len(normal_ids)) for i, raw_id in enumerate(unknown_ids)})

    raw_edge_index = pd.read_csv(path + 'elliptic_txs_edgelist.csv').values
    new_edge_index = [[], []]
    for raw_edge_index_u, raw_edge_index_v in raw_edge_index:
        new_edge_index[0].append(raw_id2new_id[raw_edge_index_u])
        new_edge_index[1].append(raw_id2new_id[raw_edge_index_v])

    features = pd.read_csv(path + 'elliptic_txs_features.csv', header=None).values
    for feature in features:
        feature[0] = raw_id2new_id[int(feature[0])]

    feature_ids = features[:, 0]
    features = features[:, 2:]
    features = features[np.argsort(feature_ids)]

    g = dgl.graph((torch.IntTensor(new_edge_index[0]), torch.IntTensor(new_edge_index[1])))
    g.ndata['feature'] = torch.FloatTensor(features)
    g.ndata['label'] = torch.tensor(labels)
    save_graphs(path + 'data.pt', [g])
    return g


def load_arxiv_dataset(g, time_bound=None, mode='unsupervised', train_ratio=0.4, folds=5):
    def build_node_mask(ind_node_ids, ind_node_mask, mode, test_ood_node_ids, train_ood_node_ids, train_ratio, fold):
        idx_train_ind, idx_test_ind = train_test_split(ind_node_ids,
                                                       train_size=train_ratio,
                                                       random_state=fold,
                                                       shuffle=True)
        if mode == 'shot':
            idx_train_ood, idx_test_ood = train_ood_node_ids, test_ood_node_ids
            idx_train = torch.cat((idx_train_ind, idx_train_ood))
            idx_test = torch.cat((idx_test_ind, idx_test_ood))

        else:
            idx_test = torch.cat((idx_test_ind, train_ood_node_ids, test_ood_node_ids))
            idx_train = idx_train_ind
        idx_valid, idx_test = train_test_split(idx_test,
                                               stratify=np.array(ind_node_mask)[idx_test],
                                               test_size=2.0 / 3,
                                               random_state=fold,
                                               shuffle=True)
        train_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        valid_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        test_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        train_mask_fold[idx_train] = True
        valid_mask_fold[idx_valid] = True
        test_mask_fold[idx_test] = True
        return train_mask_fold, valid_mask_fold, test_mask_fold


    if time_bound is None:
        time_bound = [2015, 2017]
    year = g.ndata['year']

    year_min, year_max = time_bound[0], time_bound[1]
    test_year_bound = [2017, 2018, 2019, 2020]

    ind_node_mask = (year <= year_min).squeeze(1)
    train_ood_node_mask = (year <= year_max).squeeze(1) * (year > year_min).squeeze(1)
    test_ood_node_mask = (year <= test_year_bound[-1]).squeeze(1) * (year > test_year_bound[0]).squeeze(1)

    ood_node_labels = train_ood_node_mask + test_ood_node_mask

    ind_node_ids = torch.nonzero(ind_node_mask)
    train_ood_node_ids = torch.nonzero(train_ood_node_mask)
    test_ood_node_ids = torch.nonzero(test_ood_node_mask)

    train_mask, valid_mask, test_mask = [], [], []
    for fold in range(folds):
        train_mask_fold, valid_mask_fold, test_mask_fold = build_node_mask(ind_node_ids, ind_node_mask, mode, test_ood_node_ids, train_ood_node_ids, train_ratio, fold)
        train_mask.append(train_mask_fold)
        valid_mask.append(valid_mask_fold)
        test_mask.append(test_mask_fold)

    train_mask = torch.vstack(train_mask)
    valid_mask = torch.vstack(valid_mask)
    test_mask = torch.vstack(test_mask)
    return train_mask, valid_mask, test_mask, ood_node_labels


def load_wikics_dataset(g, ood_class=None, mode='unsupervised', train_ratio=0.4, folds=5, k=10):
    def build_node_mask(ind_node_mask, ood_node_mask, mode, train_ratio, fold, k):
        ind_node_ids = torch.nonzero(ind_node_mask)
        ood_node_ids = torch.nonzero(ood_node_mask)

        idx_train_ind, idx_test_ind = train_test_split(ind_node_ids,
                                                       train_size=train_ratio,
                                                       random_state=fold,
                                                       shuffle=True)
        if mode == 'shot':
            idx_train_ood, idx_test_ood = train_test_split(ood_node_ids, train_size=k, random_state=fold, shuffle=True)
            idx_train = torch.cat((idx_train_ind, idx_train_ood))
            idx_test = torch.cat((idx_test_ind, idx_test_ood))

        else:
            idx_test = torch.cat((idx_test_ind, ood_node_ids))
            idx_train = idx_train_ind

        idx_valid, idx_test = train_test_split(idx_test,
                                               stratify=np.array(ind_node_mask)[idx_test],
                                               test_size=2.0 / 3,
                                               random_state=fold,
                                               shuffle=True)
        train_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        valid_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        test_mask_fold = torch.BoolTensor([False for _ in range(len(ind_node_mask))])
        train_mask_fold[idx_train] = True
        valid_mask_fold[idx_valid] = True
        test_mask_fold[idx_test] = True
        return train_mask_fold, valid_mask_fold, test_mask_fold

    if ood_class is None:
        ood_class = [4, 5]

    num_nodes = g.num_nodes()
    label_set = set(g.ndata['label'].numpy())
    old_label2new_label = {}
    for label in label_set:
        if label not in ood_class:
            old_label2new_label[label] = len(old_label2new_label)
    for label in ood_class:
        old_label2new_label[label] = len(old_label2new_label)
    new_class_label = torch.tensor([old_label2new_label[label] for label in g.ndata['label'].numpy()], dtype=torch.long)

    ind_node_mask = torch.BoolTensor([False for _ in range(num_nodes)])
    ood_node_mask = torch.BoolTensor([False for _ in range(num_nodes)])
    for label_id in label_set:
        if label_id in ood_class:
            ood_node_mask += (g.ndata['label'] == label_id)
        else:
            ind_node_mask += (g.ndata['label'] == label_id)

    ood_node_labels = ood_node_mask

    train_mask, valid_mask, test_mask = [], [], []
    for fold in range(folds):
        train_mask_fold, valid_mask_fold, test_mask_fold = build_node_mask(ind_node_mask, ood_node_mask, mode, train_ratio, fold, k=k)
        train_mask.append(train_mask_fold)
        valid_mask.append(valid_mask_fold)
        test_mask.append(test_mask_fold)

    train_mask = torch.vstack(train_mask)
    valid_mask = torch.vstack(valid_mask)
    test_mask = torch.vstack(test_mask)
    return train_mask, valid_mask, test_mask, ood_node_labels, new_class_label


def graph_split(dataset, labels, train_ratio=0.01, folds=5, mode='unsupervised', k=5):
    """split dataset into train and test

    Args:
        dataset (str): name of dataset
        labels (list): list of labels of nodes
        mode (str): 'unsupervised': only normal samples for ood detection
        'shot': only k fraud samples used for ood detection
    """
    assert dataset in ['amazon', 'yelp', 'elliptic', 'dgraph', 'reddit', 'weibo']
    if dataset == 'amazon':
        index = np.array(range(3305, len(labels)))
        stratify_labels = labels[3305:]
        ood_node_ids = torch.nonzero(labels == 1).squeeze(1)
        ind_node_ids = torch.nonzero(labels == 0).squeeze(1)
        ind_node_ids = ind_node_ids[ind_node_ids >= 3305]

    elif dataset == 'yelp' or dataset == 'weibo' or dataset == 'reddit':
        index = np.array(range(len(labels)))
        stratify_labels = labels
        ind_node_ids = torch.nonzero(labels == 0).squeeze(1)
        ood_node_ids = torch.nonzero(labels == 1).squeeze(1)

    elif dataset == 'elliptic':
        index = np.array(range(46564))
        stratify_labels = labels[:46564]
        ind_node_ids = torch.nonzero(labels == 0).squeeze(1)
        ood_node_ids = torch.nonzero(labels == 1).squeeze(1)

    elif dataset == 'dgraph':
        index = np.array(torch.cat((torch.nonzero(labels == 1).squeeze(1), torch.nonzero(labels == 0).squeeze(1))))
        stratify_labels = np.array(labels)[index]
        ind_node_ids = torch.nonzero(labels == 0).squeeze(1)
        ood_node_ids = torch.nonzero(labels == 1).squeeze(1)

    else:
        raise NotImplementedError

    # generate mask
    train_mask, valid_mask, test_mask = [], [], []

    for fold in range(folds):
        train_mask_fold, valid_mask_fold, test_mask_fold = build_mask_fold(labels, train_ratio, ind_node_ids,
                                                                           ood_node_ids, fold, mode, k)

        train_mask.append(train_mask_fold)
        valid_mask.append(valid_mask_fold)
        test_mask.append(test_mask_fold)

    train_mask = torch.vstack(train_mask)
    valid_mask = torch.vstack(valid_mask)
    test_mask = torch.vstack(test_mask)

    return train_mask, valid_mask, test_mask


def build_mask_fold(labels, train_ratio, ind_node_ids, ood_node_ids, fold, mode, k):
    idx_train_ind, idx_test_ind = train_test_split(ind_node_ids,
                                                   train_size=train_ratio,
                                                   random_state=fold,
                                                   shuffle=True)

    if mode == 'shot':
        idx_train_ood, idx_test_ood = train_test_split(ood_node_ids, train_size=k, random_state=fold, shuffle=True)
        idx_train = torch.cat((idx_train_ind, idx_train_ood))
        idx_test = torch.cat((idx_test_ind, idx_test_ood))

    else:
        idx_test = torch.cat((idx_test_ind, ood_node_ids))
        idx_train = idx_train_ind

    idx_valid, idx_test = train_test_split(idx_test,
                                           stratify=np.array(labels)[idx_test],
                                           test_size=2.0 / 3,
                                           random_state=fold,
                                           shuffle=True)

    train_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
    valid_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
    test_mask_fold = torch.BoolTensor([False for _ in range(len(labels))])
    train_mask_fold[idx_train] = True
    valid_mask_fold[idx_valid] = True
    test_mask_fold[idx_test] = True
    return train_mask_fold, valid_mask_fold, test_mask_fold
