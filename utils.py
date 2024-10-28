import torch
import _pickle as cPickle
import networkx as nx
import numpy as np
import scipy.sparse as sp
import gzip
import csv
import pandas as pd
import os
import re
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict as dd, OrderedDict
import kdtree
from haversine import haversine
import matplotlib.pyplot as plt


def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" % (len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    print("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: %.2f"% acc_at_161)

    metirc = {"Mean": int(np.mean(distances)),
              "Median": int(np.median(distances)),
              "Acc@161": int(acc_at_161)}

    return metirc
    # print("Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(
    #         int(acc_at_161)))
    # return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def efficient_collaboration_weighted_projected_graph2(B, nodes):
    nodes = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    all_nodes = set(B.nodes())
    i = 0
    tenpercent = len(all_nodes) / 10
    for m in all_nodes:
        if i % tenpercent == 0:
            logging.info(str(10 * i / tenpercent) + "%")
        i += 1

        nbrs = B[m]
        target_nbrs = [t for t in nbrs if t in nodes]  # 目标邻居的条件是在nodes里
        if m in nodes:  # 一个在nodes，一个在邻居节点
            for n in target_nbrs:
                if m < n:
                    if not G.has_edge(m, n):
                        G.add_edge(m, n)
        for n1 in target_nbrs:  # 都在邻居节点
            for n2 in target_nbrs:
                if n1 < n2:
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2)
    return G


class DataLoader():
    def __init__(self, data_home='../data/cmu/', bucket_size=50, encoding='latin1',
                 celebrity_threshold=5, one_hot_labels=False, mindf=10, maxdf=0.2,
                 norm='l2', idf=True, btf=True, smooth_idf=True, tokenizer=None, subtf=False, stops='english',
                 token_pattern=r'(?u)(?<![@])#?\b\w\w+\b', vocab=None):
        self.data_home = data_home  # 原始文件地址
        self.bucket_size = bucket_size  # kd-tree桶
        self.encoding = encoding  # 编码 latin1
        self.celebrity_threshold = celebrity_threshold  # 名人节点阈值
        self.one_hot_labels = one_hot_labels
        self.mindf = mindf
        self.maxdf = maxdf
        self.norm = norm
        self.idf = idf
        self.btf = btf
        self.smooth_idf = smooth_idf
        self.tokenizer = tokenizer
        self.subtf = subtf
        self.stops = stops
        self.token_pattern = token_pattern
        self.vocab = vocab
        self.biggraph = None

    """读文件，切分为四列user，lat，lon，text，小写的去重后的用户名作为索引，并按照用户名排序"""
    def load_data(self):
        logging.info('loading the dataset from %s' % self.data_home)
        train_file = os.path.join(self.data_home, 'user_info.train.gz')
        dev_file = os.path.join(self.data_home, 'user_info.dev.gz')
        test_file = os.path.join(self.data_home, 'user_info.test.gz')

        # 文本中有英文双引号时，直接用pd.read_csv会导致行数减少，应设置quoting=3或者quoting=csv.QUOTE_NONE
        df_train = pd.read_csv(train_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                               quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_dev = pd.read_csv(dev_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                             quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_test = pd.read_csv(test_file, delimiter='\t', encoding=self.encoding, names=['user', 'lat', 'lon', 'text'],
                              quoting=csv.QUOTE_NONE, error_bad_lines=False)
        df_train.dropna(inplace=True)  # dropna()丢弃含空值的行、列，inplace=True表示原地替换，返回值为None
        df_dev.dropna(inplace=True)
        df_test.dropna(inplace=True)
        df_train['user'] = df_train['user'].apply(lambda x: str(x).lower())  # apply()遍历
        df_train.drop_duplicates(['user'], inplace=True, keep='last')  # drop_duplicates()去重，'last'保留最后一个
        df_train.set_index(['user'], drop=True, append=False, inplace=True)  # 将user设置为新索引，drop删除user列
        df_train.sort_index(inplace=True)  # 排序
        df_dev['user'] = df_dev['user'].apply(lambda x: str(x).lower())
        df_dev.drop_duplicates(['user'], inplace=True, keep='last')
        df_dev.set_index(['user'], drop=True, append=False, inplace=True)
        df_dev.sort_index(inplace=True)
        df_test['user'] = df_test['user'].apply(lambda x: str(x).lower())
        df_test.drop_duplicates(['user'], inplace=True, keep='last')
        df_test.set_index(['user'], drop=True, append=False, inplace=True)
        df_test.sort_index(inplace=True)
        self.df_train = df_train
        self.df_dev = df_dev
        self.df_test = df_test


    def get_graph(self):
        g = nx.Graph()
        nodes = set(self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist())
        assert len(nodes) == len(self.df_train) + len(self.df_dev) + len(self.df_test), 'duplicate target node'
        nodes_list = self.df_train.index.tolist() + self.df_dev.index.tolist() + self.df_test.index.tolist()  # user_……
        node_id = {node: id for id, node in enumerate(nodes_list)}  # 字典{node: id}  node为user_……，id为数字
        g.add_nodes_from(node_id.values())  # 点
        for node in nodes:
            g.add_edge(node_id[node], node_id[node])  # 边
        pattern = '(?<=^|(?<=[^a-zA-Z0-9-_\\.]))@([A-Za-z]+[A-Za-z0-9_]+)'
        pattern = re.compile(pattern)
        logging.info('adding the train graph')
        for i in range(len(self.df_train)):  # 对每一个用户
            user = self.df_train.index[i]  # user_……
            user_id = node_id[user]  # id数字
            mentions = [m.lower() for m in pattern.findall(self.df_train.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)  # 加点
            for id in idmentions:
                g.add_edge(id, user_id)  # 给当前user_id加边
        logging.info('adding the dev graph')
        for i in range(len(self.df_dev)):
            user = self.df_dev.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_dev.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:  # node_id已经算过train里的
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        logging.info('adding the test graph')
        for i in range(len(self.df_test)):
            user = self.df_test.index[i]
            user_id = node_id[user]
            mentions = [m.lower() for m in pattern.findall(self.df_test.text[i])]
            idmentions = set()
            for m in mentions:
                if m in node_id:
                    idmentions.add(node_id[m])
                else:
                    id = len(node_id)
                    node_id[m] = id
                    idmentions.add(id)
            if len(idmentions) > 0:
                g.add_nodes_from(idmentions)
            for id in idmentions:
                g.add_edge(id, user_id)
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(g), nx.number_of_edges(g)))
        # 移除名人节点前 g：#nodes: 128269, #edges: 193185
        celebrities = []
        for i in range(len(nodes_list), len(node_id)):
            deg = len(g[i])
            if deg == 1 or deg > self.celebrity_threshold:
                celebrities.append(i)
        logging.info(
            'removing %d celebrity nodes with degree higher than %d' % (len(celebrities), self.celebrity_threshold))
        g.remove_nodes_from(celebrities)
        # （阈值为5）移除名人节点后g：#nodes: 36792, #edges: 86603
        # （阈值为10）移除名人节点后g：#nodes: 38150, #edges: 96153
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(g), nx.number_of_edges(g)))
        self.biggraph = g  # 用户跟他@的人
        logging.info('projecting the graph')
        projected_g = efficient_collaboration_weighted_projected_graph2(g, range(len(nodes_list)))
        logging.info('#nodes: %d, #edges: %d' % (nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        print('#nodes: %d, #edges: %d' % (nx.number_of_nodes(projected_g), nx.number_of_edges(projected_g)))
        # （阈值为5）用户-用户图projected_g：#nodes: 9475, #edges: 77155
        # （阈值为10）用户-用户图projected_g：#nodes: 9475, #edges: 100988
        self.graph = projected_g  # 用户和用户

    def tfidf(self):
        # keep both hashtags and mentions
        # token_pattern=r'(?u)@?#?\b\w\w+\b'
        # remove hashtags and mentions
        # token_pattern = r'(?u)(?<![#@])\b\w+\b'
        # just remove mentions and remove hashsign from hashtags
        # token_pattern = r'(?u)(?<![@])\b\w+\b'
        # remove mentions but keep hashtags with their sign √
        # token_pattern = r'(?u)(?<![@])#?\b\w\w+\b'
        # remove multple occurrences of a character after 2 times yesss => yess
        # re.sub(r"(.)\1+", r"\1\1", s)
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, token_pattern=self.token_pattern, use_idf=self.idf,
                                          norm=self.norm, binary=self.btf, sublinear_tf=self.subtf,
                                          min_df=self.mindf, max_df=self.maxdf, smooth_idf=self.smooth_idf,
                                          ngram_range=(1, 1), stop_words=self.stops,
                                          vocabulary=self.vocab, encoding=self.encoding, dtype='float32')
        # token_pattern:(?u)\b\w\w+\b：(?u)匹配对大小写不敏感，前后两个\b可以理解为空格，\w匹配一个字母或数字或下划线或汉字，\w+匹配一个或多个……
        # celebrity_threshold=5, one_hot_labels=False, mindf=10, maxdf=0.2, norm='l2'
        logging.info(self.vectorizer)
        # print(self.df_train.text.values[0:2])
        self.X_train = self.vectorizer.fit_transform(self.df_train.text.values)
        # print(self.vectorizer.get_feature_names())
        self.X_dev = self.vectorizer.transform(self.df_dev.text.values)
        self.X_test = self.vectorizer.transform(self.df_test.text.values)
        logging.info("training    n_samples: %d, n_features: %d" % self.X_train.shape)
        logging.info("development n_samples: %d, n_features: %d" % self.X_dev.shape)
        logging.info("test        n_samples: %d, n_features: %d" % self.X_test.shape)
        # vocab = self.vectorizer.vocabulary_
        # print(vocab)
        # print('vocab:', len(vocab))
        # with open('../data/vocab.csv', 'w', encoding='latin1') as f:
        #     [f.write('{0},{1}\n'.format(key, value)) for key, value in vocab.items()]

    def assignClasses(self):
        clusterer = kdtree.KDTreeClustering(bucket_size=self.bucket_size)
        train_locs = self.df_train[['lat', 'lon']].values
        clusterer.fit(train_locs)
        clusters = clusterer.get_clusters()  # 5685个训练样本属于哪个聚类
        # print(clusters, len(clusters))  # [45 38 50 ... 86  5 76] 5685
        cluster_points = dd(list)  # 字典
        for i, cluster in enumerate(clusters):
            cluster_points[cluster].append(train_locs[i])
            # 比如45: [array([ 33.921464, -84.340911]), array([ 33.9482206, -84.254263 ]), ……
        logging.info('#labels: %d' % len(cluster_points))
        self.cluster_median = OrderedDict()
        for cluster in sorted(cluster_points):
            points = cluster_points[cluster]
            median_lat = np.median([p[0] for p in points])
            median_lon = np.median([p[1] for p in points])
            self.cluster_median[cluster] = (median_lat, median_lon)  # 由训练集的样本来算出每个聚类的中心
        # print(self.cluster_median.values())  # 128
        dev_locs = self.df_dev[['lat', 'lon']].values
        test_locs = self.df_test[['lat', 'lon']].values
        nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1, metric=haversine, n_jobs=4)
        nnbr.fit(np.array([v for v in self.cluster_median.values()]))
        self.dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]
        self.train_classes = clusters
        if self.one_hot_labels:
            num_labels = np.max(self.train_classes) + 1
            y_train = np.zeros((len(self.train_classes), num_labels), dtype=np.float32)
            y_train[np.arange(len(self.train_classes)), self.train_classes] = 1
            y_dev = np.zeros((len(self.dev_classes), num_labels), dtype=np.float32)
            y_dev[np.arange(len(self.dev_classes)), self.dev_classes] = 1
            y_test = np.zeros((len(self.test_classes), num_labels), dtype=np.float32)
            y_test[np.arange(len(self.test_classes)), self.test_classes] = 1
            self.train_classes = y_train
            self.dev_classes = y_dev
            self.test_classes = y_test


def load_coradata(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),         # edges_unordered->edges likes:[paper_id,neigbo_paper_id]->[id,neigbo_paper_id]
                     dtype=np.int32).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # print(type(adj),type(features))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    # features = torch.FloatTensor(np.array(features.todense()))
    features = sparse_mx_to_torch_sparse_tensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_geodata():

    # #完整数据
    data = load_obj('../data/cmu/dump.pkl')
    # our graph
    # data = load_obj('/home/yanqilong/workspace/GCN-geo/data/cmu-dump-geograph-Tfidf.pkl')
    # data = load_obj('/sdc/yanqilong/workspace/Home-Computer/geographconv-master_edit/data/na/dump.pkl')

    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    # A归一化之后的邻接矩阵，X为TFIDF矩阵，Y为标签，U为索引（用户名称），classLatMedian、classLonMedian为train聚类中心坐标，userLocation包括用户和经纬度
    idx_train = range(0, X_train.shape[0])  # 数字
    idx_val = range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])
    idx_test = range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])
    # print('A:', A, A.shape)
    # print('X_train:', X_train, X_train.shape)
    # print('X_dev:', X_dev, X_dev.shape)
    # print('X_test:', X_test, X_test.shape)
    # print('Y_train:', Y_train, len(Y_train))
    # print('Y_dev:', Y_dev, len(Y_dev))
    # print('Y_test:', Y_test, len(Y_test))
    # X = sp.vstack([X_train, X_dev, X_test])
    # X_train = X_train.astype('float16')
    # X_dev = X_dev.astype('float16')
    # X_test = X_test.astype('float16')
    X_train = X_train.todense().A  # todense()稀疏矩阵变为一般矩阵
    X_dev = X_dev.todense().A
    X_test = X_test.todense().A
    A = A.todense().A
    X = np.vstack([X_train, X_dev, X_test])  # 垂直方向构成新数组
    Y = np.hstack((Y_train, Y_dev, Y_test))  # 水平方向构成新数组
    Y = Y.astype('int32')
    X = X.astype('float32')
    A = A.astype('float32')

    adj = torch.FloatTensor(A)
    # adj = sparse_mx_to_torch_sparse_tensor(A)
    # features = sparse_mx_to_torch_sparse_tensor(X)
    # train_features = sparse_mx_to_torch_sparse_tensor(X_train)
    features = torch.FloatTensor(X)
    train_features = torch.FloatTensor(X_train)
    labels = torch.LongTensor(Y)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print(X.shape)
    return adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, train_features


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dump_obj(obj, filename, protocol=-1, serializer=cPickle):
    with gzip.open(filename, 'wb') as fout:
        serializer.dump(obj, fout, protocol)


def load_obj(filename, serializer=cPickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin, encoding='iso-8859-1')  # encoding='iso-8859-1'
    return obj


dl = DataLoader()
dl.load_data()
dl.tfidf()
