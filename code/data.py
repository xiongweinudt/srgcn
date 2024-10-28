# @Time : 2023/4/20 22:09
# @Author : Jiayuan Gao
# @File : data.py
# @Software : PyCharm
import os
import logging
from utils import DataLoader, dump_obj, load_obj
import networkx as nx
import scipy as sp
import numpy as np
from my_parser import parse_args
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
    dump_file = os.path.join(data_home, 'dump.pkl')
    if os.path.exists(dump_file) and not model_args.builddata:
        logging.info('loading data from dumped file...')
        data = load_obj(dump_file)
        logging.info('loading data finished!')
        return data

    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding,
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf,
                    token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    dl.lda()
    dl.get_graph()


    vocab = dl.vectorizer.vocabulary_
    logging.info('saving vocab in {}'.format(vocab_file))
    dump_obj(vocab, vocab_file)
    logging.info('vocab dumped successfully!')
    U_test = dl.df_test.index.tolist()  # index是用户
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()

    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    adj.setdiag(0)
    # selfloop_value = np.asarray(adj.sum(axis=1)).reshape(-1,)
    selfloop_value = 1
    adj.setdiag(selfloop_value)  # 加上单位对角矩阵 A~=A+I -> adj
    n, m = adj.shape
    diags = adj.sum(axis=1).flatten()  # 行相加，变成只有一行的矩阵 -> diags
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0 / sp.sqrt(diags)  # diags^(-1/2)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0  # 溢出部分赋值为0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')  # [0]表示把diags_sqrt写在主对角线 -> 对角矩阵D~^(-1/2)
    # with np.errstate(divide='ignore'):
    #     diags_sqrt = 1.0 / np.lib.scimath.sqrt(diags)  # diags^(-1/2)
    # diags_sqrt[np.isinf(diags_sqrt)] = 0  # 溢出部分赋值为0
    # D_pow_neghalf = np.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')  # [0]表示把diags_sqrt写在主对角线 -> 对角矩阵D~^(-1/2)
    A = D_pow_neghalf * adj * D_pow_neghalf  # D~^(-1/2) * A~ * D~^(-1/2)
    A = A.astype(dtype)
    logging.info('adjacency matrix created.')

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    # lda_train = dl.lda_train
    # lda_dev = dl.lda_dev
    # lda_test = dl.lda_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
    classLatMedian = {str(c): dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c): dl.cluster_median[c][1] for c in dl.cluster_median}

    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]

    # data = (A, X_train, lda_train, Y_train, X_dev, lda_dev, Y_dev, X_test, lda_test, Y_test, U_train, U_dev, U_test,
    #         classLatMedian, classLonMedian, userLocation)
    data = (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian,
            userLocation)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data


if __name__ == '__main__':
    args = parse_args()
    model_args = args
    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket,
                           mindf=args.mindf, maxdf=args.maxdf)
    # vocab = load_obj("../data/cmu/vocab.pkl")
    # # 保存文件
    # with open('../data/cmu/my_file_1.csv', 'w', encoding='latin1') as f:
    #     [f.write('{0},{1}\n'.format(key, value)) for key, value in vocab.items()]
    #
    # # 词云
    # # fit_word函数，接受字典类型，其他类型会报错
    # wordcloud = WordCloud(font_path='times.ttf', background_color="white", width=4000, height=2000,
    #                       margin=10, max_words=1000).fit_words(vocab)
    # plt.axis('off')
    # plt.imshow(wordcloud)
    # # 显示
    # plt.show()
    # wordcloud.to_file('../data/cmu/word_cloud.png')

