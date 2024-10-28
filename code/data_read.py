# @Time : 2023/4/6 9:11
# @Author : Jiayuan Gao
# @File : data_read.py
# @Software : PyCharm
import time
import datetime
import numpy as np
import pandas as pd
import joblib
import pickle
from utils import load_obj
from utils import load_coradata, accuracy,load_geodata,geo_eval


# 读取npy、pkl文件
time_start = time.time()
# np.set_printoptions(threshold=np.inf)  # 显示出省略号里的内容
# test_1 = load_obj("../data/cmu/dump.pkl")
test_2 = load_obj("../data/cmu/vocab.pkl")
# test_3 = joblib.load(open("./gcn_1.0_percent_pred_32.pkl", 'rb'))
# adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, train_features = load_geodata()
# print('adj:', adj)
# print(adj.shape)
# print('features:', features)
# print(features.shape)
# print('lables:', labels)
# print(labels.shape)
# print('id_train:', idx_train)
# print('id_val:', idx_val)
# print('id_test:', idx_test)
# print('U_train:', U_train)
# print('U_dev:', U_dev)
# print('U_test:', U_test)
# print('classLatMedian:', classLatMedian)
# print('classLonMedian:', classLonMedian)
# print('userLocation:', userLocation)
# print('train_features:', train_features)

# print(test_1)  #用户ID以及用户的地理位置
print(test_2)  # 词频
time_end = time.time()
time_sum = time_end - time_start
print("%f s" % time_sum)
