from __future__ import division
import time
import os
import argparse
import numpy as np
import joblib
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
import logging
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from utils import load_coradata, accuracy, load_geodata, geo_eval
from models import GCN
from torch.utils.data import TensorDataset, DataLoader
from my_parser import parse_args

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Training settings
# device = torch.device("cuda:0")
device_ids = [0]

args = parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

bestAccVal = 0
lastEpoch = 0
Minloss_val = 100.0
Minloss_train = 100.0

if __name__ == '__main__':
    t_total = time.time()
    '''-----------------------Load Data--------------------'''
    adj, features, labels, idx_train, idx_val, idx_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation, train_features = load_geodata()

    '''-----------------------build model'--------------------'''
    model = GCN(nfeat=features.shape[1],  # shape[1]列
                nhid1=args.hidden,
                nhid2=args.hidden,
                nhid3=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    if torch.cuda.is_available():
        print('cuda Train')
        model.cuda(device=device_ids[0])
        features = features.cuda(device=device_ids[0])
        adj = adj.cuda(device=device_ids[0])
        labels = labels.cuda(device=device_ids[0])
        idx_train = idx_train.cuda(device=device_ids[0])
        idx_val = idx_val.cuda(device=device_ids[0])
        idx_test = idx_test.cuda(device=device_ids[0])
    '''划分数据集'''
    train_dataset = TensorDataset(idx_train, labels[idx_train])
    val_dataset = TensorDataset(idx_val, labels[idx_val])
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=512,
        shuffle=False)  # batch_size=512
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=512,
        shuffle=False)  # batch_size=512
    '''--------------------Training-----------------------'''

    # for epoch in range(args.epochs):
    #     t = time.time()
    #     model.train()
    #     optimizer.zero_grad()
    #     output = model(features, adj)
    #
    #     regularization_loss1 = 0
    #     for param in model.parameters():
    #         regularization_loss1 += torch.sum(abs(param))
    #
    #     loss_train = F.nll_loss(output[idx_train], labels[idx_train]) + regularization_loss1 * args.weight_decay
    #     acc_train = accuracy(output[idx_train], labels[idx_train])
    #     loss_train.backward()
    #     optimizer.step()
    #
    #     # for batch_index, batch_y in train_loader:
    #     #
    #     #     model.train()
    #     #     optimizer.zero_grad()
    #     #     output = model(features, adj)
    #     #     regularization_loss1 = 0
    #     #     for param in model.parameters():
    #     #         regularization_loss1 += torch.sum(abs(param))
    #     #     loss_train = F.nll_loss(output[batch_index], batch_y) + regularization_loss1 * args.weight_decay
    #     #     acc_train = accuracy(output[batch_index], batch_y)
    #     #     loss_train.backward()
    #     #     optimizer.step()
    #
    #     model.eval()
    #     output = model(features, adj)
    #     # loss_val = 0
    #     correct = 0
    #     preds = torch.zeros(64)
    #
    #     # for i,(batch_index, batch_y) in  enumerate(val_loader):
    #     #     loss_val += F.nll_loss(output[batch_index], batch_y)
    #     #     pred =  output[batch_index].max(1)[1].type_as(batch_y)
    #     #     correct += pred.eq(batch_y).sum()
    #     #     if i==0:
    #     #         preds = pred
    #     #     else:
    #     #         preds =torch.cat((preds,pred),dim=0)
    #     #
    #     # loss_val = loss_val.item()/(len(val_loader))
    #     # acc_val = correct.item()/(len(val_loader.dataset))
    #     # y_pred = preds.cpu().numpy().flatten()
    #     loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #     acc_val = accuracy(output[idx_val], labels[idx_val])
    #     y_pred = output[idx_val].data.max(1, keepdim=True)[1].cpu().numpy().flatten()
    #
    #     metric = geo_eval(None, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
    #     # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    #     # acc_train = accuracy(output[idx_train], labels[idx_train])
    #     print('Epoch: {:04d}'.format(epoch + 1),
    #           'loss_train: {:.4f}'.format(loss_train.item()),
    #           'acc_train: {:.4f}'.format(acc_train.item()),
    #           'loss_val: {:.4f}'.format(loss_val),
    #           'acc_val: {:.4f}'.format(acc_val),
    #           'time: {:.4f}s'.format(time.time() - t))
    #
    #     '''Early Stopping'''
    #
    #     if metric['Acc@161'] - bestAccVal > 0 or loss_val - Minloss_val < 0:
    #         if metric['Acc@161'] - bestAccVal > 0:
    #             bestAccVal = metric['Acc@161']
    #             Minloss_val = loss_val
    #             lastEpoch = epoch
    #             torch.save({'model_state_dict': model.state_dict(),
    #                         'epoch': epoch},
    #                        './params.pth')
    #         elif loss_val - Minloss_val < 0:
    #             # Minloss_val = metric['Acc@161']
    #             bestAccVal = metric['Acc@161']
    #             Minloss_val = loss_val
    #             lastEpoch = epoch
    #             torch.save({'model_state_dict': model.state_dict(),
    #                         'epoch': epoch},
    #                        './params.pth')
    #     else:
    #         if epoch - lastEpoch > args.early_stopping_rounds:
    #             print(
    #                 'No improved!Now epoch= {},lastEpoch= {},bestAccVal= {}, bestloss_val= {}'.format(epoch, lastEpoch,
    #                                                                                                   bestAccVal,
    #                                                                                                   Minloss_val))
    #             logging.info('No improved!')
    #             break
    #
    #     torch.cuda.empty_cache()
    #     del loss_train, loss_val, acc_val, acc_train, y_pred
    #
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    '''--------------------Testing------------------------'''

    '''Load Best Model'''
    checkpoint = torch.load('./params.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    model.to(torch.device("cpu"))
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()),
          "epoch= {}".format(epoch+1),
          )
    print('dev results:')
    y_pred = output[idx_val].data.max(1, keepdim=True)[1].cpu().numpy().flatten()
    geo_eval(None, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
    print('test results:')
    y_pred = output[idx_test].data.max(1, keepdim=True)[1].cpu().numpy().flatten()
    geo_eval(None, y_pred, U_test, classLatMedian, classLonMedian, userLocation)

    # # 混淆矩阵
    # con_mat = confusion_matrix(labels[idx_test].detach().numpy().astype(str), y_pred.astype(str))
    # # print(con_mat)
    # # classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # # classes.sort()
    # plt.imshow(con_mat, cmap=plt.cm.Blues)
    # indices = range(len(classes))
    # plt.xticks(indices, classes)
    # plt.yticks(indices, classes)
    # plt.colorbar()
    # plt.xlabel('pred')
    # plt.ylabel('true')
    # for first_index, first_value in enumerate(classes):
    #     for second_index, second_value in enumerate(classes):
    #         plt.text(first_index, second_index, con_mat[second_value][first_value], va='center', ha='center')
    # plt.figure(figsize=(6, 6))
    # # plt.savefig(os.path.join('./matrix.png'))
    # plt.show()

    # 可视化
    ts = TSNE(n_components=2)  # 二维
    # ts.fit_transform(output.to('cpu').detach().numpy())
    ts.fit_transform(features.to('cpu').detach().numpy())
    # joblib.dump(ts, './tsne.pkl')
    # ts = joblib.load('./tsne.pkl')
    x = ts.embedding_
    y = labels.to('cpu').detach().numpy()
    xi = []
    for i in [10, 33, 40, 59, 100]:  # best[10, 33, 40, 60, 78, 100, 117] CH[10, 33, 40, 59, 100]
        xi.append(x[np.where(y == i)])
    colors = ['lightgreen', 'turquoise', 'orange', 'y', 'cornflowerblue']
    # ['red', 'orange', 'olive', 'lightgreen', 'turquoise', 'royalblue', 'violet']
    # CH ['lightgreen', 'turquoise', 'orange', 'y', 'cornflowerblue']
    labels = ['11', '34', '41', '60', '101']
    # ['11', '34', '41', '61', '79', '101', '118']
    # ['11', '34', '41', '60', '101']
    plt.figure(figsize=(6, 6))
    for i in range(5):
        plt.scatter(xi[i][:, 0], xi[i][:, 1], s=30, color=colors[i], marker='.', alpha=1)
    plt.legend(labels=labels, loc='best', title="classes", prop={'size': 10})
    plt.xlabel("Dim1")
    plt.ylabel("Dim2")
    plt.savefig(os.path.join('./c.pdf'))
    plt.show()

