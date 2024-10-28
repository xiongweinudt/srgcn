import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, Highway_dense
import torch
import numpy as np
from my_parser import parse_args
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from utils import *
from utils import accuracy, load_geodata
from torch.utils.data import TensorDataset, DataLoader

args = parse_args()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout):
        super(GCN, self).__init__()
        self.linear1 = nn.Linear(nfeat, nhid1)
        self.gc1 = GraphConvolution(nhid1, nhid2)
        self.gc2 = GraphConvolution(nhid2, nhid3)
        self.gc3 = GraphConvolution(nhid3, nclass)
        self.hw1 = Highway_dense(nhid1, nhid2)
        self.hw2 = Highway_dense(nhid2, nhid3)

        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.nclass = nclass
        self.dropout = dropout
        self.highway = True
        if self.highway == False:
            print('H-GCN model.....')
        else:
            print('GCN model.....')

    def forward(self, x, adj):
        """
        highway == True: tanh(linear1)->dropout->hw1->hw2->gc3->log_softmax
        highway == False: tanh(linear1)->dropout->tanh(gc1)->dropout->tanh(gc2)->dropout->gc3->log_softmax
        """
        linear = F.relu(self.linear1(x))
        linear = F.dropout(linear, self.dropout, training=self.training)
        if self.highway:
            conv1 = self.hw1(linear, adj)
            conv2 = self.hw2(conv1, adj)
            conv3 = self.gc3(conv2, adj)

        else:
            conv1 = F.relu(self.gc1(linear, adj))
            conv1 = F.dropout(conv1, self.dropout, training=self.training)
            conv2 = F.relu(self.gc2(conv1, adj))
            conv2 = F.dropout(conv2, self.dropout, training=self.training)
            conv3 = self.gc3(conv2, adj)

        return F.log_softmax(conv3, dim=1)


class GCN_pl(pl.LightningModule):
    def __init__(self, nfeat, nhid1, nhid2, nhid3, nclass, dropout):
        super(GCN_pl, self).__init__()
        self.linear = nn.Linear(nfeat, nhid1)
        self.gc1 = GraphConvolution(nhid1, nhid2)
        self.gc2 = GraphConvolution(nhid2, nhid3)
        self.gc3 = GraphConvolution(nhid3, nclass)
        self.hw1 = Highway_dense(nhid1, nhid2)
        self.hw2 = Highway_dense(nhid2, nhid3)
        self.linear1 = nn.Linear(nfeat, nhid1)
        self.linear2 = nn.Linear(nhid2, nhid3)
        self.linear3 = nn.Linear(nhid3, nclass)

        self.nfeat = nfeat
        self.nhid1 = nhid1
        self.nhid2 = nhid2
        self.nhid3 = nhid3
        self.nclass = nclass
        self.dropout = dropout
        self.highway = True
        if self.highway == True:
            print('H-GCN model.....')
        else:
            print('GCN model.....')

    def forward(self, x, adj):

        # conv1 = F.tanh(self.linear1(x))
        # conv1 = F.dropout(conv1, self.dropout, training=self.training)
        # conv2 = F.tanh(self.linear2(conv1))
        # conv2 = F.dropout(conv2, self.dropout, training=self.training)
        # conv3 = F.tanh(self.linear3(conv2))

        linear = F.tanh(self.linear(x))
        linear = F.dropout(linear, self.dropout, training=self.training)
        if self.highway:
            conv1 = self.hw1(linear, adj)
            conv2 = self.hw2(conv1, adj)
            conv3 = self.gc3(conv2, adj)

        else:
            conv1 = F.tanh(self.gc1(linear, adj))
            conv1 = F.dropout(conv1, self.dropout, training=self.training)
            conv2 = F.tanh(self.gc2(conv1, adj))
            conv2 = F.dropout(conv2, self.dropout, training=self.training)
            conv3 = self.gc3(conv2, adj)

        return F.log_softmax(conv3, dim=1)

    def prepare_data(self):
        self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test, self.U_train, self.U_dev, self.U_test, \
        self.classLatMedian, self.classLonMedian, self.userLocation, self.train_features = load_geodata()
        self.features = self.features.cuda()
        self.adj = self.adj.cuda()
        self.labels = self.labels.cuda()
        self.idx_train = self.idx_train.cuda()
        self.idx_val = self.idx_val.cuda()
        self.idx_test = self.idx_test.cuda()

    def train_dataloader(self):
        torch_dataset = TensorDataset(self.idx_train, self.labels[self.idx_train])
        self.train_loader = DataLoader(
            dataset=torch_dataset,
            batch_size=64,
            shuffle=True)

        return self.train_loader

    def val_dataloader(self):
        val_dataset = TensorDataset(self.idx_val, self.labels[self.idx_val])
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=64,
            shuffle=False)

        return self.val_loader

    def test_dataloader(self):
        test_dataset = TensorDataset(self.idx_test, self.labels[self.idx_test])
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=64,
            shuffle=False)
        return self.test_loader

    def training_step(self, batch, batch_idx):

        batch_index, batch_y = batch
        output = self.forward(self.features, self.adj)
        regularization_loss1 = 0
        for param in self.parameters():
            regularization_loss1 += torch.sum(abs(param))
        loss_train = F.nll_loss(output[batch_index], batch_y) + regularization_loss1 * args.weight_decay
        acc_train = accuracy(output[batch_index], batch_y)
        return {'loss': loss_train,
                'log': {'batch_loss': loss_train}
                }

    def training_epoch_end(self, outputs):
        output = self.forward(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        train_metrics = {'loss_train': loss_train, 'acc_train': acc_train}

        return {'log': train_metrics, 'progress_bar': train_metrics}

    def validation_step(self, batch, batch_idx):

        batch_index, batch_y = batch
        output = self.forward(self.features, self.adj)

        loss_val = F.nll_loss(output[batch_index], batch_y)
        pred = output[batch_index].max(1)[1].type_as(batch_y)
        correct = pred.eq(batch_y).sum()

        return {'loss_val': loss_val,
                'pred': pred,
                'correct': correct}

    def validation_epoch_end(self, validation_step_outputs):

        loss_val = torch.stack([x['loss_val'] for x in validation_step_outputs]).sum()
        correct = torch.stack([x['correct'] for x in validation_step_outputs]).sum()
        preds = torch.cat([x['pred'] for x in validation_step_outputs], dim=0)

        loss_val = loss_val.item() / (len(self.val_loader))
        acc_val = correct.item() / (len(self.val_loader.dataset))
        preds = preds.cpu().numpy().flatten()
        geo_metric = geo_eval(None, preds, self.U_dev, self.classLatMedian, self.classLonMedian, self.userLocation)
        val_metrics = {'loss_val': loss_val, 'acc_val': acc_val}
        val_metrics.update(geo_metric)

        return {'log': val_metrics,
                'progress_bar': val_metrics,  # print
                }

    def test_step(self, batch, batch_idx):

        batch_index, batch_y = batch
        output = self.forward(self.features, self.adj)

        loss_test = F.nll_loss(output[batch_index], batch_y)
        pred = output[batch_index].max(1)[1].type_as(batch_y)
        correct = pred.eq(batch_y).sum()

        return {'loss_test': loss_test,
                'pred': pred,
                'correct': correct}

    def test_epoch_end(self, validation_step_outputs):

        loss_test = torch.stack([x['loss_test'] for x in validation_step_outputs]).sum()
        correct = torch.stack([x['correct'] for x in validation_step_outputs]).sum()
        preds = torch.cat([x['pred'] for x in validation_step_outputs], dim=0)

        loss_test = loss_test.item() / (len(self.test_loader))
        acc_test = correct.item() / (len(self.test_loader.dataset))
        preds = preds.cpu().numpy().flatten()
        # output = self.forward(self.features, self.adj)
        # loss_test = F.nll_loss(output[self.idx_test], self.labels[self.idx_test])
        # acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        # preds = output[self.idx_test].data.max(1, keepdim=True)[1].cpu().numpy().flatten()
        geo_metric = geo_eval(None, preds, self.U_test, self.classLatMedian, self.classLonMedian, self.userLocation)
        test_metrics = {'loss_test': loss_test, 'acc_test': acc_test}
        test_metrics.update(geo_metric)

        return {'log': test_metrics,
                'progress_bar': test_metrics,  # print
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optimizer

    def get_progress_bar_dict(self):  # override v_num
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)

        return items
