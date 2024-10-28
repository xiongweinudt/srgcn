from __future__ import division
import time
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")
import logging
from utils import load_coradata, accuracy,load_geodata,geo_eval
from models import GCN_pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import ProgressBar
from torch.utils.data import TensorDataset,DataLoader
import pytorch_lightning as pl
from pytorch_lightning import  seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from my_parser import parse_args

args = parse_args()
seed_everything(42)




if __name__ == '__main__':
    t_total = time.time()
    '''-----------------------Load Data--------------------'''
    adj, features, labels, idx_train, idx_val, idx_test,U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation,train_features = load_geodata()


    # ----------------------model and data-------------
    model = GCN_pl(nfeat=features.shape[1],
                nhid1=args.hidden,
                nhid2=args.hidden,
                nhid3=args.hidden,
                nclass=labels.max().item() + 1,
                dropout=args.dropout)
    # ----------------------trainer-----------
    early_stop_callback = EarlyStopping(
        monitor='acc_val',
        patience=50,
        mode='max'
    )
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        monitor='acc_val',
        mode='max',
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    trainer = pl.Trainer(gpus=[1],precision=32,max_epochs=1000,early_stop_callback=early_stop_callback,
                             num_sanity_val_steps=-1,checkpoint_callback=checkpoint_callback
                         )

    trainer.fit(model)
    trainer.test()
    # trainer.save_checkpoint("example.ckpt")     #手动
    # heckpoint = torch.load( "example.ckpt")
    # print(checkpoint['checkpoint_callback_best_model_score'])
    '''test'''
    # checkpoint = torch.load( '/home/yanqilong/workspace/GCN-geo/process_code/epoch=417.ckpt')
    # model.load_state_dict(checkpoint['state_dict'],checkpoint['epoch'])
    # trainer.test(model)


    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))





