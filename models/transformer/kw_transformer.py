import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader

import torch
from torch import nn
import random
from typing import Optional, Any, Union, Callable, Tuple
import mlflow

import torch
from torch import nn
from torch import Tensor
import torch.optim as optim
import torch.nn.functional as F

#from kw_lstm import LSTMModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from kw_transformer_functions import calculate_metrics, RMSELoss, RMSPELoss, plot_dataset, inverse_transform, format_predictions, train_val_test_split , plot_predictions, kw_dataload,final_split,final_dataload

import math
import pandas as pd
import numpy as np
import time
from datetime import datetime
import copy
import os

from models.transformer.kw_TransformerEncoderLayer import TransformerEncoderLayer
from models.transformer.kw_transformer_layers import PositionalEncoding
from models.transformer.my_functrions import make_dataset, get_torch_data_loaders


class TransAm(pl.LightningModule):
    def __init__(self, loss_fn=None, batch_size=32, feature_size=1, decoder_size=16, timestep=10, num_layers=1, dropout=0.1, nhead=2,
                 attn_type=None, learning_rate=1e-5, weight_decay=1e-6):
        super(TransAm, self).__init__()

        self.model_type = 'Transformer'
        self.attn_type=attn_type
        self.batch_size=batch_size
        self.nhead=nhead
        self.feature_size=feature_size
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.loss_fn = loss_fn or RMSELoss
        print('kw_batch,feature size: ',batch_size,feature_size)

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout,attn_type=attn_type)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        #self.transformer_encoder = Encoder( input_size=50,heads=2, embedding_dim=feature_size, dropout_rate=dropout, N=num_layers)
        self.decoder1 = nn.Linear(feature_size, decoder_size)
        self.decoder2 = nn.Linear(timestep * decoder_size, 1)

        #self.save_hyperparameters("feature_size","batch_size", "learning_rate","weight_decay")   
        self.init_weights()
        self.save_hyperparameters()

    def init_weights(self):
        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)

        self.decoder2.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):

        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)#, self.src_mask)
        output = self.decoder1(output)
        output = self.decoder2(output.view(self.batch_size, -1))
        #output=F.relu(output)

        #add sigmoid function <- output=sigmoid. force output to be 0-1. and
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


    def train_dataloader(self):
        # REQUIRED
        # This is an essential function. Needs to be included in the code

        return DataLoader(self.train_set, batch_size=128, num_workers=32)

    def val_dataloader(self):
        # OPTIONAL
        #loading validation dataset
        return DataLoader(self.val_set, batch_size=128, num_workers=32)

    def test_dataloader(self):
        pass
        # OPTIONAL
        # loading test dataset
        #return DataLoader(MNIST(os.getcwd(), train=False, download=False, transform=transforms.ToTensor()), batch_size=128,num_workers=32)

    #def RMSE_loss(self, logits, labels):
    #    return self.loss_fn(logits, labels)

    #def on_train_start(self):
    #    self.logger.log_hyperparams({"hp/learning_rate": self.learning_rate,
    #                                           "hp/batch_size": self.batch_size})
    #    kw_dict=dict()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = x.view([self.batch_size, -1, self.feature_size])
        pred = self.forward(x)
        # The actual forward pass is made on the
        #input to get the outcome pred from the model
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
        print('training_loss', loss)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = x.view([self.batch_size, -1, self.feature_size])
        pred = self.forward(x)
        pred = pred.view(-1,1)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss)
        #print('val_loss:',loss)
        return loss

    def test_step(self, test_batch, batch_idx,batch_size=1):
        x, y = test_batch

        x = x.view([batch_size, -1, self.feature_size])
        pred = self.forward(x)
        pred = pred.view(-1,1)

        loss = self.loss_fn(pred, y)
        self.log('Test loss', loss)

        return loss


    #    print(len(losses)) ## This will be same as number of validation batches
    def predict_step(self, batch, batch_idx):
        x, y = batch

        x = x.view([1, -1, self.feature_size])
        pred = self.forward(x)
        pred = pred.view(-1,1)
        #pred = pred.view(-1,1)


        return pred


if __name__ == '__main__':

    df = pd.read_csv(
        '../datasets/prepared/log_diffed.csv',
        parse_dates=True
    )
    target_col = 'log_returns'
    df = df.set_index(['timestamp'])
    df.index = pd.to_datetime(df.index)


    # X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df, target_col, 0.15)
    # train_loader, val_loader, test_loader, test_loader_one,scaler=kw_dataload(4, X_train, X_val, X_test, y_train, y_val, y_test)
    # feature_size = len(X_train.columns) #input_dim
    #



    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal = make_dataset(df, target_col='log_returns',
                                                            exclude_cols=['RSI-based MA'], timestep=10, ntest=21)
    train_loader, val_loader, test_loader, test_loader_one = get_torch_data_loaders(
        Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, 4
    )
    feature_size = Xtrain.shape[-1] #input_dim

    loss_fn = RMSELoss

    trainer = pl.Trainer(
        callbacks=[],
        max_epochs=10,
        logger=False,
    )

    model = TransAm(
        loss_fn=loss_fn,
        batch_size=4,
        decoder_size=16,
        timestep=10,
        feature_size=feature_size,
        num_layers=4,
        dropout=0.1,
        nhead=2,
        attn_type=''
    )
    # with mlflow.start_run(experiment_id=cfg.mlflow.experiment_id,run_name = cfg.mlflow.run_name) as run:
    #     mlflow.log_params(hyperparameters)
    trainer.fit(model, train_loader, val_loader)


#%%
