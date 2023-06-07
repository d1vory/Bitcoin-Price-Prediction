import pytorch_lightning as pl
from torch.utils.data import DataLoader

import torch
from torch import nn

import pytorch_lightning as pl

import pandas as pd
import numpy as np

from models.transformer.transformer_encoder_layer import TransformerEncoderLayer
from models.transformer.positional_encoding import PositionalEncoding
from models.transformer.my_functrions import make_dataset, get_torch_data_loaders, RMSELoss


class MyTransformer(pl.LightningModule):
    def __init__(
            self,
            loss_fn=None,
            batch_size=32,
            feature_size=1,
            decoder_size=16,
            timestep=10,
            horizon=1,
            num_layers=1,
            dropout=0.1,
            nhead=2,
            attn_type=None,
            learning_rate=1e-5,
            weight_decay=1e-6
    ):
        super(MyTransformer, self).__init__()

        self.model_type = 'Transformer'
        self.attn_type=attn_type
        self.batch_size=batch_size
        self.nhead=nhead
        self.feature_size=feature_size
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.loss_fn = loss_fn or RMSELoss
        self.source_masking = None
        self.positional_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout,attn_type=attn_type)
        self.transformer_encoder = nn.TransformerEncoder(
            TransformerEncoderLayer(d_model=feature_size, nhead=nhead, dropout=dropout,attn_type=attn_type),
            num_layers=num_layers
        )
        # self.decoder1 = nn.Linear(feature_size, decoder_size)
        # self.decoder2 = nn.Linear(timestep * decoder_size, horizon)
        self.decoder1 = nn.Linear(timestep * feature_size, horizon)

        initrange = 0.1
        self.decoder1.bias.data.zero_()
        self.decoder1.weight.data.uniform_(-initrange, initrange)
        # self.decoder2.bias.data.zero_()
        # self.decoder2.weight.data.uniform_(-initrange, initrange)
        self.save_hyperparameters()

    def forward(self, src):
        if self.source_masking is None or self.source_masking.size(0) != len(src):
            self.source_masking = self._get_mask(len(src)).to(src.device)

        src = self.positional_encoder(src)
        output = self.transformer_encoder(src, self.source_masking)
        # output = self.decoder1(output)
        # viewed = output.view(src.shape[0], -1)
        # output = self.decoder2(viewed)

        viewed = output.view(src.shape[0], -1)
        output = self.decoder1(viewed)
        return output

    @staticmethod
    def _get_mask(size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=128, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=128, num_workers=32)

    def test_dataloader(self):
        pass

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        pred = self.forward(x.view([self.batch_size, -1, self.feature_size]))
        loss = self.loss_fn(pred, y)
        self.log('training_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        kek = x.view([self.batch_size, -1, self.feature_size])
        pred = self.forward(kek)
        loss = self.loss_fn(pred, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx, batch_size=1):
        x, y = test_batch
        pred = self.forward(x.view([batch_size, -1, self.feature_size]))
        loss = self.loss_fn(pred, y)
        self.log('Test loss', loss)
        return loss
    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.view([1, -1, self.feature_size])
        pred = self.forward(x.view([1, -1, self.feature_size]))
        return pred


if __name__ == '__main__':

    df = pd.read_csv(
        '../datasets/prepared/scaler_test.csv',
        parse_dates=True
    )
    target_col = 'log_returns'
    df = df.set_index(['timestamp'])
    df.index = pd.to_datetime(df.index)

    horizon=21
    batch_size=16

    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, scaler = make_dataset(
        df, target_col='log_returns', exclude_cols=[], timestep=10, ntest=21, horizon=horizon)
    train_loader, val_loader, test_loader, test_loader_one = get_torch_data_loaders(
        Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, batch_size
    )
    feature_size = Xtrain.shape[-1] #input_dim

    loss_fn = RMSELoss

    trainer = pl.Trainer(
        callbacks=[],
        max_epochs=2,
        logger=False,
    )

    model = MyTransformer(
        loss_fn=loss_fn,
        batch_size=batch_size,
        decoder_size=16,
        timestep=10,
        horizon=horizon,
        feature_size=feature_size,
        num_layers=4,
        dropout=0.1,
        nhead=2,
        attn_type='fac_random'
    )
    trainer.fit(model, train_loader, val_loader)


    # trainer.test(model, test_loader_one)
    # preds = trainer.predict(model, test_loader_one)
#%%
