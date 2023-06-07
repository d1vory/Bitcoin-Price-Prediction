import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch import nn
import numpy as np
from torch import nn
from ray.util.client import ray


import sys
import torch
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.callbacks import TQDMProgressBar
import numpy as np
import tempfile
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from pytorch_lightning.callbacks import ModelCheckpoint
from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.lightning import (
    LightningTrainer,
    LightningConfigBuilder,
    LightningCheckpoint,
)


from models.transformer.transformer_main import TransAm
from kw_transformer_functions import RMSELoss
from models.transformer.my_functrions import make_dataset, get_torch_data_loaders
from pytorch_lightning.callbacks import TQDMProgressBar


torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



df = pd.read_csv(
    '../datasets/prepared/ada/ta_corr_01_fd.csv',
    parse_dates=True
)
target_col = 'log_returns'
df = df.set_index(['timestamp'])
df.index = pd.to_datetime(df.index)
exclude_cols = ['log_close', 'Close']
df



checkpoint_path = 'checkpoints/multioutput/additional_training/btc_eth_fac_dense_timestep=26epoch=2-val_loss=0.588.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

ntest=42

print(checkpoint["hyper_parameters"])

#loss_fn=checkpoint["hyper_parameters"]['loss_fn']
batch_size = checkpoint["hyper_parameters"]['batch_size']
#num_layers= checkpoint["hyper_parameters"]['num_layers']
dropout =  checkpoint["hyper_parameters"]['dropout']
#nhead= checkpoint["hyper_parameters"]['nhead']
#feature_size= checkpoint["hyper_parameters"]['feature_size']
learning_rate= checkpoint["hyper_parameters"]['learning_rate']
weight_decay=checkpoint["hyper_parameters"]['weight_decay']

timestep=checkpoint["hyper_parameters"]['timestep']
horizon=checkpoint["hyper_parameters"]['horizon']

attn_type = checkpoint["hyper_parameters"]['attn_type']
model_checkpoint_outputdir = 'multioutput/additional_training'
patience = 25
n_epochs = 50


Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, scaler = make_dataset(df, target_col='log_returns',
                                                                exclude_cols=exclude_cols, timestep=timestep, ntest=ntest, horizon=horizon)
train_loader, val_loader, test_loader, test_loader_one = get_torch_data_loaders(
    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, batch_size
)



model = TransAm.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), loss_fn=RMSELoss)


early_stop_callback = EarlyStopping(monitor="val_loss", patience=patience, verbose=False, mode="min")
filename = f"btc_eth_ada_{attn_type}_timestep={timestep}" + '{epoch}-{val_loss:.3f}'
checkpoint_callback = ModelCheckpoint(
    dirpath=f"checkpoints/{model_checkpoint_outputdir}",
    filename=filename,
    save_top_k=1,
    monitor="val_loss"
)


trainer = pl.Trainer(
    callbacks=[TQDMProgressBar(refresh_rate=10), early_stop_callback,checkpoint_callback],
    max_epochs=n_epochs,
    logger=True,
)

trainer.fit(model, train_loader, val_loader)