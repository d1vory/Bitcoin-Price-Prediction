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


from models.transformer.transformer_main import MyTransformer
from models.transformer.my_functrions import make_dataset, get_torch_data_loaders, RMSELoss

torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


df = pd.read_csv(
    '../datasets/prepared/btc/ta_corr_01_fd.csv',
    parse_dates=True
)
target_col = 'log_returns'
df = df.set_index(['timestamp'])
df.index = pd.to_datetime(df.index)
exclude_cols = ['log_close', 'Close']


val_df = pd.read_csv(
    '../datasets/prepared/eth/ta_corr_01_fd.csv',
    parse_dates=True
)
val_df = val_df.set_index(['timestamp'])
val_df.index = pd.to_datetime(val_df.index)


config = {
    'batch_size': tune.choice([2]),
    'num_layers': tune.choice([4]),
    'dropout': tune.choice([0.2]),
    'nhead': tune.choice([2]),
    #'attn_type': tune.grid_search([ 'dense', '', 'rv_mix']),
    #'attn_type': tune.grid_search(['', 'fac_dense', 'dense', 'rv_mix', 'dv_mix']),
    #'attn_type': tune.choice(['fac_dense', 'dense', 'dv_mix']),
    'attn_type': tune.choice(['fac_dense', ]),
    'learning_rate': tune.choice([1e-6]),
    'weight_decay': tune.choice([1e-6]),
    'patience': tune.choice([100]),
    'n_epochs': tune.choice([300]),
    'timestep': tune.choice([26]),
    'horizon': tune.choice([14]),
    'ntest': tune.choice([42]),
    'model_checkpoint_outputdir': 'multioutput/eth_as_val'
}


def train_transformer(config):
    print('CUDA', torch.cuda.is_available())

    timestep= config['timestep']
    ntest = config['ntest']
    horizon = config['horizon']
    decoder_size = 16

    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, scaler = make_dataset(
        df, target_col='log_returns', exclude_cols=exclude_cols, timestep=timestep, ntest=ntest, horizon=horizon
    )
    train_loader, val_loader, test_loader, test_loader_one = get_torch_data_loaders(
        Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, config['batch_size']
    )

    XVal, YVal, _, _, _, _, _ = make_dataset(
        val_df, target_col='log_returns', exclude_cols=exclude_cols, timestep=timestep, ntest=ntest, horizon=horizon
    )
    val_loader, _, _, _ = get_torch_data_loaders(
        Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, config['batch_size']
    )

    feature_size = Xtrain.shape[-1] #input_dim

    loss_fn = RMSELoss

    model = MyTransformer(
        loss_fn=loss_fn,
        batch_size=config['batch_size'],
        decoder_size=decoder_size,
        timestep=timestep,
        horizon=horizon,
        feature_size=feature_size,
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        nhead=config['nhead'],
        attn_type=config['attn_type'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay']
    )


    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config['patience'], verbose=False, mode="min")

    filename = f"{config['attn_type']}_timestep={timestep}" + '{epoch}-{val_loss:.3f}'
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{config['model_checkpoint_outputdir']}",
        filename=filename,
        save_top_k=1,
        monitor="val_loss"
    )

    tune_report_callback = TuneReportCheckpointCallback(
        {"loss": "val_loss"},
        on="validation_end"
    )

    trainer = pl.Trainer(
        callbacks=[early_stop_callback,checkpoint_callback, tune_report_callback],
        max_epochs=config['n_epochs'],
        logger=True,
    )

    trainer.fit(model, train_loader, val_loader)


algo = OptunaSearch()
algo = ConcurrencyLimiter(algo, max_concurrent=3)
scheduler = AsyncHyperBandScheduler()
num_samples = 1
training_iterations = 10


context = ray.init()

#trainable = tune.with_resources(
trainable = tune.with_parameters(
    train_transformer,
    # {
    #     "cpu": 3,
    #     #"gpu": 1,
    #     #'accelerator': NVIDIA_TESLA_A100
    # }
)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 12,
        #"gpu": 1,
        #'accelerator_type': NVIDIA_TESLA_A100
    },
    metric="loss",
    mode="min",
    config=config,

    search_alg = algo,
    scheduler=scheduler,
    num_samples = num_samples,

    name="tune_transformer",
    keep_checkpoints_num=3,
    local_dir=f"./checkpoints/{config['model_checkpoint_outputdir']}"
)
