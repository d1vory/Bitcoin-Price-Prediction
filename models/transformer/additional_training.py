import torch

#from pytorch_lightning.callbacks import TQDMProgressBar

# from pytorch_lightning.callbacks import TQDMProgressBar
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.transformer.my_functrions import make_dataset, get_torch_data_loaders, RMSELoss
from models.transformer.transformer_main import MyTransformer

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

checkpoint_path = 'checkpoints/multioutput/additional_training/btc_eth_fac_dense_timestep=26epoch=2-val_loss=0.588.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

ntest=42

print(checkpoint["hyper_parameters"])
batch_size = checkpoint["hyper_parameters"]['batch_size']

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



model = MyTransformer.load_from_checkpoint(checkpoint_path, map_location=torch.device('cpu'), loss_fn=RMSELoss)


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