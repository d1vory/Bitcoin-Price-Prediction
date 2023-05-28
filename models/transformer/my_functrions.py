import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_scaled_dataframe(df, exclude_cols, ntest=21):
    df = df.drop(exclude_cols, axis=1)
    train = df.iloc[:-ntest]
    test = df.iloc[-ntest:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    scaled_df = pd.DataFrame(np.concatenate((train_scaled, test_scaled)), columns=df.columns, index=df.index)
    return scaled_df, scaler

def divide_dataset_one_step_prediction(df, timestep, target_col,):
    series = df.dropna().to_numpy()
    result_series = df[target_col].to_numpy()

    T = timestep
    X = []
    Y = []
    for t in range(len(series) - T):
        x = series[t:t+T, :]
        X.append(x)
        y = result_series[t+T]
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    N = len(X)
    print("X.shape", X.shape, "Y.shape", Y.shape)
    return X, Y


def split_dataset(X, Y, ntest=21):
    Xtrain, Ytrain = X[:-ntest], Y[:-ntest]
    Xtest, Ytest = X[-ntest:], Y[-ntest:]

    Xtrain, XVal, Ytrain, YVal = train_test_split(Xtrain, Ytrain, test_size=21, shuffle=False)
    return Xtrain, Ytrain, Xtest, Ytest, XVal, YVal


def make_dataset(df, target_col, exclude_cols, timestep=10, ntest=21):
    scaled_df, scaler = get_scaled_dataframe(df, exclude_cols, ntest)
    X, Y = divide_dataset_one_step_prediction(scaled_df, timestep, target_col)
    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal = split_dataset(X, Y, ntest)
    return Xtrain, Ytrain, Xtest, Ytest, XVal, YVal

def get_torch_data_loaders(Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, batch_size):
    x_train = torch.Tensor(Xtrain)
    y_train = torch.Tensor(Ytrain)
    x_val = torch.Tensor(XVal)
    y_val = torch.Tensor(YVal)
    x_test = torch.Tensor(Xtest)
    y_test = torch.Tensor(Ytest)

    train = TensorDataset(x_train, y_train)
    val = TensorDataset(x_val, y_val)
    test = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1, shuffle=False, drop_last=True)
    return train_loader, val_loader, test_loader, test_loader_one
