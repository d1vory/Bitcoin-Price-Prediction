import pandas as pd
import numpy as np
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


def get_scaled_dataframe(df, exclude_cols, target_col, ntest=21):
    df = df.drop(exclude_cols, axis=1)
    train = df.iloc[:-ntest]
    test = df.iloc[-ntest:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    y = train[target_col].to_numpy()
    scaler.fit_transform(y.reshape(-1, 1))
    scaled_df = pd.DataFrame(np.concatenate((train_scaled, test_scaled)), columns=df.columns, index=df.index)
    return scaled_df, scaler

def divide_dataset_multi_step_prediction(df, timestep, target_col, horizon=1):
    series = df.dropna().to_numpy()
    result_series = df[target_col].to_numpy()
    X = []
    Y = []
    for t in range(len(series) - timestep - horizon + 1):
        x = series[t:t+timestep, :]
        X.append(x)
        y = result_series[t+timestep:t+timestep+horizon]
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    last_x = X[-1]
    print("X.shape", X.shape, "Y.shape", Y.shape)
    return X, Y


def split_dataset(X, Y, ntest=21):
    #Xtrain, Ytrain = X[:-ntest, :, :], Y[:-ntest]
    #Xtest, Ytest = X[-ntest:], Y[-ntest:]
    #Xtrain, Xtest, Ytrain, Ytest
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=ntest, shuffle=False)
    Xtrain, XVal, Ytrain, YVal = train_test_split(Xtrain, Ytrain, test_size=ntest, shuffle=False)
    return Xtrain, Ytrain, Xtest, Ytest, XVal, YVal


def make_dataset(df, target_col, exclude_cols, timestep=10, ntest=21, horizon=1):
    scaled_df, scaler = get_scaled_dataframe(df, exclude_cols, target_col, ntest)
    X, Y = divide_dataset_multi_step_prediction(scaled_df, timestep, target_col, horizon=horizon)
    Xtrain, Ytrain, Xtest, Ytest, XVal, YVal = split_dataset(X, Y, ntest)
    return Xtrain, Ytrain, Xtest, Ytest, XVal, YVal, scaler


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

def test_predictions_multistep(preds, real, scaler, unscale=False, last_train=None, apply_func_to_preds=None, concat=False, step=1
                               ):
    if concat:
        predictions = np.concatenate(preds)
    else:
        predictions = preds
    # if unscale:
    #     predictions = scaler.inverse_transform(predictions)
    #     real = scaler.inverse_transform(real)
    for i in range(0, predictions.shape[0] - 1, step):
        if unscale:
            unscaled_y_test = scaler.inverse_transform(real[i].reshape(-1, 1)).flatten()
            real_prices = [np.exp(last_train), *np.exp(last_train +  np.cumsum(unscaled_y_test))]


            prediction = apply_func_to_preds(predictions[i]) if apply_func_to_preds else predictions[i]
            unscaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            #tmp = np.exp(last_train +  np.cumsum(unscaled_prediction))
            #print(prediction)
            kek = [np.exp(last_train), *np.exp(last_train +  np.cumsum(unscaled_prediction))]

            df = pd.DataFrame(data={"value": real_prices, "prediction": kek})
        else:
            df = pd.DataFrame(data={"value": real[i], "prediction": predictions[i]})
        #pd.DataFrame(data={"value": real[i], "prediction": predictions[i] * 10}).plot()
        df.plot()