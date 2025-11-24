import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

def load_csv(path):
    df = pd.read_csv(path, parse_dates=True, index_col=0)
    return df

def generate_synthetic(n=10000, freq=24):
    t = np.arange(n)
    seasonal = 10 * np.sin(2 * np.pi * t / freq)
    trend = 0.001 * t
    noise = np.random.normal(0, 0.5, size=n)
    series = seasonal + trend + noise + 50
    dates = pd.date_range(start='2015-01-01', periods=n, freq='H')
    df = pd.DataFrame({'load': series}, index=dates)
    return df

def prepare_sequences(series, input_len, output_len, scaler=None):
    x = series.values.reshape(-1,1).astype(float)
    if scaler is None:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        x_scaled = scaler.transform(x)
    X, Y = [], []
    total = len(x_scaled)
    for i in range(total - input_len - output_len + 1):
        X.append(x_scaled[i:i+input_len])
        Y.append(x_scaled[i+input_len:i+input_len+output_len])
    X = np.stack(X)
    Y = np.stack(Y)
    return X.astype('float32'), Y.astype('float32'), scaler

def train_val_test_split(X, Y, val_frac=0.1, test_frac=0.1):
    n = len(X)
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - test_n
    return (X[:train_n], Y[:train_n]), (X[train_n:train_n+val_n], Y[train_n:train_n+val_n]), (X[-test_n:], Y[-test_n:])

def to_torch(x):
    return torch.from_numpy(x)
