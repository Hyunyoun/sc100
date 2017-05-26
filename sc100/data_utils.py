import pickle
import numpy as np
import os
from scipy.misc import imread

def load_batch(filename, w=100, h=100, c=3):
    """ load single batch of stock set """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        X = X.reshape(X.shape[0], w, h, c).astype("float")
        Y = datadict['labels']
        Y = Y.reshape(Y.shape[0], -1)
        return X, Y


def load_stock_charts(ROOT, stocks):
    """ load stock charts """
    xs = []
    ys = []
    for stock in stocks:
        f = os.path.join(ROOT, stock)
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    return Xtr, Ytr


def get_stock_chart_data(stocks, n_train=200000, n_valid=1000, n_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw chart data
    chart_dir = 'dpi100'
    X_raw, y_raw = load_stock_charts(chart_dir, stocks)

    # Subsample the data
    seed = 0
    np.random.seed(seed=seed)
    sample_idxs = np.random.choice(X_raw.shape[0], n_train+n_valid+n_test)

    mask = sample_idxs[:n_train]
    X_train = X_raw[mask]
    y_train = y_raw[mask]
    mask = sample_idxs[n_train:n_train+n_valid]
    X_val = X_raw[mask]
    y_val = y_raw[mask]
    mask = sample_idxs[n_train+n_valid:]
    X_test = X_raw[mask]
    y_test = y_raw[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }