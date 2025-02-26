import numpy as np
import pandas as pd
import re
import itertools
import time
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer

def apply_dimensionality_reduction(X_train, X_test, params):
    """
    Applies dimensionality reduction if selected
    and returns the modified data
    """
    # 1) Check if dimensionality reduction was selected
    reduce_dimensionality = params.get("reduce_dimensionality", False)
    if not reduce_dimensionality:
        return X_train, X_test

    # 2) Get the type of dimensionality reducer
    dimensionality_reducer = params.get("dimensionality_reducer")
    if not dimensionality_reducer:
        raise ValueError("Dimensionality reduction is enabled, but 'dimensionality_reducer' is not specified.")

    # 3) Initialize the selected reducer (used to have more but they broke from library update)
    if dimensionality_reducer == 'pca':
        pca_n_components = params.get("pca_n_components") 
        pca_svd_solver = params.get("pca_svd_solver", 'auto')
        reducer = PCA(n_components=pca_n_components, svd_solver=pca_svd_solver, random_state=42)
    else:
        raise ValueError(f"Unsupported dimensionality reducer: {dimensionality_reducer}")

    # 4) fit on train and apply to test
    try:
        X_train_reduced = reducer.fit_transform(X_train)
        X_test_reduced = reducer.transform(X_test)
    except Exception as e:
        raise RuntimeError(f"Failed to apply {dimensionality_reducer.upper()}: {e}")

    return X_train_reduced, X_test_reduced

def discretize_y(y, model):
    '''
    Converts y from a continous target to a 
    categorical target based upon a number of
    classes preselected while maintaining classes
    can only contain pos or neg values
    '''
    
    # 1) Determine number of negative vs positive values
    y = np.asarray(y)
    N = len(y)
    num_classes = model.params['num_classes']
    neg_mask = (y < 0)
    n = np.sum(neg_mask)  # number of negative elements
    m = N - n             # number of non-negative elements
    
    
    # 2) Compute how many classes to allocate to the negative vs positive
    base_class_size = N // num_classes
    num_negative_classes = int(np.ceil(n / base_class_size)) 
    num_positive_classes = num_classes - num_negative_classes
    
    # 3) Isolate pos and neg values
    neg_idx = np.where(neg_mask)[0]
    pos_idx = np.where(~neg_mask)[0]
    
    # 4) Sort within each group
    neg_vals = y[neg_idx]
    neg_sorted_order = np.argsort(neg_vals)[::-1]  # descending
    neg_sorted_idx = neg_idx[neg_sorted_order]
    
    pos_vals = y[pos_idx]
    pos_sorted_order = np.argsort(pos_vals)  # ascending
    pos_sorted_idx = pos_idx[pos_sorted_order]
    
    # 6) Assign labels and calculate averages for negative chunks
    classes = np.zeros(N, dtype=int)
    averages = []
    for i in range(num_negative_classes):
        start = i * base_class_size
        end = (i+1) * base_class_size
        if end > len(neg_sorted_idx):
            end = -1
        chunk_idx = neg_sorted_idx[start:end]
        classes[chunk_idx] = i
        if len(chunk_idx) > 0:
            lower = np.min(y[chunk_idx])
            upper = np.max(y[chunk_idx])
            averages.append((upper + lower) / 2)
    
    # 7) Assign labels and calculate averages for non-negative chunks
    for i in range(num_positive_classes):
        start = i * base_class_size
        end = (i+1) * base_class_size
        if end > len(pos_sorted_idx):
            end = -1
        chunk_idx = pos_sorted_idx[start:end]
        classes[chunk_idx] = i + num_negative_classes
        if len(chunk_idx) > 0:
            lower = np.min(y[chunk_idx])
            upper = np.max(y[chunk_idx])
            averages.append((upper + lower) / 2)
    
    model.params['class_values'] = averages
    return classes
