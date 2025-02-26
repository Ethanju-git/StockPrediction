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

def greedy_info_construction(df, model, last_seen_idx):
    '''
    Incrementally selects features by selecting the one that 
    maximizes the condition: correlation with target - max
    correlation with other selected feature. A modification of
    MRMR that is more punitive effectively. Returns df of the
    features sorted by how much they fulfill the criteria
    '''
    # 1) Initialize assign necessary variables
    selected_features = []
    cols = df.columns.tolist()
    target_col = f'Close_pct_{model.target_ticker}'

    # 2) Ensure target_col exists and exclude it from feature cols
    if target_col not in cols:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    feature_cols = [col for col in cols if col != target_col]

    # 3) Compute correlation matrix for seen data
    df_seen = df.iloc[:last_seen_idx]
    corr_matrix = df_seen.corr().abs()

    # 4) Incrementally add features based upon the dual objective maximization
    while len(selected_features) < model.max_num_features:
        # 4.1) First case, select one most correlated with target
        if len(selected_features) == 0:
            relevance = corr_matrix.loc[target_col, feature_cols]
            max_corr = relevance.max()
            max_col = relevance.idxmax()
            selected_features.append(max_col)
            feature_cols.remove(max_col)
        else:
            # 4.2) Compute which maximizes the dual objective and add it
            relevance = corr_matrix.loc[target_col, feature_cols] 
            redundancy = corr_matrix.loc[selected_features, feature_cols].max(axis=0)

            #C = [c1, c2, ... cn] of chosen features
            #P = [p1, p2, ... pm] of potential features
            #relevance = [(p1, t), (p2, t) ... (pm, t)]
            #redundancy = [c1p1 c1p2 ... c1pm \n ... \n cnp1 cnp2 ... cnpm]
            #max redun = [cjp1 (where j max redun), cop2, ... ckpm] each variable maxes it
            #score = relevance - redun = [(p1t - cjp1) (p2t - cop2) ... (pmt - ckpm)] = S = [s1 s2 ... sm]
            #max si in S = best potential col score
            #idx max = best potential col

            
            # Compute MRMR score
            scores = relevance - redundancy
            greedy_score = scores.max()
            greedy_col = scores.idxmax()

            # Select the feature with the highest MRMR score
            max_score = greedy_score.max()


            selected_features.append(greedy_col)
            feature_cols.remove(greedy_col)

    # Ensure target_col is included once
    selected_features.append(target_col)
    new_df = df[selected_features]
    return new_df



def mrmr_info_construction(df, model, last_seen_idx):
    """
    Select and sort features by how they maximizes correlation with
    the target - avg correlation with selected features. Return a df
    with the values sorted by such. 
    """
    # 1) Initialize assign necessary variables
    selected_features = []
    cols = df.columns.tolist()
    target_col = f'Close_pct_{model.target_ticker}'

    # 2) Ensure target_col exists and exclude it from feature cols
    if target_col not in cols:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    feature_cols = [col for col in cols if col != target_col]

    # 3) Compute correlation matrix for seen data
    df_seen = df.iloc[:last_seen_idx]
    corr_matrix = df_seen.corr().abs()


    # 4) Incrementally add features based upon the dual objective maximization
    while len(selected_features) < model.max_num_features:
        # 4.1) First case, select one most correlated with target
        if len(selected_features) == 0:
            relevance = corr_matrix.loc[target_col, feature_cols]
            max_corr = relevance.max()
            max_col = relevance.idxmax()
            selected_features.append(max_col)
            feature_cols.remove(max_col)
        else:
            # 4.2) Compute which maximizes the dual objective and add it
            relevance = corr_matrix.loc[target_col, feature_cols]
            redundancy = corr_matrix.loc[selected_features, feature_cols].mean(axis=0)
            
            # Compute MRMR score
            mrmr_score = relevance - redundancy

            # Select the feature with the highest MRMR score
            max_score = mrmr_score.max()

            max_col = mrmr_score.idxmax()
            selected_features.append(max_col)
            feature_cols.remove(max_col)

    # Ensure target_col is included once
    selected_features.append(target_col)
    new_df = df[selected_features]
    return new_df