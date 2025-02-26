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
import pandas as pd

def greedy_ticker_construction(df, model, last_seen_idx):
    '''
    This function utilizes a heuristic to select which tickers to 
    include for feature creation and selection. We add tickers
    incrementally based upon how they maximize the condition:
    corr with target - max correlation with other selected tickers.
    We use the stocks percent change for this as most technical
    indicators are derived from it.
    '''
    # 1) initialize and set relevant variables
    selected_tickers = []
    df_seen = df[:last_seen_idx]
    df_unseen = df[last_seen_idx:]
    corr_df = df_seen
    target_col = f'Close_pct_{model.target_ticker}'
    n = 30 #number to include in the search

    # 2) Filter columns that match the 'Close_pct_tkr' pattern, excluding the target column
    relevant_cols = [col for col in df_seen.columns if col.startswith('Close_pct_') and col != target_col]

    # 3) Calculate absolute correlations with the target column
    correlations = df_seen[relevant_cols].corrwith(df_seen[target_col]).abs().sort_values(ascending=False)

    # 4) Select the top n most correlated columns
    most_correlated_cols = correlations.nlargest(n).index.tolist()
    most_correlated_df = df_seen[most_correlated_cols]
    potential_tickers = [col.replace('Close_pct_', '') for col in most_correlated_cols]


    # 5) Search over the most correlated and incrementally add those that maximize the dual objective
    while len(selected_tickers) < model.max_n_correlated_stocks:
        # 5.1) Case of none been added, add the most correlated
        if len(selected_tickers) == 0: #none have been added yet
            most_corr = potential_tickers[0]
            selected_tickers.append(most_corr)
            potential_tickers.remove(most_corr)
        else:
            # 5.2) Otherwise, add one that maximizes corr target - max corr with other selected tickers.
            max_new_info = -9999
            for pot_tkr in potential_tickers:                                                                                                        
                target_corr = abs(df_unseen[f'Close_pct_{model.target_ticker}'].corr(df_unseen[f'Close_pct_{pot_tkr}']))
                max_corr = 0
                for tkr in selected_tickers:
                    temp_corr = abs(df_unseen[f'Close_pct_{tkr}'].corr(df_unseen[f'Close_pct_{pot_tkr}']))
                    if max_corr < temp_corr:
                        max_corr = temp_corr
                new_info = target_corr - max_corr
                if max_new_info < new_info:
                    max_new_info = new_info
                    max_new_info_tkr = pot_tkr
            selected_tickers.append(max_new_info_tkr)
            potential_tickers.remove(max_new_info_tkr)
    return selected_tickers
 

def mrmr_ticker_construction(df, model, last_seen_idx):
    '''
    This function utilizes a heuristic to select which tickers to 
    include for feature creation and selection. We add tickers
    incrementally based upon how they maximize the condition:
    corr with target - avg correlation with other selected tickers.
    We use the stocks percent change for this as most technical
    indicators are derived from it.
    '''

    # 1) initialize and set relevant variables
    selected_tickers = []
    df_seen = df.iloc[:last_seen_idx]
    corr_matrix = df_seen.corr().abs()
    target_col = f'Close_pct_{model.target_ticker}'
    n = 30  #number to include in the search

    # 2) Filter columns that match the 'Close_pct_tkr' pattern, excluding the target column
    relevant_cols = [col for col in df_seen.columns if col.startswith('Close_pct_') and col != target_col]

    # 3) Calculate absolute correlations with the target column
    correlations = df_seen[relevant_cols].corrwith(df_seen[target_col]).abs().sort_values(ascending=False)

    # 4) Select the top n most correlated columns
    most_correlated_cols = correlations.nlargest(n).index.tolist()
    most_correlated_df = df_seen[most_correlated_cols]
    potential_tickers = [col.replace('Close_pct_', '') for col in most_correlated_cols]
    potential_cols = [f'Close_pct_{tkr}' for tkr in potential_tickers]

    # 5) Search over the most correlated and incrementally add those that maximize the dual objective
    while len(selected_tickers) < model.max_n_correlated_stocks:
        if len(selected_tickers) == 0:
            # 5.1) Case of none been added, add the most correlated
            max_corr = corr_matrix.loc[target_col, potential_cols].max()
            max_col = corr_matrix.loc[target_col, potential_cols].idxmax()
            max_tkr = max_col.split('_')[-1]
            selected_tickers.append(max_tkr)
            potential_tickers.remove(max_tkr)
            potential_cols.remove(max_col)
        else:
            # 5.2) Otherwise, add one that maximizes corr target - avg corr with other selected tickers.
            relevance = corr_matrix.loc[target_col, potential_cols]
            redundancy = corr_matrix.loc[[f'Close_pct_{tkr}' for tkr in selected_tickers], potential_cols].mean(axis=0)
            
            mrmr_score = relevance - redundancy
            max_score = mrmr_score.max()
            if max_score <= 0:
                break
            max_col = mrmr_score.idxmax()
            max_tkr = max_col.split('_')[-1]
            selected_tickers.append(max_tkr)
            potential_tickers.remove(max_tkr)
            potential_cols.remove(max_col)

    return selected_tickers
