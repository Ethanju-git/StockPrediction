global_stock_data = None

from .feature_creation import find_cointegrated_combos_with_target, calculate_technical_indicators
from .ticker_selection import greedy_ticker_construction, mrmr_ticker_construction
from .data_transformations import apply_dimensionality_reduction, discretize_y
from .feature_selection import greedy_info_construction, mrmr_info_construction 
import time
import numpy as np
import pandas as pd


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
import time
import pandas as pd

def fetch_data(model):
    '''
    This function takes in a model object and creates the data that it is expecting.
    It downloads or reads the required data, filters by date, checks for nans, and
    returns a df of the relevant data
    '''
    max_missing_ratio = 0.02 #max percentage missing allowed for each stock
    global global_stock_data

    # 1) Load global dataset once, if needed
    if global_stock_data is None:
        global_stock_data = pd.read_csv('/notebooks/stock_data.csv')
        global_stock_data['Date'] = pd.to_datetime(global_stock_data['Date'])


    # 2) Infer tickers from CSV column names (columns formatted as "<prefix>_<ticker>")
    inferred_tickers = set()
    for col in global_stock_data.columns:
        if col == "Date":
            continue
        parts = col.split('_')
        if len(parts) > 1:
            inferred_tickers.add(parts[-1])
    
    # 3) Ensure the target ticker is in the data and create list of tickers
    if model.target_ticker not in inferred_tickers:
        inferred_tickers.add(model.target_ticker)
        print('REACHED THE INFERRED ADD')
    requested_tickers = list(inferred_tickers)
    
    # 4) Create a list of expected columns
    prefixes = ['Close', 'High', 'Low', 'Open', 'Volume', 'Close_pct']
    all_columns = ['Date']
    for tkr in requested_tickers:
        # Build expected column names for this ticker
        matching_cols = [f"{prefix}_{tkr}" for prefix in prefixes 
                         if f"{prefix}_{tkr}" in global_stock_data.columns]
        if matching_cols:
            all_columns += matching_cols


    # 5) Filter the global data to only the expected columns
    df = global_stock_data[all_columns]
    if df.columns.duplicated().any():
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
    else:
        print('No duplicates after filter/copy.')

    # 6) Filter for the appropriate date range
    start_date = pd.to_datetime(model.start_date)
    end_date = pd.to_datetime(model.end_date)
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
    df.set_index('Date', inplace=True)
    if df.empty:
        raise ValueError(f"No data available for date range {start_date} to {end_date}.")

    # 7) Find tickers with too many NaNs and raise error if it includes target
    tickers_to_drop = []
    for tkr in requested_tickers:
        tkr_cols = [col for col in df.columns if col.endswith(tkr)]
        if not tkr_cols:
            tickers_to_drop.append(tkr)
            continue
        total_cells = df[tkr_cols].shape[0] * df[tkr_cols].shape[1]
        num_nans = df[tkr_cols].isna().sum().sum()
        missing_ratio = num_nans / total_cells if total_cells else 1.0
        if missing_ratio > max_missing_ratio:
            tickers_to_drop.append(tkr)

    if model.target_ticker in tickers_to_drop:
        raise ValueError(
            f"Target ticker {model.target_ticker} exceeds missing threshold "
            f"({max_missing_ratio:.2f}); cannot proceed."
        )

    # 8) Remove columns corresponding to dropped tickers
    for dropped_tkr in tickers_to_drop:
        dropped_cols = [col for col in df.columns if col.endswith(dropped_tkr)]
        df.drop(columns=dropped_cols, inplace=True)

    # 10) Fill missing values (forward fill; optionally back fill)
    df.fillna(method='ffill', inplace=True)
    # df.fillna(method='bfill', inplace=True)  # Optional second pass

    # 11) Final sanity check
    target_col = f'Close_pct_{model.target_ticker}'
    if target_col not in df.columns:
        raise KeyError(f"failed to fetch data for target ticker. Please select a different ticker or timeframe.")

    
    if df.empty or len(df.columns) == 0:
        raise ValueError("All data was excluded or dropped; no columns remain.")

    if df.columns.duplicated().any():
        print('df has duplicated cols after fetching data')

    return df




def prepare_data(model):
    '''
    This function reads in the data sorted by the selected heuristic,
    filters for selected num of features discretizes it if necessary,
    applies the selected scaler, and then returns the data
    '''
    
    # 1) Read in data sorted by heuristic
    df = pd.read_csv(f"/notebooks/{model.params['heuristic']}_df.csv")
    
    # 2) Create X, y
    y = df.pop(f'Close_pct_{model.target_ticker}')
    X = df
    y = y.to_numpy()
    X = X.to_numpy()
    y = y[1:]
    X = X[:-1, :]
    X = X[:, :model.params['num_features']]
    
    # 3) Discretize if applicable
    if not model.is_regressive:
        y = discretize_y(y, model)
        unique, counts = np.unique(y, return_counts=True)
        class_balance = dict(zip(unique, counts))
        print("Class Balance After Discretization:", class_balance)
    
    # 4) Infer the last seen idx
    if model.seen_y is not None and model.seen_y.size > 0:
        last_seen_idx = len(model.seen_y)
    else:
        last_seen_idx = int(len(df) * model.init_train_perc)
    
    # 5) Seperate seen and unseen data
    model.seen_X = X[:last_seen_idx]
    model.unseen_X = X[last_seen_idx:]
    model.seen_y = y[:last_seen_idx]
    model.unseen_y = y[last_seen_idx:]
    
    # 6) Store the selected scaler
    if model.params['scaler'] == 'minmax':
        model.scaler_X = MinMaxScaler()
    elif model.params['scaler'] == 'standard':
        model.scaler_X = StandardScaler()
    elif model.params['scaler'] == 'yeo-johnson':
        model.scaler_X = PowerTransformer(method='yeo-johnson', standardize=True)
    elif model.params['scaler'] == 'quantile-normal':
        model.scaler_X = QuantileTransformer(output_distribution='normal', random_state=42)
    elif model.params['scaler'] == 'quantile-uniform':
        model.scaler_X = QuantileTransformer(output_distribution='uniform', random_state=42)
    
    # 7) Fit on seen data, apply to all data
    model.scaler_X.fit(model.seen_X)
    
    model.seen_X = model.scaler_X.transform(model.seen_X)
    model.unseen_X = model.scaler_X.transform(model.unseen_X)
    model.seen_y = model.seen_y[-len(model.seen_X):]
    model.unseen_y = model.unseen_y[-len(model.unseen_X):]
    
    # 8) Apply dimensionality reduction
    model.seen_X, model.unseen_X = apply_dimensionality_reduction(
        model.seen_X, model.unseen_X, model.params
    )

    # 9) Update input dim 
    model.params['input_dim'] = model.seen_X.shape[1]



def generate_data(model):
    '''
    This function fetches data, selects relevant tickers, creates a df
    with those tickers, creates synthetic cointegrated tickers, calculates
    technical indicators for all tickers, sorts the data by heuristic,
    saves the data related to each heuristic
    '''
    # 1) Set target col and seperate target ticker from other tickers
    model.other_tickers = [t for t in model.other_tickers if t != model.target_ticker]
    target_col = f'Close_pct_{model.target_ticker}'
    
    # 2) Fetch data
    df = fetch_data(model)
    
    # 3) Check for NaNs
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame columns: {df.columns.tolist()}")
    if df.isna().any().any():
        print("Warning: NaN values detected in fetched data.")
        df = df.fillna(0)
    if np.isinf(df.values).any():
        print("Warning: Inf values detected in fetched data.")
        df = df.replace([np.inf, -np.inf], 0)


    # 4) Infer train cutoff (seen vs. unseen split)
    if model.seen_y is not None and model.seen_y.size > 0:
        last_seen_idx = len(model.seen_y)
    else:
        last_seen_idx = int(len(df) * model.init_train_perc)
        
    # 5) Select tickers that add info
    greedy_tickers = greedy_ticker_construction(df, model, last_seen_idx)
    mrmr_tickers = mrmr_ticker_construction(df, model, last_seen_idx) 

    # 6) Create df with those tickers
    greedy_tickers.append(model.target_ticker)
    mrmr_tickers.append(model.target_ticker)
    greedy_tickers = ['_' + tkr for tkr in greedy_tickers]
    mrmr_tickers = ['_' + tkr for tkr in mrmr_tickers]
    greedy_selected_columns = [col for col in df.columns if any(col.endswith(tkr) for tkr in greedy_tickers)]
    mrmr_selected_columns = [col for col in df.columns if any(col.endswith(tkr) for tkr in mrmr_tickers)]
    greedy_df = df[greedy_selected_columns]
    mrmr_df = df[mrmr_selected_columns]
    
    
    # 7) Create cointegrated "tickers" using the heuristic selected info to limit search pool
    greedy_df, coint_results = find_cointegrated_combos_with_target(greedy_df, last_seen_idx, model)
    mrmr_df, coint_results = find_cointegrated_combos_with_target(mrmr_df, last_seen_idx, model)

    
    # 8) Calculate technical indicators with heuristic selected stocks and synthetic stocks
    greedy_df = calculate_technical_indicators(greedy_df, model)
    mrmr_df = calculate_technical_indicators(mrmr_df, model)


    # 9) Sort features by heuristic
    greedy_df = greedy_info_construction(greedy_df, model, last_seen_idx)
    mrmr_df = mrmr_info_construction(mrmr_df, model, last_seen_idx)

    # 10) NaN check
    if greedy_df.isna().any().any():
        print("Warning: NaN values detected after calculating technical indicators.")
        greedy_df = greedy_df.fillna(0)
    if np.isinf(greedy_df.values).any():
        print("Warning: Inf values detected after calculating technical indicators.")
        greedy_df = greedy_df.replace([np.inf, -np.inf], 0)
        
    if mrmr_df.isna().any().any():
        print("Warning: NaN values detected after calculating technical indicators.")
        mrmr_df = mrmr_df.fillna(0)
    if np.isinf(mrmr_df.values).any():
        print("Warning: Inf values detected after calculating technical indicators.")
        mrmr_df = mrmr_df.replace([np.inf, -np.inf], 0)

    # 11) Save to heuristic sorted data to CSV
    greedy_df.to_csv('greedy_df.csv', index=False)
    mrmr_df.to_csv('mrmr_df.csv', index=False)

    # 12) Retrieve target.
    target_col = f'Close_pct_{model.target_ticker}'
    y = greedy_df[target_col].to_numpy()

    # 13) Sanity check target
    if np.isnan(y).any():
        print(f"Error: NaN values detected in target column {target_col}.")
        y = np.nan_to_num(y)
    if np.isinf(y).any():
        print(f"Error: Inf values detected in target column {target_col}.")
        y = np.nan_to_num(y)


