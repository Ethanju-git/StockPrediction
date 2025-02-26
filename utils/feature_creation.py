import re
import itertools
import statsmodels.api as sm
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
import numpy as np
import pandas as pd

def find_cointegrated_combos_with_target(
    df,
    last_seen_idx,
    model,
    significance=0.05,
    max_abs_value=1e9  # optional: if any synthetic value exceeds this, we discard
):
    """
    Function finds linear combinations of stocks that create cointegration
    with the desired target. It then validates that relationship out of sample,
    and if it holds, we create a sythnetic stock with that combination.
    """
    target_col = f"Close_pct_{model.target_ticker}"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in df.")

    # 1) Seperate into seen, unseen. separate seen into train, val.
    train_end_idx = int(0.6 * last_seen_idx)
    val_end_idx = last_seen_idx
    df_train  = df.iloc[:train_end_idx].copy()
    df_val    = df.iloc[train_end_idx:val_end_idx].copy()
    df_unseen = df.iloc[val_end_idx:].copy()

    # 2) Determine the feature col names
    pattern = r'^Close_pct_(\w+)$'
    possible_cols = [c for c in df.columns if re.match(pattern, c)]
    feature_cols  = [c for c in possible_cols if c != target_col]

    coint_results = {}
    synth_counter = 1

    # 3) Determine the possible combos
    single_combos = [(f,) for f in feature_cols]
    pair_combos = list(itertools.combinations(feature_cols, 2))
    #triplet_combos = list(itertools.combinations(feature_cols, 3))
    # can continue this as one sees fit.

    # 4) See if the combo has a linear combination that is cointegrated, if so make it into a synthetic ticker
    all_combos = single_combos + pair_combos
    for combo in all_combos:
        # 4.1) Check for NaNs and data availability
        subdf_train = df_train[list(combo) + [target_col]].dropna()
        if len(subdf_train) < 30:
            # skip if we don't have enough data
            continue
        
        # 4.2) Seperate X, y
        X_train = subdf_train[list(combo)]
        y_train = subdf_train[target_col]
        X_train_const = sm.add_constant(X_train, prepend=True)

        # 4.3) Fit the model
        try:
            ols_model = sm.OLS(y_train, X_train_const).fit()
        except np.linalg.LinAlgError as e:
            # Singularity or other numeric issues
            print(f"[DEBUG] OLS failed for combo={combo} with error={e}")
            continue

        # 4.4) Extract fitted params
        alpha = ols_model.params[0]
        betas = ols_model.params[1:]

        # 4.5) Extract preds, infer error
        fit_train = ols_model.fittedvalues  # alpha + X_train * betas
        spread_train = y_train - fit_train
        if len(spread_train) < 30:
            continue

        # 4.6) Apply adf and retrieve p val
        adf_res_train = adfuller(spread_train, maxlag=1, regression='c', autolag='AIC')
        pval_train = adf_res_train[1]

        # 4.7) If pval => significance, validate it out of sample to see if relationship still holds
        if pval_train < significance:
            # 4.7.1) Check for NaNs and data availability
            subdf_val = df_val[list(combo) + [target_col]].dropna()
            if len(subdf_val) < 30:
                continue
            
            # 4.7.2) Seperate X,y
            X_val = subdf_val[list(combo)]
            y_val = subdf_val[target_col]

            # 4.7.3) Make predictions with params found on train, infer error
            fit_val = alpha + np.sum(X_val.values * betas.values, axis=1)
            spread_val = y_val.values - fit_val
            
            # 4.7.4) Apply adf and retrieve p val
            adf_res_val = adfuller(spread_val, maxlag=1, regression='c', autolag='AIC')
            pval_val = adf_res_val[1]

            # 4.7.5) If pval => signficance, create sythentic ticker.
            if pval_val < significance:
                # Passed validation => create synthetic columns across entire df
                synth_name_pct = f"Close_pct_synth{synth_counter}"

                # We'll check if it doesn't already exist
                if synth_name_pct in df.columns:
                    synth_counter += 1
                    continue

                # We apply the same alpha/betas to entire df (train+val+unseen)
                subdf_full = df[list(combo)].dropna()
                if len(subdf_full) == 0:
                    continue

                X_full = subdf_full.values
                betas_aligned = betas.reindex(list(combo))
                fit_full = alpha + np.sum(X_full * betas_aligned.values, axis=1)

                # Check for Inf or NaN
                if not np.all(np.isfinite(fit_full)):
                    print(f"[DEBUG] combo={combo}: Synthetic series contains Inf/NaN, skipping.")
                    continue

                # Optional: check magnitude
                if np.nanmax(np.abs(fit_full)) > max_abs_value:
                    print(f"[DEBUG] combo={combo}: Synthetic series exceeds {max_abs_value}, skipping.")
                    continue

                # Create the new synthetic Close_pct column
                df.loc[subdf_full.index, synth_name_pct] = fit_full

                def get_ticker_name(c):
                    # c is something like "Close_pct_AAPL"
                    return c.split('_')[-1]

                used_tickers = [get_ticker_name(c) for c in combo]

                # For each data_type in:
                data_types = ['Close', 'Open', 'Volume']
                for dt in data_types:
                    sub_cols = []
                    for tk in used_tickers:
                        colname = f"{dt}_{tk}"
                        if colname in df.columns:
                            sub_cols.append(colname)

                    if len(sub_cols) == len(combo):
                        # We can build the synthetic for this data_type
                        subdf_type_full = df[sub_cols].dropna()
                        if len(subdf_type_full) == 0:
                            continue

                        X_type_full = subdf_type_full.values
                        synth_name_type = f"{dt}_synth{synth_counter}"

                        if X_type_full.shape[1] != len(betas_aligned):
                            print(f"[DEBUG] mismatch in columns vs betas. Skipping {dt}_synth.")
                            continue

                        fit_type_full = alpha + np.sum(X_type_full * betas_aligned.values, axis=1)

                        if not np.all(np.isfinite(fit_type_full)):
                            print(f"[DEBUG] {synth_name_type} has Inf/NaN. Skipping.")
                            continue

                        if np.nanmax(np.abs(fit_type_full)) > max_abs_value:
                            print(f"[DEBUG] {synth_name_type} exceeds {max_abs_value}. Skipping.")
                            continue

                        df.loc[subdf_type_full.index, synth_name_type] = fit_type_full

                # Save result
                coint_results[synth_name_pct] = {
                    'combo': combo,
                    'alpha': alpha,
                    'betas': betas.to_dict(),
                    'pval_train': pval_train,
                    'pval_val': pval_val,
                    'passed_validation': True
                }
                synth_counter += 1

    return df, coint_results


def calculate_technical_indicators(old_df, model, correlated_stock_tickers= None):
    """
    Computes technical indicators for the target ticker and all correlated tickers,
    then returns a new DataFrame with those columns added.
    """

    # 1) initial setup
    periods = [2, 4, 8, 16, 32, 64, 128, 256]
    new_df = pd.DataFrame(index=old_df.index)
    
    if correlated_stock_tickers == None: # needed for greedy info construction
        correlated_stock_tickers = set(
            col.split('_')[-1] 
            for col in old_df.columns 
            if '_' in col
        ) 

    # 2) calculate technical indicators for each ticker and period length
    for ticker in correlated_stock_tickers:
        for p in periods:
            close_pct = old_df[f'Close_pct_{ticker}']
            new_df[f'Close_pct_{ticker}'] = old_df[f'Close_pct_{ticker}'] 

            # 2.1) Calculate Moving Averages (SMA, EMA)
            new_df[f'MA_{p}_{ticker}'] = close_pct.rolling(p).mean()
            new_df[f'MA_{p}_{ticker}'] = close_pct.rolling(p).mean()
            new_df[f'EMA_{p}_{ticker}'] = close_pct.ewm(span=p, adjust=False).mean()
            new_df[f'EMA_{p}_{ticker}'] = close_pct.ewm(span=p, adjust=False).mean()

            # 2.2) Calculate Bollinger Bands
            rolling_mean = close_pct.rolling(p).mean()
            rolling_std = close_pct.rolling(p).std()
            new_df[f'BB_Upper_{p}_{ticker}'] = rolling_mean + 2 * rolling_std
            new_df[f'BB_Lower_{p}_{ticker}'] = rolling_mean - 2 * rolling_std

            # 2.3) Calculate MACD
            ema1 = close_pct.ewm(span=p).mean()
            ema2 = close_pct.ewm(span=2*p).mean()
            macd = ema1 - ema2
            new_df[f'MACD_{p}_{ticker}'] = macd
            new_df[f'MACD_Signal_{p}_{ticker}'] = macd.ewm(span=p).mean()
            new_df[f'MACD_Hist_{p}_{ticker}'] = new_df[f'MACD_{p}_{ticker}'] - new_df[f'MACD_Signal_{p}_{ticker}']

            # 2.4) Calculate RSI
            delta = close_pct.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(p).mean()
            avg_loss = loss.rolling(p).mean()
            rs = avg_gain / avg_loss
            new_df[f'RSI_{p}_{ticker}'] = 100 - (100 / (1 + rs))


            # 2.5) Calculate Momentum/ROC
            new_df[f'Momentum_{p}_{ticker}'] = close_pct.diff(p)
            new_df[f'ROC_{p}_{ticker}'] = close_pct.pct_change(p) * 100


    # 3) Drop NaNs and return data
    new_df = new_df.dropna()
    return new_df