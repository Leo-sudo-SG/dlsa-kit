"""
Fetch daily prices via yfinance, compute daily returns and per-asset characteristics,
run a simplified IPCA (copied from toy logic) on the resulting (T, N) time series,
and export residuals to CSVs compatible with the toy downstream pipeline.

This script is standalone and intentionally self-contained so you can test the
framework on real data with minimal changes.

Usage (from repo root):
    .\.venv\Scripts\python.exe fetch_yfinance_prepare.py --tickers AAPL MSFT AMZN GOOG TSLA --start 2020-01-01 --end 2024-12-31

Outputs (directory: yfinance_output/):
- residuals_insample_combined.csv
- residuals_oos_combined.csv
- gamma_mapping.csv
- factors_insample.csv
- factors_oos.csv

Notes:
- Requires packages: yfinance, pandas, numpy
    pip install yfinance pandas numpy
- The script splits data into train/test by `oos_fraction` (default 0.2).
- Characteristics created: 5-day momentum, 20-day vol, 60-day momentum (you can change).
- The IPCA implementation is intentionally minimal (suitable for small N).
"""

import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    raise ImportError("Missing dependency 'yfinance'. Install with: pip install yfinance")

# ----------------------------- IPCA helper functions (from toy logic) -----------------------------
from numpy.linalg import solve, pinv


def step_factor(R_list, I_list, Gamma):
    f_list = []
    residual_list = []
    for t, (R_t, I_t) in enumerate(zip(R_list, I_list)):
        Beta_t = I_t @ Gamma
        A = Beta_t.T @ Beta_t
        b = Beta_t.T @ R_t
        try:
            f_t = solve(A, b)
        except np.linalg.LinAlgError:
            f_t = pinv(A) @ b
        residuals_t = R_t - Beta_t @ f_t
        f_list.append(f_t)
        residual_list.append(residuals_t)
    return f_list, residual_list


def step_gamma(R_list, I_list, f_list, n_factors, n_characteristics):
    n_params = n_characteristics * n_factors
    A = np.zeros((n_params, n_params))
    b = np.zeros(n_params)
    for t, (R_t, I_t, f_t) in enumerate(zip(R_list, I_list, f_list)):
        tmp_t = np.kron(I_t, f_t)
        A += tmp_t.T @ tmp_t
        b += tmp_t.T @ R_t
    try:
        Gamma_vec = solve(A, b)
    except np.linalg.LinAlgError:
        Gamma_vec = pinv(A) @ b
    Gamma = Gamma_vec.reshape((n_characteristics, n_factors))
    return Gamma


def initial_gamma(R_list, I_list, n_factors, n_characteristics):
    X = np.zeros((len(R_list), n_characteristics))
    for t, (R_t, I_t) in enumerate(zip(R_list, I_list)):
        X[t, :] = I_t.T @ R_t / len(R_t)
    evals, evecs = np.linalg.eig(X.T @ X)
    idx = np.argsort(-evals)
    Gamma = evecs[:, idx[:n_factors]]
    return Gamma


def run_ipca(R_list, I_list, n_factors, n_characteristics, max_iter=50, tol=1e-4, verbose=False):
    Gamma = initial_gamma(R_list, I_list, n_factors, n_characteristics)
    Gamma_old = np.zeros_like(Gamma)
    for iteration in range(max_iter):
        f_list, residual_list = step_factor(R_list, I_list, Gamma)
        Gamma_new = step_gamma(R_list, I_list, f_list, n_factors, n_characteristics)
        dGamma = np.max(np.abs(Gamma_old - Gamma_new))
        if verbose:
            print(f"Iter {iteration}: max|dGamma|={dGamma:.6e}")
        Gamma_old = Gamma
        Gamma = Gamma_new
        if dGamma < tol:
            if verbose:
                print(f"Converged at iter {iteration}")
            break
    # Final factors and residuals
    f_list, residual_list = step_factor(R_list, I_list, Gamma)
    return Gamma, f_list, residual_list

# ----------------------------- Characteristic engineering -----------------------------

def compute_characteristics(returns_df):
    """
    Compute simple characteristics per asset per day.
    returns_df: DataFrame indexed by date, columns=tickers, values=daily returns
    Returns: list of I_t matrices (each shape N x L) aligned with R_list (same length)
    """
    # Choose lookback windows for characteristics
    mom_short = 5
    vol_window = 20
    mom_long = 60
    tickers = list(returns_df.columns)
    N = len(tickers)

    # Prepare arrays
    T = returns_df.shape[0]
    # We'll start from index = max lookback to ensure all features available
    start_idx = max(mom_long, vol_window)

    I_list = []
    R_list = []
    dates = []
    for idx in range(start_idx, T):
        window = returns_df.iloc[idx - mom_long: idx]
        # Characteristics per asset
        # 1) short momentum: cumulative return over last 5 days
        mom5 = (returns_df.iloc[idx - mom_short: idx] + 1).prod(axis=0) - 1
        # 2) vol: std over vol_window
        vol20 = returns_df.iloc[idx - vol_window: idx].std(axis=0).fillna(0)
        # 3) long momentum: cumulative 60-day
        mom60 = (window + 1).prod(axis=0) - 1
        # Stack into I_t (N x 3)
        I_t = np.vstack([mom5.values, vol20.values, mom60.values]).T
        # Target returns at this date (we use contemporaneous return)
        R_t = returns_df.iloc[idx].values
        I_list.append(I_t)
        R_list.append(R_t)
        dates.append(returns_df.index[idx])
    return R_list, I_list, dates

# ----------------------------- Exports -----------------------------

def export_combined_residuals_csv(residuals_list, dates, tickers, output_path):
    rows = []
    for i, res in enumerate(residuals_list):
        date = dates[i]
        for asset_idx, ticker in enumerate(tickers):
            rows.append({
                'day': date.strftime('%Y-%m-%d'),
                'asset': ticker,
                'asset_index': asset_idx,
                'residual': float(res[asset_idx])
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported residuals to {output_path}")


def export_gamma_csv(Gamma, output_path, char_names=None, factor_names=None):
    if char_names is None:
        char_names = [f"char_{i}" for i in range(Gamma.shape[0])]
    if factor_names is None:
        factor_names = [f"Factor_{i}" for i in range(Gamma.shape[1])]
    df = pd.DataFrame(Gamma, index=char_names, columns=factor_names)
    df.to_csv(output_path)
    print(f"Exported Gamma to {output_path}")


def export_factors_csv(factors_list, dates, output_path):
    df = pd.DataFrame(np.array(factors_list), index=[d.strftime('%Y-%m-%d') for d in dates])
    df.index.name = 'day'
    df.to_csv(output_path)
    print(f"Exported factors to {output_path}")

# ----------------------------- Main script -----------------------------

def main(tickers, start, end, oos_fraction=0.2, n_factors=2, out_dir='yfinance_output'):
    os.makedirs(out_dir, exist_ok=True)
    # Download data
    tickers_str = ' '.join(tickers)
    print(f"Downloading {len(tickers)} tickers: {tickers_str} from {start} to {end}")
    raw = yf.download(tickers, start=start, end=end, progress=False)

    # yfinance may return a MultiIndex DataFrame with levels like ('Adj Close', ticker)
    # or a simple DataFrame of adjusted prices when auto_adjust=True. Handle both.
    if isinstance(raw, pd.DataFrame):
        # MultiIndex columns (level 0 contains field names like 'Adj Close')
        if isinstance(raw.columns, pd.MultiIndex):
            if 'Adj Close' in raw.columns.get_level_values(0):
                data = raw['Adj Close']
            elif 'Close' in raw.columns.get_level_values(0):
                data = raw['Close']
            else:
                # Fall back to using the first top-level (usually OHLC) or the raw frame
                # This keeps behavior robust across yfinance versions.
                try:
                    data = raw.xs('Adj Close', level=0, axis=1)
                except Exception:
                    data = raw
        else:
            # Single-level columns: assume these are adjusted close prices already
            data = raw
    elif isinstance(raw, pd.Series):
        data = raw.to_frame()
    data = data.dropna(axis=0, how='any')
    returns = data.pct_change().dropna(how='any')
    tickers_clean = list(returns.columns)
    print(f"Downloaded and computed returns. Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

    # Compute R_list and I_list
    R_list_full, I_list_full, dates_full = compute_characteristics(returns)
    T = len(R_list_full)
    if T < 10:
        raise ValueError("Too few usable periods after characteristic construction. Try longer date range or fewer lookbacks.")

    # Split in-sample / OOS
    oos_start = int(T * (1 - oos_fraction))
    R_is = R_list_full[:oos_start]
    I_is = I_list_full[:oos_start]
    dates_is = dates_full[:oos_start]
    R_oos = R_list_full[oos_start:]
    I_oos = I_list_full[oos_start:]
    dates_oos = dates_full[oos_start:]

    print(f"Prepared {len(R_is)} train periods and {len(R_oos)} OOS periods")

    # Run IPCA
    n_char = I_is[0].shape[1]
    Gamma_est, f_list_est, residuals_in_sample = run_ipca(R_is, I_is, n_factors, n_char, max_iter=50, tol=1e-4, verbose=True)

    # Compute OOS residuals using learned Gamma
    oos_residuals = []
    oos_factors = []
    for R_o, I_o in zip(R_oos, I_oos):
        Beta_o = I_o @ Gamma_est
        A = Beta_o.T @ Beta_o
        b = Beta_o.T @ R_o
        try:
            f_o = solve(A, b)
        except np.linalg.LinAlgError:
            f_o = pinv(A) @ b
        res_o = R_o - Beta_o @ f_o
        oos_factors.append(f_o)
        oos_residuals.append(res_o)

    # Export combined CSVs
    export_combined_residuals_csv(residuals_in_sample, dates_is, tickers_clean, os.path.join(out_dir, 'residuals_insample_combined.csv'))
    export_combined_residuals_csv(oos_residuals, dates_oos, tickers_clean, os.path.join(out_dir, 'residuals_oos_combined.csv'))
    export_gamma_csv(Gamma_est, os.path.join(out_dir, 'gamma_mapping.csv'), char_names=['mom5','vol20','mom60'])
    export_factors_csv(f_list_est, dates_is, os.path.join(out_dir, 'factors_insample.csv'))
    export_factors_csv(oos_factors, dates_oos, os.path.join(out_dir, 'factors_oos.csv'))

    print("All exports completed. Use toy_downstream_pipeline.py to run downstream tests on these residuals.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch data from yfinance and prepare residuals via IPCA')
    parser.add_argument('--tickers', nargs='+', default=['AAPL','MSFT','AMZN','GOOG','TSLA'], help='List of tickers')
    parser.add_argument('--start', default=(datetime.now().year - 5).__str__() + '-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD')
    parser.add_argument('--oos_fraction', type=float, default=0.2, help='Fraction of samples reserved for OOS')
    parser.add_argument('--n_factors', type=int, default=2, help='Number of factors for IPCA')
    parser.add_argument('--out_dir', default='yfinance_output', help='Output directory for CSVs')
    args = parser.parse_args()

    main(args.tickers, args.start, args.end, args.oos_fraction, args.n_factors, args.out_dir)
