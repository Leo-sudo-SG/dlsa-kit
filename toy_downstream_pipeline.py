"""
Downstream Pipeline Script for DLSA-Kit Toy Example

This script demonstrates the complete end-to-end workflow:
1. Load exported residuals from toy_ipca_example.py (CSV)
2. Preprocess residuals into feature windows (cumsum, fourier, or OU)
3. Train a CNN/Transformer model to learn trading weights
4. Evaluate model on out-of-sample data
5. Compute backtest statistics (Sharpe, returns, turnover)

Usage:
    python toy_downstream_pipeline.py
    
Configuration:
    - lookback: Window size for preprocessing (days)
    - model_type: "CNNTransformer" or "RawFFN"
    - preprocess_func: "cumsum", "fourier", or "ou"
    - num_epochs: Training iterations
"""

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Import from dlsa-kit modules
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess_cumsum, preprocess_fourier, preprocess_ou
from models.CNNTransformer import CNNTransformer
from models.RawFFN import RawFFN

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'residuals_insample_csv': 'toy_ipca_output/residuals_insample_combined.csv',
    'residuals_oos_csv': 'toy_ipca_output/residuals_oos_combined.csv',
    'factors_insample_csv': 'toy_ipca_output/factors_insample.csv',
    'factors_oos_csv': 'toy_ipca_output/factors_oos.csv',
    
    # Preprocessing
        'lookback': 30,  # Default for daily data (days)
    'preprocess_func': 'cumsum',  # 'cumsum', 'fourier', 'ou'
    
    # Model
    'model_type': 'CNNTransformer',  # 'CNNTransformer', 'RawFFN'
    'random_seed': 42,
    'device': 'cpu',  # 'cpu' or 'cuda:0'
    
    # Training
        'num_epochs': 200,  # More epochs → longer training (adjust for daily runs)
        'batch_size': 32,  # Larger batch for daily-style training
    'learning_rate': 0.005,
    'optimizer_name': 'Adam',
    'early_stopping': False,
    'trans_cost': 0.0,  # Transaction cost per unit turnover
    'hold_cost': 0.0,   # Cost for short positions
    'objective': 'sharpe',  # 'sharpe', 'meanvar', 'sqrtMeanSharpe'
    
    # Model-specific hyperparameters (don't include lookback here - it's passed separately)
    'cnn_transformer_params': {
        'filter_numbers': [1, 4],  # Smaller for toy
        'attention_heads': 1,
        'hidden_units_factor': 2,  # Use factor, NOT hidden_units
        'normalization_conv': False,  # Disable for small lookback
        'filter_size': 1,  # Must be <= lookback
        'dropout': 0.1,
        'use_convolution': True,
        'use_transformer': True,
    },
    'raw_ffn_params': {
        'layer_sizes': [32, 16],
        'dropout': 0.1,
    },
    
    # Output
    'output_dir': 'toy_downstream_output',
    'save_plots': True,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_residuals_from_csv(insample_csv, oos_csv):
    """
    Load residuals from CSV exports and convert to numpy arrays.
    
    Args:
      insample_csv: Path to residuals_insample_combined.csv
      oos_csv: Path to residuals_oos_combined.csv
      
    Returns:
      residuals_is: numpy array (T_is, N) - in-sample residuals
      residuals_oos: numpy array (T_oos, N) - out-of-sample residuals
    """
    logger.info(f"Loading residuals from CSV files...")
    
    # Load in-sample residuals (support 'month' or 'day'/'time' index)
    df_is = pd.read_csv(insample_csv)
    # Determine index column
    if 'month' in df_is.columns:
        idx_col = 'month'
    elif 'day' in df_is.columns:
        idx_col = 'day'
    elif 'time' in df_is.columns:
        idx_col = 'time'
    else:
        # Fallback to first column if nothing matches
        idx_col = df_is.columns[0]
    residuals_is = df_is.pivot(index=idx_col, columns='asset', values='residual').values
    logger.info(f"In-sample residuals shape: {residuals_is.shape}")
    
    # Load OOS residuals
    df_oos = pd.read_csv(oos_csv)
    if 'month' in df_oos.columns:
        idx_col_oos = 'month'
    elif 'day' in df_oos.columns:
        idx_col_oos = 'day'
    elif 'time' in df_oos.columns:
        idx_col_oos = 'time'
    else:
        idx_col_oos = df_oos.columns[0]
    residuals_oos = df_oos.pivot(index=idx_col_oos, columns='asset', values='residual').values
    logger.info(f"OOS residuals shape: {residuals_oos.shape}")
    
    return residuals_is, residuals_oos


def preprocess_residuals(data, lookback, preprocess_func_name):
    """
    Preprocess residuals into feature windows.
    
    Args:
      data: numpy array (T, N) - residuals
      lookback: int - window size
      preprocess_func_name: str - 'cumsum', 'fourier', or 'ou'
      
    Returns:
      windows: numpy array (T-lookback, N, signal_length)
      idxs_selected: torch.Tensor (T-lookback, N) - boolean mask of valid assets
    """
    logger.info(f"Preprocessing with {preprocess_func_name}, lookback={lookback}")
    
    if preprocess_func_name == 'cumsum':
        windows, idxs_selected = preprocess_cumsum(data, lookback)
    elif preprocess_func_name == 'fourier':
        windows, idxs_selected = preprocess_fourier(data, lookback)
    elif preprocess_func_name == 'ou':
        windows, idxs_selected = preprocess_ou(data, lookback)
    else:
        raise ValueError(f"Unknown preprocessing function: {preprocess_func_name}")
    
    logger.info(f"Feature windows shape: {windows.shape}")
    logger.info(f"Valid assets mask shape: {idxs_selected.shape}")
    
    return windows, idxs_selected


# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def create_model(model_type, config, lookback, logdir):
    """
    Initialize the trading model.
    
    Args:
      model_type: str - 'CNNTransformer' or 'RawFFN'
      config: dict - Configuration dict
      lookback: int - Lookback window size
      logdir: str - Directory for checkpoints
      
    Returns:
      model: torch.nn.Module instance
    """
    logger.info(f"Creating {model_type} model...")
    
    if model_type == 'CNNTransformer':
        model = CNNTransformer(
            logdir=logdir,
            random_seed=config['random_seed'],
            lookback=lookback,
            device=config['device'],
            hidden_units=None,  # Use hidden_units_factor instead
            **config['cnn_transformer_params']
        )
    elif model_type == 'RawFFN':
        model = RawFFN(
            logdir=logdir,
            random_seed=config['random_seed'],
            lookback=lookback,
            device=config['device'],
            **config['raw_ffn_params']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    return model


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_model(model, windows_train, idxs_selected_train, residuals_train,
                windows_oos, idxs_selected_oos, residuals_oos,
                config):
    """
    Train the model on in-sample data and evaluate on OOS.
    
    Args:
      model: torch.nn.Module
      windows_train: numpy array (T-lookback, N, signal_length)
      idxs_selected_train: torch.Tensor (T-lookback, N)
      residuals_train: numpy array (T, N) - for computing returns
      windows_oos: numpy array (T-lookback, N, signal_length)
      idxs_selected_oos: torch.Tensor (T-lookback, N)
      residuals_oos: numpy array (T, N) - for computing returns
      config: dict - Configuration
      
    Returns:
      model: Trained model
      train_results: dict with training metrics
      oos_results: dict with OOS metrics
    """
    logger.info("="*70)
    logger.info("TRAINING MODEL")
    logger.info("="*70 + "\n")
    
    device = torch.device(config['device'])
    model.to(device)
    model.train()
    
    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer_name'])(
        model.parameters(), lr=config['learning_rate']
    )
    
    T_train, N = windows_train.shape[0], windows_train.shape[1]
    T_oos = windows_oos.shape[0]
    
    # Select assets with sufficient data
    assets_to_trade = np.count_nonzero(residuals_train, axis=0) >= config['lookback']
    logger.info(f"Assets to trade (OOS): {np.sum(assets_to_trade)}/{N}")
    
    train_results = {
        'epochs': [],
        'train_loss': [],
        'train_sharpe': [],
        'oos_sharpe': [],
        'oos_returns': None,
        'oos_weights': None,
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        epoch_returns = []
        
        # Minibatch training
        for batch_idx in range(0, T_train, config['batch_size']):
            batch_end = min(batch_idx + config['batch_size'], T_train)
            batch_size_actual = batch_end - batch_idx
            
            # Get valid assets for this batch
            batch_idxs = idxs_selected_train[batch_idx:batch_end, :]
            batch_windows = windows_train[batch_idx:batch_end, :, :]
            
            # Prepare input and target
            weights = torch.zeros((batch_size_actual, N), device=device)
            
            for i in range(batch_size_actual):
                valid_assets = batch_idxs[i, :].numpy()
                if np.sum(valid_assets) > 0:
                    # Input: valid assets for this time step
                    X = torch.tensor(
                        batch_windows[i, valid_assets, :],
                        dtype=torch.float32,
                        device=device
                    )
                    
                    # Forward pass
                    w = model(X)
                    
                    # Place weights in correct positions
                    weights[i, valid_assets] = w
            
            # Normalize weights
            abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
            weights = weights / (abs_sum + 1e-8)
            
            # Compute returns
            batch_residuals = residuals_train[
                config['lookback'] + batch_idx:config['lookback'] + batch_end, :
            ]
            rets = torch.sum(
                weights * torch.tensor(batch_residuals, dtype=torch.float32, device=device),
                axis=1
            )
            
            # Compute loss
            mean_ret = torch.mean(rets)
            # Use population std (unbiased=False) so std is defined for small batches
            std_ret = torch.std(rets, unbiased=False)
            
            if config['objective'] == 'sharpe':
                loss = -mean_ret / (std_ret + 1e-8)
            elif config['objective'] == 'meanvar':
                loss = -mean_ret * 252 + std_ret * 15.9
            elif config['objective'] == 'sqrtMeanSharpe':
                loss = -torch.sign(mean_ret) * torch.sqrt(torch.abs(mean_ret)) / (std_ret + 1e-8)
            else:
                raise ValueError(f"Unknown objective: {config['objective']}")
            
            # Backward pass (skip if loss is NaN/Inf)
            if not torch.isfinite(loss):
                logger.warning(f"Skipping batch {batch_idx}-{batch_end} due to non-finite loss: {loss}")
            else:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_returns.append(rets.detach().cpu().numpy())
        
        # Log epoch progress
        if len(epoch_returns) > 0:
            epoch_returns = np.concatenate(epoch_returns)
            # Use population std for epoch-level calculation as well
            epoch_sharpe = np.mean(epoch_returns) / (np.std(epoch_returns) + 1e-8) * np.sqrt(252)
        else:
            epoch_returns = np.array([])
            epoch_sharpe = float('nan')

        train_results['epochs'].append(epoch)
        # If loss is finite use it, otherwise append nan
        try:
            loss_value = loss.item()
            if not np.isfinite(loss_value):
                loss_value = float('nan')
        except Exception:
            loss_value = float('nan')
        train_results['train_loss'].append(loss_value)
        train_results['train_sharpe'].append(epoch_sharpe)
        
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}/{config['num_epochs']}: "
                       f"Loss={loss.item():.4f}, Sharpe={epoch_sharpe:.4f}")
    
    logger.info("\nTraining complete!\n")
    
    # Evaluate on OOS data
    logger.info("="*70)
    logger.info("OUT-OF-SAMPLE EVALUATION")
    logger.info("="*70 + "\n")
    
    model.eval()
    oos_weights = torch.zeros((T_oos, N), device=device)
    oos_sharpe = 0.0
    
    with torch.no_grad():
        for t in range(T_oos):
            valid_assets = idxs_selected_oos[t, :].numpy()
            if np.sum(valid_assets) > 0:
                X = torch.tensor(
                    windows_oos[t, valid_assets, :],
                    dtype=torch.float32,
                    device=device
                )
                w = model(X)
                oos_weights[t, valid_assets] = w
        
        # Normalize weights
        abs_sum = torch.sum(torch.abs(oos_weights), axis=1, keepdim=True)
        oos_weights = oos_weights / (abs_sum + 1e-8)
        
        # Compute OOS returns (only if we have OOS windows)
        if oos_weights.shape[0] > 0:
            oos_residuals = residuals_oos[config['lookback']:, :]
            oos_rets = torch.sum(
                oos_weights * torch.tensor(oos_residuals, dtype=torch.float32, device=device),
                axis=1
            )

            oos_mean_ret = torch.mean(oos_rets)
            # Use population std for small sample sizes
            oos_std_ret = torch.std(oos_rets, unbiased=False)
            oos_sharpe = (oos_mean_ret / (oos_std_ret + 1e-8) * np.sqrt(252)).item()
        else:
            oos_rets = torch.tensor([], dtype=torch.float32)
            oos_mean_ret = torch.tensor(float('nan'))
            oos_std_ret = torch.tensor(float('nan'))
            oos_sharpe = float('nan')
        
        train_results['oos_sharpe'] = oos_sharpe
        train_results['oos_returns'] = oos_rets.cpu().numpy()
        train_results['oos_weights'] = oos_weights.cpu().numpy()
    
    logger.info(f"OOS Sharpe Ratio: {oos_sharpe:.4f}")
    logger.info(f"OOS Mean Return (daily): {oos_mean_ret.item()*252:.4f}")
    logger.info(f"OOS Std Dev (annualized): {oos_std_ret.item()*np.sqrt(252):.4f}")
    logger.info(f"OOS Returns shape: {train_results['oos_returns'].shape}\n")
    
    return model, train_results


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def compute_backtest_statistics(returns, weights, residuals, lookback, config):
    """
    Compute detailed backtest statistics.
    
    Args:
      returns: numpy array (T,) - daily returns
      weights: numpy array (T, N) - portfolio weights
      residuals: numpy array (T, N) - asset residuals
      lookback: int - lookback window
      config: dict - Configuration
      
    Returns:
      stats: dict with backtest statistics
    """
    logger.info("Computing backtest statistics...")
    
    # Handle empty returns
    if len(returns) == 0:
        logger.warning("No returns to analyze (OOS period too short)")
        return {
            'mean_return_daily': np.nan,
            'mean_return_annual': np.nan,
            'std_return_annual': np.nan,
            'sharpe_ratio': np.nan,
            'cumulative_return': np.nan,
            'max_drawdown': np.nan,
            'avg_turnover': 0.0,
            'avg_short_proportion': 0.0,
            'positive_months': 0,
            'total_months': 0,
            'win_rate_months': 0.0,
        }
    
    # Basic stats
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)
    cum_ret = np.cumprod(1 + returns)[-1] - 1
    
    # Turnover
    if len(weights) > 1:
        turnover = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
        avg_turnover = np.mean(turnover)
    else:
        avg_turnover = 0.0
    
    # Short proportion
    short_prop = np.sum(np.abs(np.minimum(weights, 0)), axis=1)
    avg_short_prop = np.mean(short_prop)
    
    # Drawdown
    cum_rets = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_rets)
    drawdown = (cum_rets - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Monthly returns (approximate)
    if len(returns) >= 20:
        monthly_returns = np.array([
            np.mean(returns[i:i+20]) for i in range(0, len(returns) - 20, 20)
        ])
        positive_months = np.sum(monthly_returns > 0)
        total_months = len(monthly_returns)
    else:
        positive_months = 0
        total_months = 1
    
    stats = {
        'mean_return_daily': mean_ret,
        'mean_return_annual': mean_ret * 252,
        'std_return_annual': std_ret * np.sqrt(252),
        'sharpe_ratio': sharpe,
        'cumulative_return': cum_ret,
        'max_drawdown': max_drawdown,
        'avg_turnover': avg_turnover,
        'avg_short_proportion': avg_short_prop,
        'positive_months': positive_months,
        'total_months': total_months,
        'win_rate_months': positive_months / max(total_months, 1),
    }
    
    return stats


def print_results(train_results, oos_stats, config):
    """Pretty-print results summary."""
    logger.info("="*70)
    logger.info("FINAL RESULTS")
    logger.info("="*70 + "\n")
    
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: {config['model_type']}")
    logger.info(f"  Preprocessing: {config['preprocess_func']}")
    logger.info(f"  Lookback: {config['lookback']}")
    logger.info(f"  Epochs: {config['num_epochs']}")
    logger.info(f"  Objective: {config['objective']}\n")
    
    logger.info(f"In-Sample Training:")
    logger.info(f"  Final Sharpe: {train_results['train_sharpe'][-1]:.4f}")
    logger.info(f"  Peak Sharpe: {max(train_results['train_sharpe']):.4f}\n")
    
    logger.info(f"Out-of-Sample Performance:")
    logger.info(f"  Sharpe Ratio: {oos_stats['sharpe_ratio']:.4f}")
    logger.info(f"  Annual Return: {oos_stats['mean_return_annual']:.4f}")
    logger.info(f"  Annual Std Dev: {oos_stats['std_return_annual']:.4f}")
    logger.info(f"  Cumulative Return: {oos_stats['cumulative_return']:.4f}")
    logger.info(f"  Max Drawdown: {oos_stats['max_drawdown']:.4f}")
    logger.info(f"  Avg Turnover: {oos_stats['avg_turnover']:.4f}")
    logger.info(f"  Avg Short Proportion: {oos_stats['avg_short_proportion']:.4f}")
    logger.info(f"  Win Rate (months): {oos_stats['win_rate_months']:.2%}\n")


def save_results(train_results, oos_stats, config, output_dir):
    """Save results to CSV and plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training history
    df_train = pd.DataFrame({
        'epoch': train_results['epochs'],
        'train_loss': train_results['train_loss'],
        'train_sharpe': train_results['train_sharpe'],
    })
    df_train.to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    logger.info(f"Saved training history to {output_dir}/training_history.csv")
    
    # Save OOS returns
    df_returns = pd.DataFrame({
        'oos_returns': train_results['oos_returns'],
        'cumulative_returns': np.cumprod(1 + train_results['oos_returns']),
    })
    df_returns.to_csv(os.path.join(output_dir, 'oos_returns.csv'), index=False)
    logger.info(f"Saved OOS returns to {output_dir}/oos_returns.csv")
    
    # Save statistics
    df_stats = pd.DataFrame([oos_stats])
    df_stats.to_csv(os.path.join(output_dir, 'backtest_statistics.csv'), index=False)
    logger.info(f"Saved backtest statistics to {output_dir}/backtest_statistics.csv")
    
    # Save weights
    np.save(os.path.join(output_dir, 'oos_weights.npy'), train_results['oos_weights'])
    logger.info(f"Saved OOS weights to {output_dir}/oos_weights.npy")
    
    if config['save_plots']:
        import matplotlib.pyplot as plt
        
        # Training history plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].plot(train_results['epochs'], train_results['train_loss'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(train_results['epochs'], train_results['train_sharpe'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Sharpe Ratio (annualized)')
        axes[1].set_title('In-Sample Sharpe Ratio')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=100)
        logger.info(f"Saved training plot to {output_dir}/training_history.png")
        plt.close()
        
        # Returns plot
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        cum_returns = np.cumprod(1 + train_results['oos_returns'])
        axes[0].plot(cum_returns)
        axes[0].set_ylabel('Cumulative Returns')
        axes[0].set_title('Out-of-Sample Cumulative Returns')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(train_results['oos_returns'])
        axes[1].set_xlabel('Days')
        axes[1].set_ylabel('Daily Returns')
        axes[1].set_title('Out-of-Sample Daily Returns')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'oos_returns.png'), dpi=100)
        logger.info(f"Saved returns plot to {output_dir}/oos_returns.png")
        plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run the complete downstream pipeline."""
    logger.info("="*70)
    logger.info("DLSA-KIT TOY DOWNSTREAM PIPELINE")
    logger.info("="*70 + "\n")
    
    start_time = time.time()
    
    # Create output directory
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load residuals
    logger.info("STEP 1: Load Residuals from CSV")
    logger.info("-"*70 + "\n")
    residuals_is, residuals_oos = load_residuals_from_csv(
        CONFIG['residuals_insample_csv'],
        CONFIG['residuals_oos_csv']
    )
    
    # Determine lookback to use (ensure it's smaller than available training length)
    lookback_requested = CONFIG['lookback']
    if residuals_is.shape[0] <= 1:
        raise ValueError("Not enough in-sample data to form windows")

    # Choose a lookback that respects both in-sample and OOS lengths.
    # If possible, pick lookback <= min(in_sample-1, oos-1) so there is at least one OOS window.
    max_lookback_is = max(1, residuals_is.shape[0] - 1)
    if residuals_oos is not None and residuals_oos.shape[0] > 1:
        max_lookback_oos = max(1, residuals_oos.shape[0] - 1)
        lookback_used = min(lookback_requested, max_lookback_is, max_lookback_oos)
    else:
        lookback_used = min(lookback_requested, max_lookback_is)

    if lookback_used != lookback_requested:
        logger.warning(f"Requested lookback={lookback_requested} reduced to lookback_used={lookback_used} to match data length")

    # Step 2: Preprocess
    logger.info("\nSTEP 2: Preprocess Residuals")
    logger.info("-"*70 + "\n")
    windows_is, idxs_is = preprocess_residuals(
        residuals_is, lookback_used, CONFIG['preprocess_func']
    )

    # Preprocess OOS only if there is enough OOS data for the chosen lookback
    if residuals_oos.shape[0] > lookback_used:
        windows_oos, idxs_oos = preprocess_residuals(
            residuals_oos, lookback_used, CONFIG['preprocess_func']
        )
    else:
        logger.warning(f"Not enough OOS periods ({residuals_oos.shape[0]}) for lookback={lookback_used}; skipping OOS preprocessing")
        N = residuals_is.shape[1]
        windows_oos = np.zeros((0, N, lookback_used), dtype=np.float32)
        idxs_oos = torch.zeros((0, N), dtype=torch.bool)
    
    # Step 3: Create model
    logger.info("\nSTEP 3: Initialize Model")
    logger.info("-"*70 + "\n")
    model = create_model(
        CONFIG['model_type'],
        CONFIG,
        lookback_used,
        output_dir
    )
    
    # Step 4: Train and evaluate
    logger.info("\nSTEP 4: Train and Evaluate")
    logger.info("-"*70 + "\n")
    # Pass a local copy of CONFIG with the adjusted lookback to training
    local_config = CONFIG.copy()
    local_config['lookback'] = lookback_used

    model, train_results = train_model(
        model, windows_is, idxs_is, residuals_is,
        windows_oos, idxs_oos, residuals_oos,
        local_config
    )
    
    # Step 5: Compute statistics
    logger.info("\nSTEP 5: Compute Statistics")
    logger.info("-"*70 + "\n")
    oos_stats = compute_backtest_statistics(
        train_results['oos_returns'],
        train_results['oos_weights'],
        residuals_oos,
        lookback_used,
        CONFIG
    )
    
    # Step 6: Print and save results
    logger.info("\nSTEP 6: Results")
    logger.info("-"*70 + "\n")
    print_results(train_results, oos_stats, CONFIG)
    
    logger.info("\nSTEP 7: Save Results")
    logger.info("-"*70 + "\n")
    save_results(train_results, oos_stats, CONFIG, output_dir)
    
    elapsed = time.time() - start_time
    logger.info(f"\n✓ Pipeline completed in {elapsed:.2f}s")
    logger.info(f"✓ Results saved to {output_dir}/\n")


if __name__ == "__main__":
    main()
