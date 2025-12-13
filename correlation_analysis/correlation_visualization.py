#!/usr/bin/env python3
"""
Visualization tool for historic correlations between stock returns/residuals.
Uses window-based approach for correlation analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


def load_residuals_data(n_factors: int, base_path: str = "residuals/ipca_normalized") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load IPCA residuals data for specified number of factors.

    Args:
        n_factors: Number of factors (0 for raw returns, 5 for 5-factor residuals)
        base_path: Path to residuals directory

    Returns:
        Tuple of (data_array, mask_array)
    """
    filename = f"IPCA_DailyOOSresiduals_{n_factors}_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy"

    # Try .gz first, then uncompressed in directory
    gz_path = os.path.join(base_path, filename + ".gz")
    npy_dir_path = os.path.join(base_path, filename, filename)

    if os.path.exists(gz_path):
        print(f"Loading {gz_path}")
        # For gz files, we need to handle them differently
        import gzip
        with gzip.open(gz_path, 'rb') as f:
            data = np.load(f)
    elif os.path.exists(npy_dir_path):
        print(f"Loading {npy_dir_path}")
        data = np.load(npy_dir_path)
    else:
        raise FileNotFoundError(f"Could not find {filename} in expected locations")

    # Load mask
    mask_path = os.path.join(base_path, "..", "superMask.npy")
    mask = np.load(mask_path)

    print(f"Loaded data shape: {data.shape}")
    print(f"Loaded mask shape: {mask.shape}")

    return data, mask


def select_asset_subset(data: np.ndarray, mask: np.ndarray, n_assets: Optional[int] = None,
                       random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select a subset of assets for analysis.

    Args:
        data: Full data array (T x N)
        mask: Boolean mask for valid assets
        n_assets: Number of assets to select (None for all)
        random_seed: Random seed for reproducible selection

    Returns:
        Tuple of (subset_data, selected_indices)
    """
    valid_assets = np.where(mask)[0]

    if n_assets is None or n_assets >= len(valid_assets):
        selected_indices = valid_assets
    else:
        np.random.seed(random_seed)
        selected_indices = np.random.choice(valid_assets, size=n_assets, replace=False)
        selected_indices = np.sort(selected_indices)

    subset_data = data[:, selected_indices]

    print(f"Selected {len(selected_indices)} assets out of {len(valid_assets)} valid assets")
    print(f"Subset data shape: {subset_data.shape}")

    return subset_data, selected_indices


def rolling_correlation_matrix(data: np.ndarray, window_size: int = 252,
                              step_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate rolling correlation matrices over time windows.

    Args:
        data: Time series data (T x N)
        window_size: Size of rolling window (default 252 trading days ~ 1 year)
        step_size: Step size for sliding window (default 21 ~ 1 month)

    Returns:
        Tuple of (correlation_matrices, time_indices)
        correlation_matrices: (n_windows, N, N) array
        time_indices: Array of starting time indices for each window
    """
    T, N = data.shape
    n_windows = (T - window_size) // step_size + 1

    correlation_matrices = np.zeros((n_windows, N, N))
    time_indices = np.arange(0, n_windows) * step_size + window_size

    print(f"Calculating {n_windows} correlation matrices with window size {window_size}")

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        window_data = data[start_idx:end_idx]

        # Calculate correlation matrix for this window
        # Handle NaN values by using nan-safe correlation
        corr_matrix = np.corrcoef(window_data.T)

        # Fill any NaN values with 0 (can happen with constant time series)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        correlation_matrices[i] = corr_matrix

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{n_windows} windows")

    return correlation_matrices, time_indices


def plot_correlation_evolution(corr_matrices: np.ndarray, time_indices: np.ndarray,
                              asset_names: Optional[list] = None, title: str = "Correlation Evolution",
                              figsize: Tuple[int, int] = (15, 10)):
    """
    Plot the evolution of correlations over time.

    Args:
        corr_matrices: Array of correlation matrices (n_windows, N, N)
        time_indices: Time indices for each correlation matrix
        asset_names: Optional names for assets
        title: Plot title
        figsize: Figure size
    """
    n_windows, N, _ = corr_matrices.shape

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(N)]

    # Calculate average correlation over time for each pair
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Plot 1: Average correlation over time
    mean_corr_over_time = np.mean(corr_matrices, axis=(1, 2))
    axes[0, 0].plot(time_indices, mean_corr_over_time, 'b-', linewidth=2)
    axes[0, 0].set_title('Average Correlation Over Time')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Average Correlation')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Correlation heatmap for the most recent window
    recent_corr = corr_matrices[-1]
    sns.heatmap(recent_corr, annot=False, cmap='RdYlBu_r', center=0,
                xticklabels=asset_names, yticklabels=asset_names, ax=axes[0, 1])
    axes[0, 1].set_title(f'Correlation Matrix (Most Recent Window)')

    # Plot 3: Correlation distribution over time
    corr_values_over_time = corr_matrices.reshape(n_windows, -1)
    axes[1, 0].boxplot(corr_values_over_time.T, showfliers=False)
    axes[1, 0].set_title('Correlation Distribution Over Time')
    axes[1, 0].set_xlabel('Time Window')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Correlation volatility (standard deviation over time)
    corr_std_over_time = np.std(corr_values_over_time, axis=1)
    axes[1, 1].plot(time_indices, corr_std_over_time, 'r-', linewidth=2)
    axes[1, 1].set_title('Correlation Volatility Over Time')
    axes[1, 1].set_xlabel('Time Index')
    axes[1, 1].set_ylabel('Correlation Std Dev')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def compare_returns_vs_residuals(returns_corr: np.ndarray, residuals_corr: np.ndarray,
                                time_indices: np.ndarray, asset_names: Optional[list] = None,
                                figsize: Tuple[int, int] = (15, 10)):
    """
    Compare correlations between raw returns and residuals.

    Args:
        returns_corr: Correlation matrices for returns (n_windows, N, N)
        residuals_corr: Correlation matrices for residuals (n_windows, N, N)
        time_indices: Time indices
        asset_names: Optional asset names
        figsize: Figure size
    """
    n_windows, N, _ = returns_corr.shape

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(N)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Returns vs Residuals Correlation Comparison', fontsize=16)

    # Plot 1: Average correlation comparison
    returns_avg = np.mean(returns_corr, axis=(1, 2))
    residuals_avg = np.mean(residuals_corr, axis=(1, 2))

    axes[0, 0].plot(time_indices, returns_avg, 'b-', label='Returns', linewidth=2)
    axes[0, 0].plot(time_indices, residuals_avg, 'r-', label='Residuals', linewidth=2)
    axes[0, 0].set_title('Average Correlation: Returns vs Residuals')
    axes[0, 0].set_xlabel('Time Index')
    axes[0, 0].set_ylabel('Average Correlation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Difference in average correlation
    diff_avg = returns_avg - residuals_avg
    axes[0, 1].plot(time_indices, diff_avg, 'g-', linewidth=2)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title('Correlation Difference (Returns - Residuals)')
    axes[0, 1].set_xlabel('Time Index')
    axes[0, 1].set_ylabel('Correlation Difference')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Correlation heatmap comparison for most recent window
    returns_recent = returns_corr[-1]
    residuals_recent = residuals_corr[-1]
    diff_recent = returns_recent - residuals_recent

    sns.heatmap(diff_recent, annot=False, cmap='RdYlBu_r', center=0,
                xticklabels=asset_names, yticklabels=asset_names, ax=axes[1, 0])
    axes[1, 0].set_title('Correlation Difference (Most Recent)')

    # Plot 4: Distribution comparison
    returns_values = returns_corr.reshape(n_windows, -1)
    residuals_values = residuals_corr.reshape(n_windows, -1)

    axes[1, 1].hist(returns_values[-1], alpha=0.7, label='Returns', bins=20, density=True)
    axes[1, 1].hist(residuals_values[-1], alpha=0.7, label='Residuals', bins=20, density=True)
    axes[1, 1].set_title('Correlation Distribution (Most Recent)')
    axes[1, 1].set_xlabel('Correlation')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()

    plt.tight_layout()
    return fig


def plot_correlation_network(corr_matrix: np.ndarray, asset_names: Optional[list] = None,
                           threshold: float = 0.3, title: str = "Correlation Network",
                           figsize: Tuple[int, int] = (12, 8)):
    """
    Plot correlation network showing strong correlations.

    Args:
        corr_matrix: Correlation matrix (N x N)
        asset_names: Optional asset names
        threshold: Minimum absolute correlation to show
        title: Plot title
        figsize: Figure size
    """
    N = corr_matrix.shape[0]

    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(N)]

    fig, ax = plt.subplots(figsize=figsize)

    # Create adjacency matrix based on threshold
    adj_matrix = np.abs(corr_matrix) > threshold
    np.fill_diagonal(adj_matrix, False)  # Remove self-correlations

    # Plot heatmap with threshold
    masked_corr = corr_matrix.copy()
    masked_corr[np.abs(masked_corr) < threshold] = 0

    sns.heatmap(masked_corr, annot=False, cmap='RdYlBu_r', center=0,
                xticklabels=asset_names, yticklabels=asset_names, ax=ax)
    ax.set_title(f'{title} (|corr| > {threshold})')

    # Add some statistics
    n_connections = np.sum(adj_matrix) // 2  # Divide by 2 since matrix is symmetric
    avg_corr = np.mean(np.abs(corr_matrix[np.abs(corr_matrix) > threshold]))
    ax.text(0.02, 0.98, f'Connections: {n_connections}\nAvg |corr|: {avg_corr:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize historic correlations between stock returns/residuals')
    parser.add_argument('--n-assets', type=int, default=10,
                       help='Number of assets to analyze (default: 10)')
    parser.add_argument('--window-size', type=int, default=252,
                       help='Rolling window size in days (default: 252)')
    parser.add_argument('--step-size', type=int, default=21,
                       help='Step size for sliding window in days (default: 21)')
    parser.add_argument('--correlation-threshold', type=float, default=0.3,
                       help='Threshold for correlation network visualization (default: 0.3)')
    parser.add_argument('--output-dir', type=str, default='correlation_plots',
                       help='Output directory for plots (default: correlation_plots)')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')

    args = parser.parse_args()

    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")

    # Load returns (0 factors) and residuals (5 factors)
    returns_data, mask = load_residuals_data(0)
    residuals_data, _ = load_residuals_data(5)

    # Select subset of assets
    returns_subset, selected_indices = select_asset_subset(returns_data, mask, args.n_assets)
    residuals_subset, _ = select_asset_subset(residuals_data, mask, args.n_assets, random_seed=42)

    # Use last portion of data (like in the exploration notebook)
    T_start = max(0, returns_subset.shape[0] - 1200)  # Last ~5 years
    returns_subset = returns_subset[T_start:, :]
    residuals_subset = residuals_subset[T_start:, :]

    print(f"Using data from index {T_start} onwards, shape: {returns_subset.shape}")

    # Calculate rolling correlations
    print("\nCalculating rolling correlations for returns...")
    returns_corr, time_indices = rolling_correlation_matrix(
        returns_subset, args.window_size, args.step_size
    )

    print("\nCalculating rolling correlations for residuals...")
    residuals_corr, _ = rolling_correlation_matrix(
        residuals_subset, args.window_size, args.step_size
    )

    # Create visualizations
    print("\nCreating visualizations...")

    # Asset names
    asset_names = [f"Asset_{i+1}" for i in range(args.n_assets)]

    # Plot correlation evolution for returns
    fig1 = plot_correlation_evolution(
        returns_corr, time_indices, asset_names,
        title="Returns Correlation Evolution"
    )

    # Plot correlation evolution for residuals
    fig2 = plot_correlation_evolution(
        residuals_corr, time_indices, asset_names,
        title="Residuals Correlation Evolution"
    )

    # Plot comparison
    fig3 = compare_returns_vs_residuals(
        returns_corr, residuals_corr, time_indices, asset_names
    )

    # Plot correlation networks
    fig4 = plot_correlation_network(
        returns_corr[-1], asset_names, args.correlation_threshold,
        title="Returns Correlation Network"
    )

    fig5 = plot_correlation_network(
        residuals_corr[-1], asset_names, args.correlation_threshold,
        title="Residuals Correlation Network"
    )

    # Show plots
    plt.show()

    # Save plots if requested
    if args.save_plots:
        fig1.savefig(os.path.join(args.output_dir, 'returns_correlation_evolution.png'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(args.output_dir, 'residuals_correlation_evolution.png'), dpi=300, bbox_inches='tight')
        fig3.savefig(os.path.join(args.output_dir, 'returns_vs_residuals_comparison.png'), dpi=300, bbox_inches='tight')
        fig4.savefig(os.path.join(args.output_dir, 'returns_correlation_network.png'), dpi=300, bbox_inches='tight')
        fig5.savefig(os.path.join(args.output_dir, 'residuals_correlation_network.png'), dpi=300, bbox_inches='tight')
        print(f"\nPlots saved to {args.output_dir}/")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
