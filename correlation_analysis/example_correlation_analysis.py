#!/usr/bin/env python3
"""
Example usage of the correlation visualization tool.
This script demonstrates how to analyze correlations between stock returns and residuals.
"""

import os
import sys

# Add the current directory to the path for importing correlation_visualization
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from correlation_visualization import (
    load_residuals_data,
    select_asset_subset,
    rolling_correlation_matrix,
    plot_correlation_evolution,
    compare_returns_vs_residuals,
    plot_correlation_network
)
import matplotlib.pyplot as plt
import numpy as np

def example_small_analysis():
    """
    Example analysis with a small subset of assets for quick testing.
    """
    print("=== Correlation Analysis Example ===\n")

    # Load data
    print("1. Loading data...")
    returns_data, mask = load_residuals_data(0)  # Raw returns
    residuals_data, _ = load_residuals_data(5)   # 5-factor residuals

    # Select small subset
    print("\n2. Selecting asset subset...")
    n_assets = 10
    returns_subset, selected_indices = select_asset_subset(returns_data, mask, n_assets)
    residuals_subset, _ = select_asset_subset(residuals_data, mask, n_assets, random_seed=42)

    # Use recent data
    T_start = max(0, returns_subset.shape[0] - 1000)  # Last ~4 years
    returns_subset = returns_subset[T_start:, :]
    residuals_subset = residuals_subset[T_start:, :]

    print(f"Using data shape: {returns_subset.shape}")

    # Calculate correlations
    print("\n3. Calculating rolling correlations...")
    window_size = 30  # ~6 months
    step_size = 1     # ~1 month

    returns_corr, time_indices = rolling_correlation_matrix(
        returns_subset, window_size, step_size
    )
    residuals_corr, _ = rolling_correlation_matrix(
        residuals_subset, window_size, step_size
    )

    # Create visualizations
    print("\n4. Creating visualizations...")
    asset_names = [f"Stock_{i+1}" for i in range(n_assets)]

    # Evolution plots
    fig1 = plot_correlation_evolution(
        returns_corr, time_indices, asset_names,
        title="Returns Correlation Evolution (Small Sample)"
    )

    fig2 = plot_correlation_evolution(
        residuals_corr, time_indices, asset_names,
        title="Residuals Correlation Evolution (Small Sample)"
    )

    # Comparison plot
    fig3 = compare_returns_vs_residuals(
        returns_corr, residuals_corr, time_indices, asset_names
    )

    # Network plots
    fig4 = plot_correlation_network(
        returns_corr[-1], asset_names, threshold=0.2,
        title="Returns Correlation Network (Recent)"
    )

    fig5 = plot_correlation_network(
        residuals_corr[-1], asset_names, threshold=0.2,
        title="Residuals Correlation Network (Recent)"
    )

    print("\n5. Analysis complete!")
    print("Generated plots:")
    print("- Returns correlation evolution")
    print("- Residuals correlation evolution")
    print("- Returns vs Residuals comparison")
    print("- Correlation networks for both")

    return {
        'returns_corr': returns_corr,
        'residuals_corr': residuals_corr,
        'time_indices': time_indices,
        'asset_names': asset_names
    }

def example_summary_stats(analysis_results):
    """
    Print summary statistics from the correlation analysis.
    """
    returns_corr = analysis_results['returns_corr']
    residuals_corr = analysis_results['residuals_corr']

    print("\n=== Summary Statistics ===")

    # Average correlations
    returns_avg = np.mean(returns_corr, axis=(1, 2))
    residuals_avg = np.mean(residuals_corr, axis=(1, 2))

    print(".3f")
    print(".3f")

    # Correlation difference
    diff_avg = returns_avg - residuals_avg
    print(".3f")

    # Most recent correlations
    recent_returns = returns_corr[-1]
    recent_residuals = residuals_corr[-1]

    # Average absolute correlations
    returns_abs_avg = np.mean(np.abs(np.triu(recent_returns, k=1)))
    residuals_abs_avg = np.mean(np.abs(np.triu(recent_residuals, k=1)))

    print("Most Recent Window:")
    print(".3f")
    print(".3f")
    print(".3f")

if __name__ == "__main__":
    # Run example analysis
    results = example_small_analysis()

    # Print summary statistics
    example_summary_stats(results)

    # Show plots
    plt.show()
