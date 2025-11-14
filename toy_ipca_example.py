"""
Toy IPCA Example: Synthetic Data Walkthrough

This script demonstrates the IPCA algorithm on a tiny synthetic dataset:
- 4 months of data
- 5 assets per month
- 3 characteristics (L=3)
- 2 factors (nFactors=2)

It walks through:
1. Constructing synthetic monthly returns and characteristics
2. Initialization (PCA-based)
3. Iterative Gamma and factor estimation (step_gamma, step_factor)
4. Computing out-of-sample residuals
5. Verifying residual properties

Expected output: printed shapes and numerical values at each stage.
"""

import numpy as np
import pandas as pd
import logging
from numpy.linalg import solve, pinv
import os
import argparse

# Set up logging for readability
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_ipca_data(n_months=4, n_assets=5, n_characteristics=3, n_factors=2, random_seed=42):
    """
    Generate synthetic monthly returns and characteristics for toy IPCA.
    
    Returns:
      R_list: list of n_months arrays, each shape (n_assets, 1) or (n_assets,)
      I_list: list of n_months arrays, each shape (n_assets, n_characteristics)
      Gamma_true: ground truth mapping (n_characteristics, n_factors)
    """
    np.random.seed(random_seed)
    
    # Ground truth Gamma: characteristics -> factor loadings
    Gamma_true = np.random.randn(n_characteristics, n_factors) * 0.5
    logger.info(f"Ground truth Gamma shape: {Gamma_true.shape}")
    logger.info(f"Gamma_true:\n{Gamma_true}\n")
    
    R_list = []
    I_list = []
    
    for month in range(n_months):
        # Generate characteristics I_month (n_assets, n_characteristics)
        I_month = np.random.randn(n_assets, n_characteristics)
        
        # Generate true factors f_month (n_factors,)
        f_month = np.random.randn(n_factors)
        
        # Generate betas from true Gamma: Beta_month = I_month @ Gamma_true
        Beta_month = I_month @ Gamma_true  # (n_assets, n_factors)
        
        # Generate returns: R_month = Beta_month @ f_month + noise (idiosyncratic)
        idiosyncratic = np.random.randn(n_assets) * 0.1  # small noise
        R_month = Beta_month @ f_month + idiosyncratic  # (n_assets,)
        
        R_list.append(R_month)
        I_list.append(I_month)
        
        logger.info(f"Month {month}:")
        logger.info(f"  I_month shape: {I_month.shape}")
        logger.info(f"  f_month: {f_month}")
        logger.info(f"  Beta_month shape: {Beta_month.shape}")
        logger.info(f"  R_month shape: {R_month.shape}")
        logger.info(f"  R_month: {R_month}\n")
    
    return R_list, I_list, Gamma_true


# ============================================================================
# IPCA ALGORITHM COMPONENTS (simplified versions from ipca.py)
# ============================================================================

def step_factor(R_list, I_list, Gamma):
    """
    Given Gamma, estimate factors for each month.
    
    For each month t:
      Beta_t = I_t @ Gamma  (n_assets_t, n_factors)
      Solve: (Beta_t.T @ Beta_t) f_t = Beta_t.T @ R_t
      Returns: f_t (n_factors,)
    
    Args:
      R_list: list of monthly returns (n_assets,)
      I_list: list of characteristics (n_assets, n_characteristics)
      Gamma: (n_characteristics, n_factors)
    
    Returns:
      f_list: list of factor vectors, each shape (n_factors,)
      residual_list: list of residuals for each month
    """
    f_list = []
    residual_list = []
    
    for t, (R_t, I_t) in enumerate(zip(R_list, I_list)):
        Beta_t = I_t @ Gamma  # (n_assets, n_factors)
        
        # Solve normal equation: (Beta_t.T @ Beta_t) f_t = Beta_t.T @ R_t
        A = Beta_t.T @ Beta_t  # (n_factors, n_factors)
        b = Beta_t.T @ R_t     # (n_factors,)
        
        try:
            f_t = solve(A, b)
        except np.linalg.LinAlgError:
            logger.warning(f"Singular matrix in month {t}, using pinv")
            f_t = pinv(A) @ b
        
        # Residuals: R_t - Beta_t @ f_t
        residuals_t = R_t - Beta_t @ f_t
        
        f_list.append(f_t)
        residual_list.append(residuals_t)
    
    return f_list, residual_list


def step_gamma(R_list, I_list, f_list, n_factors, n_characteristics):
    """
    Given factors, estimate Gamma across all months.
    
    Build and solve the normal equation:
      A = sum_t kron(I_t, f_t.T).T @ kron(I_t, f_t.T)
      b = sum_t kron(I_t, f_t.T).T @ R_t
      Gamma_vec = solve(A, b)
    
    Then reshape Gamma_vec to (n_characteristics, n_factors).
    
    Args:
      R_list, I_list: monthly data
      f_list: factor estimates for each month
      n_factors, n_characteristics: dimensions
    
    Returns:
      Gamma: (n_characteristics, n_factors)
    """
    n_params = n_characteristics * n_factors
    A = np.zeros((n_params, n_params))
    b = np.zeros(n_params)
    
    for t, (R_t, I_t, f_t) in enumerate(zip(R_list, I_list, f_list)):
        # kron(I_t, f_t.T) is (n_assets * n_characteristics, 1) when reshaped
        # More precisely: each row corresponds to one coefficient in vectorized form
        tmp_t = np.kron(I_t, f_t)  # (n_assets, n_characteristics * n_factors)
        
        A += tmp_t.T @ tmp_t
        b += tmp_t.T @ R_t
    
    try:
        Gamma_vec = solve(A, b)
    except np.linalg.LinAlgError:
        logger.warning("Singular matrix in step_gamma, using pinv")
        Gamma_vec = pinv(A) @ b
    
    Gamma = Gamma_vec.reshape((n_characteristics, n_factors))
    return Gamma


def initial_gamma(R_list, I_list, n_factors, n_characteristics):
    """
    Initialize Gamma via PCA on X = sum_t I_t.T @ R_t.
    
    Returns top n_factors eigenvectors of X.T @ X.
    """
    X = np.zeros((len(R_list), n_characteristics))
    for t, (R_t, I_t) in enumerate(zip(R_list, I_list)):
        X[t, :] = I_t.T @ R_t / len(R_t)
    
    # PCA: top eigenvectors of X.T @ X
    evals, evecs = np.linalg.eig(X.T @ X)
    # Sort by eigenvalue (descending)
    idx = np.argsort(-evals)
    Gamma = evecs[:, idx[:n_factors]]
    
    logger.info(f"Initial Gamma from PCA, shape: {Gamma.shape}")
    logger.info(f"Initial Gamma:\n{Gamma}\n")
    
    return Gamma


# ============================================================================
# MAIN IPCA ITERATION
# ============================================================================

def run_ipca(R_list, I_list, n_factors, n_characteristics, max_iter=50, tol=1e-3):
    """
    Run IPCA iteration: alternately update Gamma and f until convergence.
    """
    logger.info("="*70)
    logger.info("IPCA ITERATION")
    logger.info("="*70 + "\n")
    
    # Initialize Gamma
    Gamma = initial_gamma(R_list, I_list, n_factors, n_characteristics)
    Gamma_old = np.zeros_like(Gamma)
    
    for iteration in range(max_iter):
        logger.info(f"Iteration {iteration}:")
        
        # Step 1: update factors given Gamma
        f_list, residual_list = step_factor(R_list, I_list, Gamma)
        logger.info(f"  Factors estimated:")
        for t, f_t in enumerate(f_list):
            logger.info(f"    Month {t}: f_t = {f_t}")
        
        # Step 2: update Gamma given factors
        Gamma_new = step_gamma(R_list, I_list, f_list, n_factors, n_characteristics)
        logger.info(f"  Gamma updated, shape: {Gamma_new.shape}")
        logger.info(f"  Gamma:\n{Gamma_new}")
        
        # Check convergence
        dGamma = np.max(np.abs(Gamma_old - Gamma_new))
        logger.info(f"  max|dGamma| = {dGamma:.6e}\n")
        
        Gamma_old = Gamma
        Gamma = Gamma_new
        
        if dGamma < tol:
            logger.info(f"Converged at iteration {iteration}")
            break
    
    return Gamma, f_list, residual_list


# ============================================================================
# OUT-OF-SAMPLE RESIDUAL COMPUTATION
# ============================================================================

def compute_oos_residuals(R_oos_list, I_oos_list, Gamma):
    """
    Given converged Gamma, compute out-of-sample residuals.
    
    For each OOS month:
      Beta_month = I_month @ Gamma
      Solve: (Beta.T @ Beta) f_oos = Beta.T @ R_oos
      residuals_month = R_oos - Beta @ f_oos
    
    Returns:
      oos_residuals: list of residual arrays
      oos_factors: list of OOS factor arrays
    """
    logger.info("="*70)
    logger.info("OUT-OF-SAMPLE RESIDUAL COMPUTATION")
    logger.info("="*70 + "\n")
    
    oos_residuals = []
    oos_factors = []
    
    for month, (R_oos, I_oos) in enumerate(zip(R_oos_list, I_oos_list)):
        logger.info(f"OOS Month {month}:")
        logger.info(f"  R_oos shape: {R_oos.shape}")
        logger.info(f"  I_oos shape: {I_oos.shape}")
        
        Beta_oos = I_oos @ Gamma  # (n_assets, n_factors)
        logger.info(f"  Beta_oos shape: {Beta_oos.shape}")
        
        # Solve for factors
        A = Beta_oos.T @ Beta_oos
        b = Beta_oos.T @ R_oos
        try:
            f_oos = solve(A, b)
        except np.linalg.LinAlgError:
            f_oos = pinv(A) @ b
        
        logger.info(f"  f_oos: {f_oos}")
        
        # Compute residuals
        residuals_oos = R_oos - Beta_oos @ f_oos
        logger.info(f"  residuals_oos shape: {residuals_oos.shape}")
        logger.info(f"  residuals_oos: {residuals_oos}")
        logger.info(f"  residuals_oos norm: {np.linalg.norm(residuals_oos):.6e}\n")
        
        oos_residuals.append(residuals_oos)
        oos_factors.append(f_oos)
    
    return oos_residuals, oos_factors


# ============================================================================
# VERIFICATION: Residuals should be orthogonal to characteristic space
# ============================================================================

def verify_residuals(residuals, I_list):
    """
    Verify that residuals are roughly orthogonal to the characteristic space.
    
    For each month, compute I.T @ residuals; should be small.
    """
    logger.info("="*70)
    logger.info("RESIDUAL VERIFICATION")
    logger.info("="*70 + "\n")
    
    for month, (res, I) in enumerate(zip(residuals, I_list)):
        ortho_check = I.T @ res  # (n_characteristics,)
        norm_ortho = np.linalg.norm(ortho_check)
        logger.info(f"Month {month}:")
        logger.info(f"  I.T @ residuals: {ortho_check}")
        logger.info(f"  norm(I.T @ residuals): {norm_ortho:.6e}\n")


# ============================================================================
# EXPORT RESIDUALS TO CSV
# ============================================================================

def export_residuals_to_csv(residuals_list, output_dir="toy_ipca_output", phase="insample", asset_names=None, time_label="month"):
    """
    Export residuals from multiple months to CSV files for downstream processing.
    
    Args:
      residuals_list: list of residual arrays, each shape (n_assets,)
      output_dir: directory to save CSV files (created if not exists)
      phase: "insample" or "oos" (for naming output files)
      asset_names: optional list of asset names; if None, uses Asset_0, Asset_1, etc.
    
    Returns:
      output_paths: list of CSV file paths created
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    output_paths = []
    
    # Export each month's residuals to its own CSV
    for month, residuals_month in enumerate(residuals_list):
        n_assets = len(residuals_month)
        
        # Use provided asset names or generate default names
        if asset_names is None:
            asset_names_month = [f"Asset_{i}" for i in range(n_assets)]
        else:
            asset_names_month = asset_names[:n_assets]
        
        # Create DataFrame: one row per asset, residual value
        df = pd.DataFrame({
            'asset': asset_names_month,
            f'residual_{phase}': residuals_month
        })
        # Add time label column (month/day) for downstream compatibility
        df[time_label] = month
        
        # Save to CSV
        output_path = os.path.join(output_dir, f"residuals_{phase}_month_{month:02d}.csv")
        df.to_csv(output_path, index=False)
        output_paths.append(output_path)
        
        logger.info(f"Exported {phase} month {month} residuals to: {output_path}")
    
    return output_paths


def export_combined_residuals_csv(residuals_list, output_dir="toy_ipca_output", phase="insample", 
                                   asset_names=None, start_month=0, time_label="month"):
    """
    Export all residuals from multiple months into a single CSV file (long format).
    
    Args:
      residuals_list: list of residual arrays, each shape (n_assets,)
      output_dir: directory to save CSV file
      phase: "insample" or "oos" (for naming)
      asset_names: optional list of asset names
      start_month: starting month index (for labeling)
    
    Returns:
      output_path: CSV file path created
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    all_rows = []
    
    for month, residuals_month in enumerate(residuals_list):
        n_assets = len(residuals_month)
        
        # Use provided asset names or generate default names
        if asset_names is None:
            asset_names_month = [f"Asset_{i}" for i in range(n_assets)]
        else:
            asset_names_month = asset_names[:n_assets]
        
        # Create rows for this month
        for asset_idx, (asset_name, residual_val) in enumerate(zip(asset_names_month, residuals_month)):
            all_rows.append({
                time_label: start_month + month,
                'asset': asset_name,
                'asset_index': asset_idx,
                'residual': residual_val
            })
    
    # Create combined DataFrame
    df_combined = pd.DataFrame(all_rows)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"residuals_{phase}_combined.csv")
    df_combined.to_csv(output_path, index=False)
    
    logger.info(f"Exported combined {phase} residuals to: {output_path}")
    logger.info(f"  Shape: {df_combined.shape[0]} rows × {df_combined.shape[1]} columns")
    
    return output_path


def export_gamma_to_csv(Gamma, output_dir="toy_ipca_output", asset_names=None, char_names=None):
    """
    Export learned Gamma (characteristic → factor loading mapping) to CSV.
    
    Args:
      Gamma: array of shape (n_characteristics, n_factors)
      output_dir: directory to save CSV file
      asset_names: optional list of characteristic names (length = n_characteristics)
      char_names: optional list of factor names (length = n_factors)
    
    Returns:
      output_path: CSV file path created
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    n_char, n_factors = Gamma.shape
    
    # Use provided names or generate defaults
    if asset_names is None:
        char_names_list = [f"Characteristic_{i}" for i in range(n_char)]
    else:
        char_names_list = asset_names[:n_char]
    
    if char_names is None:
        factor_names_list = [f"Factor_{i}" for i in range(n_factors)]
    else:
        factor_names_list = char_names[:n_factors]
    
    # Create DataFrame from Gamma matrix
    df_gamma = pd.DataFrame(Gamma, index=char_names_list, columns=factor_names_list)
    
    # Save to CSV
    output_path = os.path.join(output_dir, "gamma_mapping.csv")
    df_gamma.to_csv(output_path)
    
    logger.info(f"Exported Gamma (characteristic → factor) mapping to: {output_path}")
    logger.info(f"  Shape: {df_gamma.shape}")
    
    return output_path


def export_factors_to_csv(factors_list, output_dir="toy_ipca_output", phase="insample", factor_names=None, time_label="month"):
    """
    Export estimated factors to CSV (one row per month, one column per factor).
    
    Args:
      factors_list: list of factor arrays, each shape (n_factors,)
      output_dir: directory to save CSV file
      phase: "insample" or "oos"
      factor_names: optional list of factor names (length = n_factors)
    
    Returns:
      output_path: CSV file path created
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    n_factors = len(factors_list[0]) if len(factors_list) > 0 else 0
    
    # Use provided names or generate defaults
    if factor_names is None:
        factor_names_list = [f"Factor_{i}" for i in range(n_factors)]
    else:
        factor_names_list = factor_names[:n_factors]
    
    # Create DataFrame: one row per month, columns = factors
    df_factors = pd.DataFrame(np.array(factors_list), columns=factor_names_list)
    df_factors[time_label] = range(len(factors_list))
    df_factors = df_factors[[time_label] + factor_names_list]  # Reorder columns
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"factors_{phase}.csv")
    df_factors.to_csv(output_path, index=False)
    
    logger.info(f"Exported {phase} factors to: {output_path}")
    logger.info(f"  Shape: {df_factors.shape} (months × factors)")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("IPCA TOY EXAMPLE: Synthetic Data")
    logger.info("="*70 + "\n")
    
    # Generate synthetic data
    logger.info("STEP 1: Generate Synthetic Data")
    logger.info("="*70 + "\n")

    parser = argparse.ArgumentParser(description="Toy IPCA example (synthetic data).")
    parser.add_argument('--daily', action='store_true', help='Generate daily-length synthetic data (e.g., 252 trading days)')
    args = parser.parse_args()

    if args.daily:
        # Daily-style toy data (one trading year + quarter OOS)
        n_months = 252
        n_months_oos = 63
        time_label = 'day'
    else:
        # Compact toy defaults (months)
        n_months = 20
        n_months_oos = 8
        time_label = 'month'

    n_assets = 5
    n_characteristics = 3
    n_factors = 2

    R_list, I_list, Gamma_true = generate_synthetic_ipca_data(
        n_months=n_months,
        n_assets=n_assets,
        n_characteristics=n_characteristics,
        n_factors=n_factors,
        random_seed=42
    )

    logger.info(f"Generated {n_months} periods of data:")
    logger.info(f"  Each period: {n_assets} assets")
    logger.info(f"  Characteristics per asset: {n_characteristics}")
    logger.info(f"  Number of factors: {n_factors}\n")
    
    # Run IPCA
    logger.info("STEP 2: Run IPCA (Gamma Estimation)")
    logger.info("="*70 + "\n")
    
    Gamma_est, f_list_est, residuals_in_sample = run_ipca(
        R_list, I_list, n_factors, n_characteristics, max_iter=100, tol=1e-4
    )
    
    logger.info(f"IPCA Converged.")
    logger.info(f"Estimated Gamma shape: {Gamma_est.shape}")
    logger.info(f"Estimated Gamma:\n{Gamma_est}")
    logger.info(f"True Gamma:\n{Gamma_true}")
    logger.info(f"Difference (Gamma_est - Gamma_true) norm: {np.linalg.norm(Gamma_est - Gamma_true):.6e}\n")
    
    # Verify in-sample residuals
    logger.info("STEP 3: Verify In-Sample Residuals")
    logger.info("="*70 + "\n")
    verify_residuals(residuals_in_sample, I_list)
    
    # Generate OOS data and compute OOS residuals
    logger.info("STEP 4: Generate Out-of-Sample Data")
    logger.info("="*70 + "\n")
    
    R_oos_list, I_oos_list, _ = generate_synthetic_ipca_data(
        n_months=8,
        n_assets=n_assets,
        n_characteristics=n_characteristics,
        n_factors=n_factors,
        random_seed=123  # different seed for OOS
    )
    logger.info(f"Generated {len(R_oos_list)} OOS months\n")
    
    oos_residuals, oos_factors = compute_oos_residuals(R_oos_list, I_oos_list, Gamma_est)
    
    # Verify OOS residuals
    logger.info("STEP 5: Verify Out-of-Sample Residuals")
    logger.info("="*70 + "\n")
    verify_residuals(oos_residuals, I_oos_list)
    
    # Summary
    logger.info("="*70)
    logger.info("SUMMARY")
    logger.info("="*70 + "\n")
    logger.info(f"Training data:     {n_months} months × {n_assets} assets × {n_characteristics} characteristics")
    logger.info(f"OOS data:          {len(R_oos_list)} months × {n_assets} assets × {n_characteristics} characteristics")
    logger.info(f"Learned Gamma:     {Gamma_est.shape}")
    logger.info(f"In-sample factors: {len(f_list_est)} months, each shape {f_list_est[0].shape}")
    logger.info(f"OOS factors:       {len(oos_factors)} months, each shape {oos_factors[0].shape}")
    logger.info(f"OOS residuals:     {len(oos_residuals)} months, each shape {oos_residuals[0].shape}")
    logger.info(f"\nAll residuals saved to in-sample and OOS lists.")
    logger.info(f"Residuals are ready for downstream preprocessing (cumsum windows, CNN, etc.)\n")
    
    # Export residuals to CSV
    logger.info("="*70)
    logger.info("EXPORTING RESIDUALS TO CSV")
    logger.info("="*70 + "\n")
    
    asset_names = [f"Asset_{i}" for i in range(n_assets)]
    factor_names = [f"Factor_{i}" for i in range(Gamma_est.shape[1])]
    char_names = [f"Characteristic_{i}" for i in range(n_characteristics)]
    
    # Export in-sample residuals (combined format)
    export_combined_residuals_csv(residuals_in_sample, output_dir="toy_ipca_output", 
                                  phase="insample", asset_names=asset_names, start_month=0)
    
    # Export OOS residuals (combined format)
    export_combined_residuals_csv(oos_residuals, output_dir="toy_ipca_output", 
                                  phase="oos", asset_names=asset_names, start_month=n_months)
    
    # Export Gamma mapping
    export_gamma_to_csv(Gamma_est, output_dir="toy_ipca_output", 
                       asset_names=char_names, char_names=factor_names)
    
    # Export in-sample factors
    export_factors_to_csv(f_list_est, output_dir="toy_ipca_output", 
                         phase="insample", factor_names=factor_names)
    
    # Export OOS factors
    export_factors_to_csv(oos_factors, output_dir="toy_ipca_output", 
                         phase="oos", factor_names=factor_names)
    
    logger.info("\nAll exports completed successfully!")


if __name__ == "__main__":
    main()
