"""
Graph construction utilities for GNN-enhanced statistical arbitrage.

This module provides functions to build graphs for GNN models using:
1. Residual correlation (primary method - works with available data)
2. Graphical Lasso (sparse precision matrix for direct dependencies)
3. Factor loading similarity (requires IPCA gamma - optional)

Scientific justification for residual-based graphs:
- Assets with correlated residuals share unmodeled common factors
- GNN can leverage this structure for spatial denoising
- Lead-lag relationships persist in residual correlations
"""

import os
import logging
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import GraphicalLassoCV, LedoitWolf
import warnings


def compute_factor_loadings(characteristics: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    Compute factor loadings (betas) for assets.
    
    Args:
        characteristics: (N, L) array of asset characteristics (normalized)
        gamma: (L, K) array of factor loading coefficients from IPCA
        
    Returns:
        beta: (N, K) array of factor loadings for each asset
    """
    # Handle NaN values in characteristics
    characteristics_clean = np.nan_to_num(characteristics, nan=0.0)
    beta = characteristics_clean @ gamma  # (N, L) @ (L, K) = (N, K)
    return beta


def compute_similarity_matrix(beta: np.ndarray, method: str = 'cosine') -> np.ndarray:
    """
    Compute similarity matrix between assets based on factor loadings.
    
    Args:
        beta: (N, K) array of factor loadings
        method: Similarity metric ('cosine', 'correlation', 'euclidean')
        
    Returns:
        similarity: (N, N) similarity matrix
    """
    if method == 'cosine':
        # Cosine similarity: (beta_i · beta_j) / (||beta_i|| ||beta_j||)
        similarity = cosine_similarity(beta)
    elif method == 'correlation':
        # Pearson correlation
        similarity = np.corrcoef(beta)
    elif method == 'euclidean':
        # Inverse euclidean distance (normalized)
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(beta)
        # Normalize and invert: similarity = 1 / (1 + distance)
        similarity = 1.0 / (1.0 + distances)
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(similarity, 1.0)
    
    # Handle NaN/inf values
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
    
    return similarity


def construct_adjacency_knn(similarity: np.ndarray, k: int = 10, 
                             include_self: bool = False) -> np.ndarray:
    """
    Construct adjacency matrix using k-nearest neighbors.
    
    Args:
        similarity: (N, N) similarity matrix
        k: Number of nearest neighbors to connect
        include_self: Whether to include self-loops
        
    Returns:
        adjacency: (N, N) binary adjacency matrix
    """
    N = similarity.shape[0]
    adjacency = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        # Get k+1 nearest neighbors (includes self)
        if include_self:
            top_k_indices = np.argpartition(similarity[i], -(k+1))[-(k+1):]
        else:
            # Exclude self by setting diagonal to -inf temporarily
            sim_copy = similarity[i].copy()
            sim_copy[i] = -np.inf
            top_k_indices = np.argpartition(sim_copy, -k)[-k:]
        
        adjacency[i, top_k_indices] = 1.0
    
    # Make adjacency symmetric (undirected graph)
    adjacency = np.maximum(adjacency, adjacency.T)
    
    if include_self:
        np.fill_diagonal(adjacency, 1.0)
    
    return adjacency


def construct_adjacency_threshold(similarity: np.ndarray, threshold: float = 0.7,
                                   include_self: bool = False) -> np.ndarray:
    """
    Construct adjacency matrix using similarity threshold.
    
    Args:
        similarity: (N, N) similarity matrix
        threshold: Minimum similarity for edge creation
        include_self: Whether to include self-loops
        
    Returns:
        adjacency: (N, N) binary adjacency matrix
    """
    adjacency = (similarity >= threshold).astype(np.float32)
    
    if not include_self:
        np.fill_diagonal(adjacency, 0.0)
    else:
        np.fill_diagonal(adjacency, 1.0)
    
    return adjacency


def create_edge_index(adjacency: np.ndarray) -> torch.Tensor:
    """
    Convert adjacency matrix to PyTorch Geometric edge_index format.
    
    Args:
        adjacency: (N, N) adjacency matrix
        
    Returns:
        edge_index: (2, E) tensor of edge indices where E is number of edges
    """
    # Get non-zero entries (edges)
    edge_list = np.array(np.nonzero(adjacency))  # (2, E)
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    return edge_index


def create_edge_weights(adjacency: np.ndarray, similarity: np.ndarray) -> torch.Tensor:
    """
    Extract edge weights from similarity matrix based on adjacency.
    
    Args:
        adjacency: (N, N) binary adjacency matrix
        similarity: (N, N) similarity matrix
        
    Returns:
        edge_weights: (E,) tensor of edge weights
    """
    # Get weights for edges that exist
    weights = similarity[adjacency > 0]
    edge_weights = torch.tensor(weights, dtype=torch.float32)
    return edge_weights


def construct_graph(beta: np.ndarray, 
                   method: str = 'knn',
                   k: int = 10,
                   threshold: float = 0.7,
                   similarity_metric: str = 'cosine',
                   include_self: bool = True,
                   return_weights: bool = True) -> Dict[str, torch.Tensor]:
    """
    Main function to construct graph from factor loadings.
    
    Args:
        beta: (N, K) array of factor loadings
        method: Graph construction method ('knn' or 'threshold')
        k: Number of neighbors for k-NN
        threshold: Similarity threshold
        similarity_metric: Similarity computation method
        include_self: Whether to include self-loops
        return_weights: Whether to return edge weights
        
    Returns:
        graph_data: Dictionary containing:
            - 'edge_index': (2, E) tensor of edges
            - 'edge_weight': (E,) tensor of weights (if return_weights=True)
            - 'num_nodes': number of nodes
    """
    # Compute similarity matrix
    similarity = compute_similarity_matrix(beta, method=similarity_metric)
    
    # Construct adjacency matrix
    if method == 'knn':
        adjacency = construct_adjacency_knn(similarity, k=k, include_self=include_self)
    elif method == 'threshold':
        adjacency = construct_adjacency_threshold(similarity, threshold=threshold, 
                                                  include_self=include_self)
    else:
        raise ValueError(f"Unknown graph construction method: {method}")
    
    # Create edge index
    edge_index = create_edge_index(adjacency)
    
    graph_data = {
        'edge_index': edge_index,
        'num_nodes': beta.shape[0]
    }
    
    if return_weights:
        edge_weights = create_edge_weights(adjacency, similarity)
        graph_data['edge_weight'] = edge_weights
    
    # Log graph statistics
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / beta.shape[0]
    logging.debug(f"Graph constructed: {beta.shape[0]} nodes, {num_edges} edges, "
                 f"avg degree: {avg_degree:.2f}")
    
    return graph_data


def get_graph_statistics(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, float]:
    """
    Compute statistics about the graph structure.
    
    Args:
        edge_index: (2, E) tensor of edges
        num_nodes: Number of nodes
        
    Returns:
        stats: Dictionary of graph statistics
    """
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / num_nodes
    
    # Compute degree distribution
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        degrees[i] = (edge_index[0] == i).sum()
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'max_degree': degrees.max().item(),
        'min_degree': degrees.min().item(),
        'density': num_edges / (num_nodes * (num_nodes - 1))
    }
    
    return stats


def aggregate_betas_over_period(beta_list: list, method: str = 'mean') -> np.ndarray:
    """
    Aggregate factor loadings over multiple time periods.
    
    Args:
        beta_list: List of (N, K) beta arrays over time
        method: Aggregation method ('mean', 'median', 'last')
        
    Returns:
        beta_agg: (N, K) aggregated factor loadings
    """
    beta_stack = np.stack(beta_list, axis=0)  # (T, N, K)
    
    if method == 'mean':
        beta_agg = np.mean(beta_stack, axis=0)
    elif method == 'median':
        beta_agg = np.median(beta_stack, axis=0)
    elif method == 'last':
        beta_agg = beta_stack[-1]
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    
    return beta_agg


def save_graph(graph_data: Dict[str, torch.Tensor], filepath: str):
    """
    Save graph data to disk.
    
    Args:
        graph_data: Dictionary of graph tensors
        filepath: Path to save file
    """
    torch.save(graph_data, filepath)
    logging.info(f"Graph saved to {filepath}")


def load_graph(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load graph data from disk.
    
    Args:
        filepath: Path to load file
        
    Returns:
        graph_data: Dictionary of graph tensors
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Graph file not found: {filepath}")
    
    graph_data = torch.load(filepath)
    logging.info(f"Graph loaded from {filepath}")
    return graph_data


# =============================================================================
# RESIDUAL-BASED GRAPH CONSTRUCTION METHODS
# These methods work directly with residual time series data
# =============================================================================

def compute_residual_correlation(residuals: np.ndarray, 
                                  min_obs: int = 30) -> np.ndarray:
    """
    Compute correlation matrix from residual time series.
    
    Handles missing data (zeros) by computing pairwise correlations
    only over periods where both assets have valid observations.
    
    Args:
        residuals: (T, N) array of residual time series (zeros = missing)
        min_obs: Minimum observations required for valid correlation
        
    Returns:
        corr_matrix: (N, N) correlation matrix
    """
    T, N = residuals.shape
    
    # Replace zeros with NaN for correlation computation
    residuals_clean = residuals.copy().astype(np.float64)
    residuals_clean[residuals_clean == 0] = np.nan
    
    # Compute correlation matrix handling NaNs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Use pandas for robust pairwise correlation
        try:
            import pandas as pd
            df = pd.DataFrame(residuals_clean)
            corr_matrix = df.corr(method='pearson', min_periods=min_obs).values
        except ImportError:
            # Fallback to numpy (less robust with missing data)
            # Standardize each column
            means = np.nanmean(residuals_clean, axis=0, keepdims=True)
            stds = np.nanstd(residuals_clean, axis=0, keepdims=True)
            stds[stds == 0] = 1  # Avoid division by zero
            standardized = (residuals_clean - means) / stds
            
            # Compute correlation
            valid_counts = np.sum(~np.isnan(standardized), axis=0)
            corr_matrix = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    mask = ~np.isnan(standardized[:, i]) & ~np.isnan(standardized[:, j])
                    if np.sum(mask) >= min_obs:
                        corr_matrix[i, j] = np.corrcoef(
                            standardized[mask, i], 
                            standardized[mask, j]
                        )[0, 1]
                        corr_matrix[j, i] = corr_matrix[i, j]
                    else:
                        corr_matrix[i, j] = 0
                        corr_matrix[j, i] = 0
    
    # Handle NaN values
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(corr_matrix, 1.0)
    
    return corr_matrix


def compute_graphical_lasso_precision(residuals: np.ndarray,
                                       alpha: Optional[float] = None,
                                       min_obs: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sparse precision matrix using Graphical Lasso.
    
    The precision matrix (inverse covariance) captures DIRECT dependencies
    between assets, controlling for all other assets. Non-zero entries
    indicate conditional dependencies - ideal for GNN graph structure.
    
    Scientific justification:
    - Precision matrix entries measure partial correlations
    - Non-zero P[i,j] means i and j are directly related (not through others)
    - L1 regularization produces sparse graph (computational efficiency)
    - Well-established in financial network analysis (Billio et al., 2012)
    
    Args:
        residuals: (T, N) array of residual time series
        alpha: L1 regularization parameter (None = cross-validation)
        min_obs: Minimum observations per asset
        
    Returns:
        precision: (N, N) sparse precision matrix
        covariance: (N, N) estimated covariance matrix
    """
    T, N = residuals.shape
    
    # Clean data: replace zeros with NaN, then impute with column mean
    residuals_clean = residuals.copy().astype(np.float64)
    residuals_clean[residuals_clean == 0] = np.nan
    
    # Impute missing values with column means (simple but effective)
    col_means = np.nanmean(residuals_clean, axis=0)
    for j in range(N):
        mask = np.isnan(residuals_clean[:, j])
        residuals_clean[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0
    
    # Handle columns with too few observations
    valid_counts = np.sum(residuals != 0, axis=0)
    invalid_cols = valid_counts < min_obs
    
    if np.any(invalid_cols):
        logging.warning(f"Graphical Lasso: {np.sum(invalid_cols)} assets have < {min_obs} observations")
        # Set invalid columns to small random noise (to avoid singular matrix)
        residuals_clean[:, invalid_cols] = np.random.randn(T, np.sum(invalid_cols)) * 1e-6
    
    try:
        if alpha is None:
            # Use cross-validation to find optimal alpha
            model = GraphicalLassoCV(cv=5, max_iter=500, tol=1e-4)
        else:
            from sklearn.covariance import GraphicalLasso
            model = GraphicalLasso(alpha=alpha, max_iter=500, tol=1e-4)
        
        model.fit(residuals_clean)
        precision = model.precision_
        covariance = model.covariance_
        
        if alpha is None:
            logging.info(f"Graphical Lasso: optimal alpha = {model.alpha_:.4f}")
        
    except Exception as e:
        logging.warning(f"Graphical Lasso failed: {e}. Using Ledoit-Wolf shrinkage.")
        # Fallback to Ledoit-Wolf shrinkage estimator
        model = LedoitWolf()
        model.fit(residuals_clean)
        covariance = model.covariance_
        # Compute pseudo-precision (not sparse but robust)
        try:
            precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(covariance)
    
    return precision, covariance


def construct_graph_from_residuals(residuals: np.ndarray,
                                    method: str = 'correlation',
                                    graph_type: str = 'knn',
                                    k: int = 10,
                                    threshold: float = 0.3,
                                    alpha: Optional[float] = None,
                                    include_self: bool = True,
                                    min_obs: int = 30,
                                    return_weights: bool = True) -> Dict[str, torch.Tensor]:
    """
    Main function to construct graph directly from residual time series.
    
    This is the PRIMARY method for graph construction when IPCA gamma
    files are not available. It uses the correlation structure of
    residuals to determine asset relationships.
    
    Methods:
    - 'correlation': Use pairwise correlation of residuals (simple, robust)
    - 'abs_correlation': Use absolute correlation (captures negative relationships)
    - 'graphical_lasso': Use sparse precision matrix (direct dependencies)
    
    Graph types:
    - 'knn': Connect each asset to k most similar neighbors
    - 'threshold': Connect if similarity exceeds threshold
    
    Args:
        residuals: (T, N) array of residual time series
        method: 'correlation', 'abs_correlation', or 'graphical_lasso'
        graph_type: 'knn' or 'threshold'
        k: Number of neighbors for k-NN
        threshold: Similarity threshold (0.3 for correlation, adjust for precision)
        alpha: L1 regularization for graphical lasso (None = auto)
        include_self: Include self-loops
        min_obs: Minimum observations for valid correlation
        return_weights: Return edge weights
        
    Returns:
        graph_data: Dictionary with 'edge_index', 'edge_weight', 'num_nodes'
    """
    T, N = residuals.shape
    logging.info(f"Constructing graph from residuals: T={T}, N={N}, method={method}")
    
    # Compute similarity/precision matrix based on method
    if method == 'correlation':
        similarity = compute_residual_correlation(residuals, min_obs=min_obs)
        # Transform to [0, 1] range: (corr + 1) / 2
        similarity_for_edges = (similarity + 1) / 2
        
    elif method == 'abs_correlation':
        corr = compute_residual_correlation(residuals, min_obs=min_obs)
        # Use absolute correlation (captures negative relationships)
        similarity = np.abs(corr)
        similarity_for_edges = similarity
        
    elif method == 'graphical_lasso':
        precision, _ = compute_graphical_lasso_precision(
            residuals, alpha=alpha, min_obs=min_obs
        )
        # Use absolute precision values as similarity
        # Non-zero entries indicate direct dependencies
        similarity = np.abs(precision)
        # Normalize to [0, 1]
        max_val = np.max(similarity[~np.eye(N, dtype=bool)])
        if max_val > 0:
            similarity = similarity / max_val
        similarity_for_edges = similarity
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'correlation', 'abs_correlation', or 'graphical_lasso'")
    
    # Construct adjacency matrix based on graph_type
    if graph_type == 'knn':
        adjacency = construct_adjacency_knn(similarity, k=k, include_self=include_self)
    elif graph_type == 'threshold':
        adjacency = construct_adjacency_threshold(similarity, threshold=threshold, include_self=include_self)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Create edge index
    edge_index = create_edge_index(adjacency)
    
    graph_data = {
        'edge_index': edge_index,
        'num_nodes': N
    }
    
    if return_weights:
        edge_weights = create_edge_weights(adjacency, similarity_for_edges)
        graph_data['edge_weight'] = edge_weights
    
    # Log statistics
    num_edges = edge_index.shape[1]
    avg_degree = num_edges / N if N > 0 else 0
    sparsity = 1 - (num_edges / (N * N)) if N > 0 else 1
    
    logging.info(f"Graph constructed: {N} nodes, {num_edges} edges, "
                f"avg_degree={avg_degree:.2f}, sparsity={sparsity:.4f}")
    
    return graph_data


def construct_graph_from_residuals_windowed(residuals: np.ndarray,
                                             lookback: int = 250,
                                             method: str = 'correlation',
                                             graph_type: str = 'knn',
                                             k: int = 10,
                                             threshold: float = 0.3,
                                             include_self: bool = True,
                                             min_obs: int = 30) -> Dict[str, torch.Tensor]:
    """
    Construct graph using a rolling window of residuals.
    
    Uses the most recent `lookback` observations to compute
    correlations, making the graph adaptive to recent market conditions.
    
    Args:
        residuals: (T, N) array of residual time series
        lookback: Number of recent observations to use
        method: Graph construction method
        graph_type: 'knn' or 'threshold'
        k: Number of neighbors
        threshold: Similarity threshold
        include_self: Include self-loops
        min_obs: Minimum observations for valid correlation
        
    Returns:
        graph_data: Dictionary with graph tensors
    """
    T, N = residuals.shape
    
    # Use most recent `lookback` observations
    if T > lookback:
        residuals_window = residuals[-lookback:, :]
    else:
        residuals_window = residuals
        logging.warning(f"Residuals length {T} < lookback {lookback}, using all data")
    
    return construct_graph_from_residuals(
        residuals_window,
        method=method,
        graph_type=graph_type,
        k=k,
        threshold=threshold,
        include_self=include_self,
        min_obs=min_obs,
        return_weights=True
    )


def get_active_assets_mask(residuals: np.ndarray, min_obs: int = 30) -> np.ndarray:
    """
    Get mask of assets with sufficient observations.
    
    Args:
        residuals: (T, N) residual array
        min_obs: Minimum non-zero observations required
        
    Returns:
        mask: (N,) boolean array
    """
    obs_counts = np.sum(residuals != 0, axis=0)
    return obs_counts >= min_obs


def filter_graph_to_active_assets(graph_data: Dict[str, torch.Tensor],
                                   active_mask: np.ndarray) -> Dict[str, torch.Tensor]:
    """
    Filter graph to only include active assets.
    
    Args:
        graph_data: Original graph data
        active_mask: (N,) boolean mask of active assets
        
    Returns:
        filtered_graph: Graph with only active assets
    """
    edge_index = graph_data['edge_index'].numpy()
    
    # Create mapping from old to new indices
    old_to_new = np.full(len(active_mask), -1, dtype=np.int64)
    old_to_new[active_mask] = np.arange(np.sum(active_mask))
    
    # Filter edges
    src, dst = edge_index[0], edge_index[1]
    valid_edges = active_mask[src] & active_mask[dst]
    
    new_src = old_to_new[src[valid_edges]]
    new_dst = old_to_new[dst[valid_edges]]
    
    new_edge_index = torch.tensor(np.stack([new_src, new_dst]), dtype=torch.long)
    
    filtered_graph = {
        'edge_index': new_edge_index,
        'num_nodes': int(np.sum(active_mask))
    }
    
    if 'edge_weight' in graph_data:
        filtered_graph['edge_weight'] = graph_data['edge_weight'][valid_edges]
    
    return filtered_graph

