"""
Data loading utilities for graph construction in GNN-enhanced statistical arbitrage.

This module provides graph construction from residual time series.

PRIMARY METHOD: construct_graph_from_residual_data()
- Uses correlation structure of IPCA residuals
- Does NOT require gamma files or characteristics
- Scientifically sound: correlated residuals share unmodeled factors

FALLBACK METHOD: get_graph_for_period() 
- Uses IPCA gamma and characteristics (if available)
- Factor loading similarity approach
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Optional, Tuple
import hashlib
import pickle

from models.graph_utils import (
    compute_factor_loadings, 
    construct_graph,
    construct_graph_from_residuals,
    construct_graph_from_residuals_windowed,
    get_active_assets_mask,
    save_graph,
    load_graph
)


class GraphDataLoader:
    """
    Manages loading of IPCA data and graph construction for GNN models.
    """
    
    def __init__(self, 
                 ipca_dir: str = "residuals/ipca_normalized",
                 cache_dir: str = "residuals/ipca_normalized_graphs",
                 use_cache: bool = True):
        """
        Initialize GraphDataLoader.
        
        Args:
            ipca_dir: Directory containing IPCA residuals and metadata
            cache_dir: Directory for caching constructed graphs
            use_cache: Whether to use cached graphs
        """
        self.ipca_dir = ipca_dir
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            logging.info(f"Created graph cache directory: {cache_dir}")
        
        # Cache for loaded data
        self._gamma_cache = {}
        self._characteristics_cache = {}
        self._graph_cache = {}
    
    def _get_gamma_path(self, n_factors: int, initial_months: int, 
                       window: int, reestimation_freq: int, cap: float) -> str:
        """Get path to gamma files directory."""
        stuff_dir = f"{self.ipca_dir}_stuff"
        return stuff_dir
    
    def _get_characteristics_path(self) -> str:
        """Get path to characteristics data."""
        # IPCA characteristics are typically stored in data/MonthlyData.npz
        return "data/MonthlyData.npz"
    
    def load_ipca_gamma(self, n_factors: int, initial_months: int = 420,
                       window: int = 240, reestimation_freq: int = 12,
                       cap: float = 0.01) -> Dict[int, np.ndarray]:
        """
        Load IPCA gamma coefficients from all reestimation windows.
        
        Args:
            n_factors: Number of factors
            initial_months: Initial training months
            window: Rolling window size
            reestimation_freq: Reestimation frequency in months
            cap: Market cap proportion threshold
            
        Returns:
            gammas: Dictionary mapping window index to (L, K) gamma arrays
        """
        cache_key = (n_factors, initial_months, window, reestimation_freq, cap)
        if cache_key in self._gamma_cache:
            return self._gamma_cache[cache_key]
        
        gammas = {}
        stuff_dir = self._get_gamma_path(n_factors, initial_months, window, 
                                         reestimation_freq, cap)
        
        if not os.path.exists(stuff_dir):
            logging.warning(f"Gamma directory not found: {stuff_dir}")
            return gammas
        
        # Load gamma files for each window
        window_idx = 0
        while True:
            gamma_file = os.path.join(
                stuff_dir,
                f"gamma_{n_factors}_factors_{initial_months}_initialMonths_"
                f"{window}_window_{reestimation_freq}_reestimationFreq_"
                f"{cap}_cap_{window_idx}.npy"
            )
            
            if not os.path.exists(gamma_file):
                break
            
            gamma = np.load(gamma_file)
            gammas[window_idx] = gamma
            logging.debug(f"Loaded gamma for window {window_idx}: shape {gamma.shape}")
            window_idx += 1
        
        logging.info(f"Loaded {len(gammas)} gamma matrices for {n_factors} factors")
        self._gamma_cache[cache_key] = gammas
        return gammas
    
    def load_ipca_characteristics(self) -> Dict[str, np.ndarray]:
        """
        Load IPCA characteristics data.
        
        Returns:
            data: Dictionary containing:
                - 'data': (T, N, L) characteristics array
                - 'date': (T,) array of dates
                - 'permno': (N,) array of asset identifiers
                - 'variable': (L,) array of characteristic names
        """
        if 'characteristics' in self._characteristics_cache:
            return self._characteristics_cache['characteristics']
        
        char_path = self._get_characteristics_path()
        
        if not os.path.exists(char_path):
            logging.warning(f"Characteristics file not found: {char_path}")
            return {}
        
        try:
            data = np.load(char_path, allow_pickle=True)
            char_data = {
                'data': data['data'],
                'date': data['date'],
                'permno': data['permno'],
                'variable': data['variable']
            }
            logging.info(f"Loaded characteristics: shape {char_data['data'].shape}")
            self._characteristics_cache['characteristics'] = char_data
            return char_data
        except Exception as e:
            logging.error(f"Error loading characteristics: {e}")
            return {}
    
    def load_ipca_mask(self, initial_months: int = 420, window: int = 240,
                       cap: float = 0.01) -> Optional[np.ndarray]:
        """
        Load IPCA mask indicating which assets are available at each time.
        
        Args:
            initial_months: Initial training months
            window: Window size
            cap: Market cap proportion threshold
            
        Returns:
            mask: (T, N) boolean array
        """
        mask_path = os.path.join(
            f"{self.ipca_dir}_stuff",
            f"mask_{initial_months}_initialMonths_{window}_window_{cap}_cap.npy"
        )
        
        if not os.path.exists(mask_path):
            logging.warning(f"Mask file not found: {mask_path}")
            return None
        
        mask = np.load(mask_path)
        logging.info(f"Loaded mask: shape {mask.shape}")
        return mask
    
    def compute_asset_betas(self, characteristics: np.ndarray, 
                           gamma: np.ndarray) -> np.ndarray:
        """
        Compute factor loadings for assets.
        
        Args:
            characteristics: (N, L) or (T, N, L) array of characteristics
            gamma: (L, K) array of factor coefficients
            
        Returns:
            beta: (N, K) or (T, N, K) array of factor loadings
        """
        if characteristics.ndim == 2:
            # Single time period: (N, L)
            return compute_factor_loadings(characteristics, gamma)
        elif characteristics.ndim == 3:
            # Multiple time periods: (T, N, L)
            T = characteristics.shape[0]
            betas = []
            for t in range(T):
                beta_t = compute_factor_loadings(characteristics[t], gamma)
                betas.append(beta_t)
            return np.stack(betas, axis=0)  # (T, N, K)
        else:
            raise ValueError(f"Invalid characteristics shape: {characteristics.shape}")
    
    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters."""
        key_str = str(sorted(kwargs.items()))
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_graph_for_period(self,
                            n_factors: int,
                            window_idx: int = 0,
                            initial_months: int = 420,
                            window: int = 240,
                            reestimation_freq: int = 12,
                            cap: float = 0.01,
                            graph_method: str = 'knn',
                            k: int = 10,
                            threshold: float = 0.7,
                            similarity_metric: str = 'cosine',
                            include_self: bool = True,
                            return_weights: bool = True) -> Dict[str, torch.Tensor]:
        """
        Get or construct graph for a specific training period.
        
        Args:
            n_factors: Number of IPCA factors
            window_idx: Training window index
            initial_months: Initial training months
            window: Window size
            reestimation_freq: Reestimation frequency
            cap: Market cap threshold
            graph_method: 'knn' or 'threshold'
            k: Number of neighbors for k-NN
            threshold: Similarity threshold
            similarity_metric: Similarity computation method
            include_self: Include self-loops
            return_weights: Return edge weights
            
        Returns:
            graph_data: Dictionary with 'edge_index', 'edge_weight', 'num_nodes'
        """
        # Generate cache key
        cache_key = self._get_cache_key(
            n_factors=n_factors, window_idx=window_idx,
            initial_months=initial_months, window=window,
            reestimation_freq=reestimation_freq, cap=cap,
            graph_method=graph_method, k=k, threshold=threshold,
            similarity_metric=similarity_metric, include_self=include_self
        )
        
        # Check memory cache
        if cache_key in self._graph_cache:
            logging.debug(f"Using cached graph for window {window_idx}")
            return self._graph_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"graph_{cache_key}.pt")
        if self.use_cache and os.path.exists(cache_file):
            graph_data = load_graph(cache_file)
            self._graph_cache[cache_key] = graph_data
            return graph_data
        
        # Construct new graph
        logging.info(f"Constructing graph for window {window_idx}, {n_factors} factors")
        
        # Load gamma for this window
        gammas = self.load_ipca_gamma(n_factors, initial_months, window, 
                                      reestimation_freq, cap)
        if window_idx not in gammas:
            raise ValueError(f"Gamma not found for window {window_idx}")
        gamma = gammas[window_idx]
        
        # Load characteristics
        char_data = self.load_ipca_characteristics()
        if not char_data:
            raise ValueError("Could not load characteristics data")
        
        # Load mask to get active assets
        mask = self.load_ipca_mask(initial_months, window, cap)
        
        # Get characteristics for the relevant period
        # Window corresponds to months [initial_months + window_idx * reestimation_freq - window : 
        #                                initial_months + window_idx * reestimation_freq]
        start_month = initial_months + window_idx * reestimation_freq - window
        end_month = initial_months + window_idx * reestimation_freq
        
        if mask is not None:
            # Get assets that are active during this period
            active_mask = np.any(mask[start_month:end_month], axis=0)
            char_period = char_data['data'][start_month:end_month, active_mask, :]
        else:
            char_period = char_data['data'][start_month:end_month, :, :]
        
        # Average characteristics over the period
        char_avg = np.nanmean(char_period, axis=0)  # (N, L)
        
        # Compute factor loadings
        beta = compute_factor_loadings(char_avg, gamma)  # (N, K)
        
        # Construct graph
        graph_data = construct_graph(
            beta,
            method=graph_method,
            k=k,
            threshold=threshold,
            similarity_metric=similarity_metric,
            include_self=include_self,
            return_weights=return_weights
        )
        
        # Cache the graph
        self._graph_cache[cache_key] = graph_data
        if self.use_cache:
            save_graph(graph_data, cache_file)
        
        return graph_data
    
    def get_simple_graph_from_residuals(self,
                                       residuals: np.ndarray,
                                       lookback: int = 250,
                                       graph_method: str = 'knn',
                                       k: int = 10,
                                       threshold: float = 0.7) -> Dict[str, torch.Tensor]:
        """
        DEPRECATED: Use construct_graph_from_residual_data() instead.
        
        Construct graph directly from residual correlations (legacy fallback method).
        """
        logging.warning("get_simple_graph_from_residuals is deprecated. "
                       "Use construct_graph_from_residual_data() instead.")
        return self.construct_graph_from_residual_data(
            residuals=residuals,
            lookback=lookback,
            method='correlation',
            graph_type=graph_method,
            k=k,
            threshold=threshold
        )
    
    def construct_graph_from_residual_data(self,
                                           residuals: np.ndarray,
                                           lookback: int = 250,
                                           method: str = 'correlation',
                                           graph_type: str = 'knn',
                                           k: int = 10,
                                           threshold: float = 0.3,
                                           alpha: Optional[float] = None,
                                           include_self: bool = True,
                                           min_obs: int = 30,
                                           use_cache: bool = True) -> Dict[str, torch.Tensor]:
        """
        PRIMARY METHOD: Construct graph directly from residual time series.
        
        This method does NOT require IPCA gamma files or characteristics.
        It uses the correlation/dependency structure of residuals.
        
        Scientific justification:
        - Assets with correlated residuals share unmodeled common factors
        - These relationships are useful for spatial denoising in GNN
        - Lead-lag effects persist in residual correlations
        
        Methods available:
        - 'correlation': Pairwise Pearson correlation (simple, robust)
        - 'abs_correlation': Absolute correlation (captures negative relationships)
        - 'graphical_lasso': Sparse precision matrix (direct dependencies)
        
        Args:
            residuals: (T, N) residual time series (zeros = missing data)
            lookback: Number of recent observations to use for correlation
            method: 'correlation', 'abs_correlation', or 'graphical_lasso'
            graph_type: 'knn' or 'threshold'
            k: Number of neighbors for k-NN graph
            threshold: Similarity threshold (0.3 recommended for correlation)
            alpha: L1 regularization for graphical lasso (None = auto CV)
            include_self: Include self-loops in graph
            min_obs: Minimum observations required per asset
            use_cache: Whether to cache the result
            
        Returns:
            graph_data: Dictionary with:
                - 'edge_index': (2, E) tensor of edges
                - 'edge_weight': (E,) tensor of weights
                - 'num_nodes': number of nodes
        """
        T, N = residuals.shape
        logging.info(f"Constructing graph from residuals: T={T}, N={N}, "
                    f"method={method}, graph_type={graph_type}, k={k}")
        
        # Generate cache key
        cache_key = self._get_cache_key(
            T=T, N=N, lookback=lookback, method=method,
            graph_type=graph_type, k=k, threshold=threshold,
            alpha=alpha, include_self=include_self
        )
        
        # Check memory cache
        if use_cache and cache_key in self._graph_cache:
            logging.debug("Using cached graph from residuals")
            return self._graph_cache[cache_key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"residual_graph_{cache_key}.pt")
        if use_cache and self.use_cache and os.path.exists(cache_file):
            try:
                graph_data = load_graph(cache_file)
                self._graph_cache[cache_key] = graph_data
                return graph_data
            except Exception as e:
                logging.warning(f"Could not load cached graph: {e}")
        
        # Use windowed construction for efficiency
        graph_data = construct_graph_from_residuals_windowed(
            residuals=residuals,
            lookback=lookback,
            method=method,
            graph_type=graph_type,
            k=k,
            threshold=threshold,
            include_self=include_self,
            min_obs=min_obs
        )
        
        # Cache the result
        if use_cache:
            self._graph_cache[cache_key] = graph_data
            if self.use_cache:
                try:
                    save_graph(graph_data, cache_file)
                except Exception as e:
                    logging.warning(f"Could not save graph cache: {e}")
        
        return graph_data
    
    def get_graph_for_training_window(self,
                                      residuals: np.ndarray,
                                      window_start: int,
                                      window_end: int,
                                      method: str = 'correlation',
                                      graph_type: str = 'knn',
                                      k: int = 10,
                                      threshold: float = 0.3,
                                      min_obs: int = 30) -> Dict[str, torch.Tensor]:
        """
        Get graph for a specific training window.
        
        Uses residuals from [window_start:window_end] to construct graph.
        This ensures no lookahead bias.
        
        Args:
            residuals: (T, N) full residual time series
            window_start: Start index of training window
            window_end: End index of training window
            method: Graph construction method
            graph_type: 'knn' or 'threshold'
            k: Number of neighbors
            threshold: Similarity threshold
            min_obs: Minimum observations per asset
            
        Returns:
            graph_data: Dictionary with graph tensors
        """
        # Extract window
        window_residuals = residuals[window_start:window_end, :]
        
        # Construct graph from window only (no lookahead)
        return construct_graph_from_residuals(
            residuals=window_residuals,
            method=method,
            graph_type=graph_type,
            k=k,
            threshold=threshold,
            include_self=True,
            min_obs=min_obs,
            return_weights=True
        )
    
    def clear_cache(self):
        """Clear all cached data."""
        self._gamma_cache.clear()
        self._characteristics_cache.clear()
        self._graph_cache.clear()
        logging.info("Cleared all caches")


def create_graph_from_residuals(residuals: np.ndarray,
                                 method: str = 'correlation',
                                 graph_type: str = 'knn',
                                 k: int = 10,
                                 lookback: int = 250) -> Dict[str, torch.Tensor]:
    """
    Convenience function to create graph from residuals without class instantiation.
    
    This is the RECOMMENDED way to construct graphs for GNN models.
    
    Example usage:
        residuals = np.load('residuals/ipca_normalized/IPCA_..._5_factors_....npy')
        graph_data = create_graph_from_residuals(residuals, k=10)
        model.update_graph(graph_data['edge_index'], graph_data['edge_weight'])
    
    Args:
        residuals: (T, N) residual time series
        method: 'correlation', 'abs_correlation', or 'graphical_lasso'
        graph_type: 'knn' or 'threshold'
        k: Number of neighbors for k-NN
        lookback: Number of recent observations to use
        
    Returns:
        graph_data: Dictionary with 'edge_index', 'edge_weight', 'num_nodes'
    """
    loader = GraphDataLoader(use_cache=False)
    return loader.construct_graph_from_residual_data(
        residuals=residuals,
        lookback=lookback,
        method=method,
        graph_type=graph_type,
        k=k,
        use_cache=False
    )

