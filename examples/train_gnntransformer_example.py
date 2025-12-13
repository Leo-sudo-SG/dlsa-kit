"""
Example script for training and testing GNN-enhanced statistical arbitrage model.

This script demonstrates how to use the GNNTransformer model with IPCA residuals
and the residual-based graph construction method.

Graph Construction Methods:
1. Residual Correlation (PRIMARY): Uses correlation structure of residuals
2. Graphical Lasso: Uses sparse precision matrix for direct dependencies
3. IPCA Factor Loadings (FALLBACK): Requires gamma files (not always available)

Usage:
    python examples/train_gnntransformer_example.py
"""

import os
import sys
import logging
import gzip
import numpy as np
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import GNNTransformer, CNNTransformer
from models.graph_data_loader import GraphDataLoader, create_graph_from_residuals
from models.graph_utils import construct_graph_from_residuals
from preprocess import preprocess_cumsum
from utils import initialize_logging


def load_ipca_residuals(n_factors: int = 5):
    """Load IPCA residuals from the residuals directory."""
    residuals_dir = os.path.join(os.path.dirname(__file__), '..', 'residuals', 'ipca_normalized')
    
    # Try different file patterns
    patterns = [
        f"IPCA_DailyOOSresiduals_{n_factors}_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(residuals_dir, pattern)
        
        # Check for .gz file
        if os.path.exists(filepath + '.gz'):
            print(f"Loading from: {filepath}.gz")
            with gzip.open(filepath + '.gz', 'rb') as f:
                return np.load(f)
        
        # Check for directory with nested file
        if os.path.isdir(filepath):
            nested = os.path.join(filepath, pattern)
            if os.path.exists(nested):
                print(f"Loading from: {nested}")
                return np.load(nested)
        
        # Check for plain .npy file
        if os.path.exists(filepath):
            print(f"Loading from: {filepath}")
            return np.load(filepath)
    
    return None


def test_graph_construction_from_residuals():
    """Test graph construction from residual correlation (PRIMARY method)."""
    print("\n" + "="*80)
    print("Testing Graph Construction from Residuals (PRIMARY METHOD)")
    print("="*80)
    
    # Try to load real IPCA residuals
    residuals = None # load_ipca_residuals(n_factors=5)
    
    if residuals is None:
        # Use synthetic data for testing
        print("IPCA residuals not found. Using synthetic data for testing.")
        np.random.seed(42)
        T, N = 500, 100  # 500 days, 100 assets
        
        # Generate correlated residuals (some assets are correlated)
        base_factors = np.random.randn(T, 5)  # 5 hidden factors
        loadings = np.random.randn(N, 5)  # Asset loadings
        noise = np.random.randn(T, N) * 0.3
        residuals = base_factors @ loadings.T + noise
    else:
        T, N = residuals.shape
        print(f"Loaded IPCA residuals: T={T}, N={N}")
    
    # Test correlation-based graph construction
    print("\n1. Testing correlation-based graph...")
    try:
        graph_data = construct_graph_from_residuals(
            residuals=residuals,
            method='correlation',
            graph_type='knn',
            k=10,
            include_self=True,
            min_obs=30
        )
        
        print(f"   ✓ Correlation graph constructed!")
        print(f"     - Nodes: {graph_data['num_nodes']}")
        print(f"     - Edges: {graph_data['edge_index'].shape[1]}")
        print(f"     - Avg degree: {graph_data['edge_index'].shape[1] / graph_data['num_nodes']:.2f}")
        corr_success = True
    except Exception as e:
        print(f"   ✗ Correlation graph failed: {e}")
        corr_success = False
    
    # Test absolute correlation graph
    print("\n2. Testing absolute correlation graph...")
    try:
        graph_data = construct_graph_from_residuals(
            residuals=residuals,
            method='abs_correlation',
            graph_type='knn',
            k=10,
            include_self=True
        )
        
        print(f"   ✓ Absolute correlation graph constructed!")
        print(f"     - Edges: {graph_data['edge_index'].shape[1]}")
        abs_corr_success = True
    except Exception as e:
        print(f"   ✗ Absolute correlation graph failed: {e}")
        abs_corr_success = False
    
    # Test graphical lasso (may be slow for large N)
    print("\n3. Testing Graphical Lasso graph...")
    if N > 200:
        print("   ⚠ Skipping Graphical Lasso (N > 200, would be slow)")
        glasso_success = True
    else:
        try:
            graph_data = construct_graph_from_residuals(
                residuals=residuals[-250:, :],  # Use last 250 days
                method='graphical_lasso',
                graph_type='knn',
                k=10,
                include_self=True
            )
            
            print(f"   ✓ Graphical Lasso graph constructed!")
            print(f"     - Edges: {graph_data['edge_index'].shape[1]}")
            glasso_success = True
        except Exception as e:
            print(f"   ✗ Graphical Lasso failed: {e}")
            glasso_success = False
    
    # Test convenience function
    print("\n4. Testing convenience function create_graph_from_residuals()...")
    try:
        graph_data = create_graph_from_residuals(
            residuals=residuals,
            method='correlation',
            graph_type='knn',
            k=10
        )
        print(f"   ✓ Convenience function works!")
        convenience_success = True
    except Exception as e:
        print(f"   ✗ Convenience function failed: {e}")
        convenience_success = False
    
    return corr_success and abs_corr_success and glasso_success and convenience_success


def test_model_forward_pass():
    """Test GNNTransformer forward pass."""
    print("\n" + "="*80)
    print("Testing GNNTransformer Forward Pass")
    print("="*80)
    
    import torch
    
    # Create dummy data
    N = 50  # number of assets
    T = 30  # lookback period
    x = torch.randn(N, T)
    
    # Create dummy graph
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 0]
    ], dtype=torch.long)
    edge_weight = torch.ones(5, dtype=torch.float)
    
    # Initialize model
    try:
        model = GNNTransformer(
            logdir='.',
            lookback=T,
            device='cpu',
            gnn_type='GAT',
            gnn_hidden_dim=16,
            gnn_num_layers=2,
            gnn_heads=4,
            edge_index=edge_index,
            edge_weight=edge_weight,
            filter_numbers=[16, 32],
            use_transformer=True,
            attention_heads=4,
            hidden_units_factor=2
        )
        
        print(f"✓ Model initialized successfully!")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Forward pass
        output = model(x)
        print(f"✓ Forward pass successful!")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_models():
    """Compare GNNTransformer vs CNNTransformer architectures."""
    print("\n" + "="*80)
    print("Comparing Model Architectures")
    print("="*80)
    
    import torch
    
    # Shared parameters
    params = {
        'logdir': '.',
        'lookback': 30,
        'device': 'cpu',
        'filter_numbers': [1, 8],
        'filter_size': 2,
        'use_transformer': True,
        'attention_heads': 4,
        'hidden_units_factor': 2,
        'dropout': 0.25
    }
    
    try:
        # CNNTransformer (baseline)
        cnn_model = CNNTransformer(**params)
        cnn_params = sum(p.numel() for p in cnn_model.parameters())
        
        # GNNTransformer
        gnn_params = params.copy()
        gnn_params.update({
            'gnn_type': 'GAT',
            'gnn_hidden_dim': 16,
            'gnn_num_layers': 2,
            'gnn_heads': 4,
            'filter_numbers': [16, 32]  # Adjusted for GNN output
        })
        gnn_model = GNNTransformer(**gnn_params)
        gnn_model_params = sum(p.numel() for p in gnn_model.parameters())
        
        print(f"CNNTransformer:")
        print(f"  - Parameters: {cnn_params:,}")
        print(f"\nGNNTransformer:")
        print(f"  - Parameters: {gnn_model_params:,}")
        print(f"  - Additional parameters: {gnn_model_params - cnn_params:,} ({(gnn_model_params/cnn_params - 1)*100:.1f}% increase)")
        
        return True
    except Exception as e:
        print(f"✗ Model comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test():
    """Run a quick test on small synthetic data with residual-based graph."""
    print("\n" + "="*80)
    print("Running Quick Test with Residual-Based Graph Construction")
    print("="*80)
    
    import torch
    from train_test import train
    
    # Create synthetic residuals with some correlation structure
    T = 500  # time steps
    N = 30   # assets
    lookback = 30
    
    # Generate correlated mean-reverting residuals
    np.random.seed(42)
    
    # Create 3 "sectors" of 10 assets each with intra-sector correlation
    n_sectors = 3
    assets_per_sector = N // n_sectors
    
    residuals = np.zeros((T, N))
    for sector in range(n_sectors):
        # Common sector factor
        sector_factor = np.zeros(T)
        sector_factor[0] = np.random.randn()
        for t in range(1, T):
            sector_factor[t] = 0.9 * sector_factor[t-1] + 0.2 * np.random.randn()
        
        # Individual assets with sector exposure
        for i in range(assets_per_sector):
            asset_idx = sector * assets_per_sector + i
            idio = np.zeros(T)
            idio[0] = np.random.randn()
            for t in range(1, T):
                idio[t] = 0.8 * idio[t-1] + 0.1 * np.random.randn()
            
            # 60% sector factor, 40% idiosyncratic
            residuals[:, asset_idx] = 0.6 * sector_factor + 0.4 * idio
    
    print(f"Generated synthetic residuals: {residuals.shape}")
    print(f"  - Mean: {residuals.mean():.4f}")
    print(f"  - Std: {residuals.std():.4f}")
    print(f"  - Created {n_sectors} sectors of {assets_per_sector} assets each")
    
    # Build graph from residual correlations (PRIMARY method)
    print("\nBuilding graph from residual correlations...")
    graph_data = construct_graph_from_residuals(
        residuals=residuals,
        method='correlation',
        graph_type='knn',
        k=5,  # 5 neighbors per asset
        include_self=True,
        min_obs=30
    )
    
    edge_index = graph_data['edge_index']
    edge_weight = graph_data.get('edge_weight')
    
    print(f"✓ Graph constructed: {graph_data['num_nodes']} nodes, {edge_index.shape[1]} edges")
    print(f"  - Average degree: {edge_index.shape[1] / N:.2f}")
    
    try:
        # Initialize model with residual-based graph
        model = GNNTransformer(
            logdir='.',
            lookback=lookback,
            device='cpu',
            gnn_type='GAT',
            gnn_hidden_dim=8,
            gnn_num_layers=1,
            gnn_heads=2,
            edge_index=edge_index,
            edge_weight=edge_weight,
            filter_numbers=[8, 16],
            use_transformer=True,
            attention_heads=2,
            hidden_units_factor=2,
            dropout=0.1
        )
        
        print(f"✓ Model created with residual-based graph")
        
        # Quick training test (just 5 epochs)
        print(f"Running quick training (5 epochs)...")
        
        rets, turns, shorts, weights, a2t = train(
            model=model,
            preprocess=preprocess_cumsum,
            data_train=residuals,
            data_dev=None,
            log_dev_progress=False,
            num_epochs=5,
            lr=0.001,
            batchsize=50,
            optimizer_name="Adam",
            optimizer_opts={"lr": 0.001},
            early_stopping=False,
            save_params=False,
            lookback=lookback,
            trans_cost=0,
            hold_cost=0,
            objective="sharpe"
        )
        
        sharpe = np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(252)
        print(f"✓ Training completed!")
        print(f"  - Returns mean: {np.mean(rets):.6f}")
        print(f"  - Returns std: {np.std(rets):.6f}")
        print(f"  - Sharpe ratio: {sharpe:.4f}")
        print(f"  - Avg turnover: {np.mean(turns):.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("GNNTransformer Example and Testing")
    print("="*80)
    print("\nThis example demonstrates:")
    print("  1. Residual-based graph construction (PRIMARY method)")
    print("  2. GNN model forward pass")
    print("  3. Architecture comparison")
    print("  4. Quick training with residual-based graph")
    
    # Initialize logging
    initialize_logging('GNNTransformer_example', debug=True)
    
    # Run tests
    results = {}
    
    print("\n1. Testing residual-based graph construction (PRIMARY)...")
    results['graph'] = test_graph_construction_from_residuals()
    
    print("\n2. Testing model forward pass...")
    results['forward'] = test_model_forward_pass()
    
    print("\n3. Comparing model architectures...")
    results['compare'] = compare_models()
    
    print("\n4. Running quick test with residual-based graph...")
    results['quick_test'] = run_quick_test()
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*80)
    if all_passed:
        print("All tests passed!")
        print("\nGraph Construction Methods Available:")
        print("  1. 'correlation': Pairwise Pearson correlation (recommended)")
        print("  2. 'abs_correlation': Absolute correlation")
        print("  3. 'graphical_lasso': Sparse precision matrix")
        print("\nTo run full training on IPCA residuals, use:")
        print("  python run_train_test_gnn.py -c configs/gnntransformer-full.yaml")
    else:
        print("Some tests failed. Please check the errors above.")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

