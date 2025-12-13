"""
GNN-enhanced Transformer model for statistical arbitrage.

This model extends the CNN+Transformer architecture with Graph Neural Network layers
to capture spatial dependencies between assets based on factor loading similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

# Import PyTorch Geometric layers
try:
    from torch_geometric.nn import GATConv, GCNConv
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available. GNNTransformer will not work.")

from models.CNNTransformer import CNN_Block


class GNN_Layer(nn.Module):
    """
    Graph Neural Network layer wrapper supporting GAT and GCN.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 gnn_type: str = 'GAT',
                 heads: int = 4,
                 dropout: float = 0.25,
                 concat: bool = True):
        """
        Initialize GNN layer.
        
        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension (per head for GAT)
            gnn_type: Type of GNN ('GAT' or 'GCN')
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            concat: Whether to concatenate heads (GAT) or average
        """
        super(GNN_Layer, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNN_Layer")
        
        self.gnn_type = gnn_type
        self.concat = concat
        
        if gnn_type == 'GAT':
            self.conv = GATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                dropout=dropout,
                concat=concat
            )
            self.out_dim = out_channels * heads if concat else out_channels
        elif gnn_type == 'GCN':
            self.conv = GCNConv(
                in_channels=in_channels,
                out_channels=out_channels * heads if concat else out_channels,
                improved=True,
                add_self_loops=True
            )
            self.out_dim = out_channels * heads if concat else out_channels
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GNN layer.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Edge indices (2, E)
            edge_weight: Edge weights (E,) [optional]
            
        Returns:
            out: Output features (N, out_channels)
        """
        if self.gnn_type == 'GAT':
            # GAT doesn't use edge_weight in the same way
            out = self.conv(x, edge_index)
        else:  # GCN
            out = self.conv(x, edge_index, edge_weight=edge_weight)
        
        out = self.activation(out)
        out = self.dropout(out)
        return out


class GNNTransformer(nn.Module):
    """
    GNN-enhanced CNN+Transformer model for statistical arbitrage.
    
    Architecture:
        1. GNN layers: Spatial feature aggregation across assets
        2. CNN blocks: Temporal feature extraction
        3. Transformer: Temporal attention and aggregation
        4. Linear: Portfolio weight prediction
    """
    
    def __init__(self,
                 logdir,
                 random_seed: int = 0,
                 lookback: int = 30,
                 device: str = "cpu",
                 # GNN parameters
                 gnn_type: str = 'GAT',
                 gnn_hidden_dim: int = 16,
                 gnn_num_layers: int = 2,
                 gnn_heads: int = 4,
                 gnn_dropout: float = 0.25,
                 # Graph parameters (stored but not used in forward)
                 edge_index: Optional[torch.Tensor] = None,
                 edge_weight: Optional[torch.Tensor] = None,
                 # CNN parameters
                 normalization_conv: bool = True,
                 filter_numbers: list = [16, 32],
                 filter_size: int = 2,
                 # Transformer parameters
                 use_transformer: bool = True,
                 attention_heads: int = 4,
                 hidden_units_factor: int = 2,
                 hidden_units: Optional[int] = None,
                 dropout: float = 0.25):
        """
        Initialize GNNTransformer model.
        
        Args:
            logdir: Directory for logging
            random_seed: Random seed
            lookback: Lookback window size
            device: Device for computation
            gnn_type: 'GAT' or 'GCN'
            gnn_hidden_dim: Hidden dimension for GNN layers
            gnn_num_layers: Number of GNN layers
            gnn_heads: Number of attention heads (for GAT)
            gnn_dropout: Dropout for GNN layers
            edge_index: Graph edge indices (2, E)
            edge_weight: Graph edge weights (E,)
            normalization_conv: Normalize CNN layers
            filter_numbers: CNN filter dimensions
            filter_size: CNN kernel size
            use_transformer: Use transformer layer
            attention_heads: Number of transformer attention heads
            hidden_units_factor: Multiplier for hidden units
            hidden_units: Explicit hidden units (overrides factor)
            dropout: General dropout rate
        """
        super(GNNTransformer, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric is required for GNNTransformer")
        
        # Validate hidden units configuration
        if hidden_units and hidden_units_factor and hidden_units != hidden_units_factor * filter_numbers[-1]:
            raise Exception(f"`hidden_units` conflicts with `hidden_units_factor`; provide one or the other, but not both.")
        if hidden_units_factor and not hidden_units:
            hidden_units = hidden_units_factor * filter_numbers[-1]
        
        self.logdir = logdir
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.device = torch.device(device)
        
        # GNN configuration
        self.gnn_type = gnn_type
        self.gnn_num_layers = gnn_num_layers
        self.gnn_hidden_dim = gnn_hidden_dim
        
        # Store graph structure (will be updated during training)
        self.register_buffer('edge_index', edge_index if edge_index is not None 
                           else torch.empty((2, 0), dtype=torch.long))
        self.register_buffer('edge_weight', edge_weight if edge_weight is not None
                           else torch.empty(0, dtype=torch.float))
        
        # CNN/Transformer configuration
        self.filter_numbers = filter_numbers
        self.use_transformer = use_transformer
        self.is_trainable = True
        
        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First GNN layer: 1 (input channels) -> gnn_hidden_dim
        self.gnn_layers.append(
            GNN_Layer(
                in_channels=1,
                out_channels=gnn_hidden_dim // gnn_heads if gnn_type == 'GAT' else gnn_hidden_dim,
                gnn_type=gnn_type,
                heads=gnn_heads,
                dropout=gnn_dropout,
                concat=True
            )
        )
        
        # Additional GNN layers
        for _ in range(gnn_num_layers - 1):
            self.gnn_layers.append(
                GNN_Layer(
                    in_channels=gnn_hidden_dim,
                    out_channels=gnn_hidden_dim // gnn_heads if gnn_type == 'GAT' else gnn_hidden_dim,
                    gnn_type=gnn_type,
                    heads=gnn_heads,
                    dropout=gnn_dropout,
                    concat=True
                )
            )
        
        # Projection layer to match CNN input channels
        # GNN outputs gnn_hidden_dim, we need filter_numbers[0]
        if filter_numbers[0] != gnn_hidden_dim:
            self.gnn_projection = nn.Linear(gnn_hidden_dim, filter_numbers[0])
        else:
            self.gnn_projection = None
        
        # Build CNN blocks
        self.convBlocks = nn.ModuleList()
        for i in range(len(filter_numbers) - 1):
            self.convBlocks.append(
                CNN_Block(
                    filter_numbers[i],
                    filter_numbers[i + 1],
                    normalization=normalization_conv,
                    filter_size=filter_size
                )
            )
        
        # Transformer encoder
        if use_transformer:
            self.encoder = nn.TransformerEncoderLayer(
                d_model=filter_numbers[-1],
                nhead=attention_heads,
                dim_feedforward=hidden_units,
                dropout=dropout
            )
        
        # Output layer
        self.linear = nn.Linear(filter_numbers[-1], 1)
    
    def update_graph(self, edge_index: torch.Tensor, 
                    edge_weight: Optional[torch.Tensor] = None):
        """
        Update graph structure (for use between training windows).
        
        Args:
            edge_index: New edge indices (2, E)
            edge_weight: New edge weights (E,) [optional]
        """
        self.edge_index = edge_index.to(self.device)
        if edge_weight is not None:
            self.edge_weight = edge_weight.to(self.device)
        else:
            self.edge_weight = torch.empty(0, dtype=torch.float, device=self.device)
    
    def has_valid_graph(self) -> bool:
        """Check if the model has a valid graph structure."""
        return self.edge_index is not None and self.edge_index.shape[1] > 0
    
    def build_graph_from_residuals(self, residuals_np: 'np.ndarray',
                                    method: str = 'correlation',
                                    graph_type: str = 'knn',
                                    k: int = 10,
                                    lookback: int = 250):
        """
        Build graph directly from residual time series.
        
        This is the PRIMARY method for graph construction when IPCA gamma
        files are not available.
        
        Args:
            residuals_np: (T, N) numpy array of residuals
            method: 'correlation', 'abs_correlation', or 'graphical_lasso'
            graph_type: 'knn' or 'threshold'
            k: Number of neighbors for k-NN
            lookback: Number of recent observations to use
        """
        from models.graph_data_loader import create_graph_from_residuals
        
        graph_data = create_graph_from_residuals(
            residuals=residuals_np,
            method=method,
            graph_type=graph_type,
            k=k,
            lookback=lookback
        )
        
        self.update_graph(
            graph_data['edge_index'],
            graph_data.get('edge_weight')
        )
        
        return graph_data
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GNNTransformer.
        
        Args:
            x: Input tensor (N, T) where N is number of assets, T is time steps
            
        Returns:
            weights: Portfolio weights (N,)
        """
        N, T = x.shape
        
        # Check if we have a valid graph
        has_graph = self.has_valid_graph()
        
        if has_graph:
            # Full GNN processing: spatial aggregation across assets
            # Reshape for GNN: (N, T) -> process each time step through GNN
            # We'll apply GNN to each time step independently, treating assets as graph nodes
            
            # Initialize output: (N, gnn_hidden_dim, T)
            gnn_output = torch.zeros(N, self.gnn_hidden_dim, T, device=x.device)
            
            # Apply GNN to each time step
            for t in range(T):
                # Get features for this time step: (N, 1)
                x_t = x[:, t].unsqueeze(1)
                
                # Apply GNN layers
                for gnn_layer in self.gnn_layers:
                    if len(self.edge_weight) > 0:
                        x_t = gnn_layer(x_t, self.edge_index, self.edge_weight)
                    else:
                        x_t = gnn_layer(x_t, self.edge_index)
                
                # x_t is now (N, gnn_hidden_dim)
                gnn_output[:, :, t] = x_t
            
            # Project to CNN input dimension if needed
            if self.gnn_projection is not None:
                # Reshape: (N, gnn_hidden_dim, T) -> (N, T, gnn_hidden_dim)
                gnn_output = gnn_output.permute(0, 2, 1)
                # Apply projection: (N, T, gnn_hidden_dim) -> (N, T, filter_numbers[0])
                gnn_output = self.gnn_projection(gnn_output)
                # Reshape back: (N, T, filter_numbers[0]) -> (N, filter_numbers[0], T)
                gnn_output = gnn_output.permute(0, 2, 1)
            
            # Now gnn_output is (N, filter_numbers[0], T)
            x = gnn_output
        else:
            # No graph available: Skip GNN, use input directly
            # Expand input to match expected CNN input channels
            # x is (N, T), need (N, filter_numbers[0], T)
            x = x.unsqueeze(1)  # (N, 1, T)
            if self.filter_numbers[0] > 1:
                # Repeat input across channels (simple expansion)
                x = x.repeat(1, self.filter_numbers[0], 1)
        
        # Apply CNN blocks
        for conv_block in self.convBlocks:
            x = conv_block(x)  # (N, C, T)
        
        # Reshape for transformer: (N, C, T) -> (T, N, C)
        x = x.permute(2, 0, 1)
        
        # Apply transformer
        if self.use_transformer:
            x = self.encoder(x)  # (T, N, C)
        
        # Take last time step and apply linear layer
        # x[-1, :, :] is (N, C)
        output = self.linear(x[-1, :, :]).squeeze()  # (N,)
        
        return output

