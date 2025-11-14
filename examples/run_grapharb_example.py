"""
Example runner for `models.GraphArb`.

This script:
- loads an IPCA residuals .npy file from `residuals/ipca_normalized`;
- selects the first `max_assets` assets that are marked in `residuals/superMask.npy` (or the first columns if no mask);
- prepares a lookback window of residuals and runs a forward pass through `GraphArb`;
- prints summary statistics and saves weights to `examples/grapharb_weights.npy`.

Run (from repo root):
    .\.venv\Scripts\python.exe examples\run_grapharb_example.py

Warning: building the full correlation graph is O(N^2). This example uses a
subset (default 500 assets) to keep runtime and memory reasonable.
"""

import os
import numpy as np
import torch
import sys
import os as _os
# ensure repo root is on sys.path so `models` package can be imported when running as a script
_repo_root = _os.path.dirname(_os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models.GraphArb import GraphArb, example_build_edges_from_residuals_numpy


def main():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    # choose a residuals file available in the repo
    res_path = os.path.join(repo_root, 'residuals', 'ipca_normalized',
                            'IPCA_DailyOOSresiduals_1_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy',
                            'IPCA_DailyOOSresiduals_1_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy')

    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Residuals not found: {res_path}")

    print('Loading residuals from', res_path)
    arr = np.load(res_path)  # shape (T, N)
    T, N = arr.shape
    print(f'Residuals shape: T={T}, N={N}')

    # load superMask if present to select active assets
    sm_path = os.path.join(repo_root, 'residuals', 'superMask.npy')
    if os.path.exists(sm_path):
        supermask = np.load(sm_path)
        active_idx = np.where(supermask)[0]
        if len(active_idx) == 0:
            active_idx = np.arange(N)
    else:
        active_idx = np.arange(N)

    max_assets = 500
    chosen = active_idx[:min(max_assets, len(active_idx))]
    print(f'Selected {len(chosen)} assets (first {max_assets} from superMask)')

    # Choose lookback
    lookback = 30
    if T < lookback:
        raise ValueError('Not enough history in residuals to form lookback windows')

    # Build node-level temporal inputs: use last `lookback` days
    last_slice = arr[-lookback:, :][:, chosen]  # (lookback, n_sel)
    x = last_slice.T  # (N_sel, lookback)

    device = torch.device('cpu')
    x_t = torch.from_numpy(x).float().to(device)

    # Instantiate model
    # use edge_dim=1 to match correlation scalar edge attributes built by the example
    model = GraphArb(logdir='.', lookback=lookback, device=str(device), d_model=64, lstm_hidden=64, gnn_layers=2, topk=16, edge_dim=1)
    model.to(device)
    model.eval()

    # Run forward pass (this will build a correlation graph internally)
    with torch.no_grad():
        w = model(x_t)

    w_np = w.cpu().numpy()
    out_path = os.path.join(repo_root, 'examples', 'grapharb_weights.npy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, w_np)

    print('Saved weights to', out_path)
    print('weights shape', w_np.shape)
    print('weights summary: min', float(w_np.min()), 'max', float(w_np.max()), 'mean', float(w_np.mean()))
    # print top 10 by absolute weight
    idx_top = np.argsort(-np.abs(w_np))[:10]
    print('Top 10 asset indices and weights:')
    for i in idx_top:
        print(i, w_np[i])


if __name__ == '__main__':
    main()
