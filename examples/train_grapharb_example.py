"""
Small training run example that uses `train_test.train` with `GraphArb`.
This runs a few epochs on a small subset of the existing residuals.
"""
import os
import sys
import numpy as np
import torch

# make sure repo root is importable
_repo_root = os.path.dirname(os.path.dirname(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from models.GraphArb import GraphArb
from preprocess import preprocess_cumsum
import train_test


def main():
    repo_root = _repo_root
    res_path = os.path.join(repo_root, 'residuals', 'ipca_normalized',
                            'IPCA_DailyOOSresiduals_1_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy',
                            'IPCA_DailyOOSresiduals_1_factors_420_initialMonths_240_window_12_reestimationFreq_0.01_cap.npy')
    arr = np.load(res_path)
    sm = np.load(os.path.join(repo_root, 'residuals', 'superMask.npy'))
    chosen = np.where(sm)[0][:500]
    print('Using', len(chosen), 'assets for training')

    # use a shorter time window for speed
    Tfull = arr.shape[0]
    Tstart = max(0, Tfull - 1200)
    arr_small = arr[Tstart:, :][:, chosen]
    print('arr_small shape', arr_small.shape)

    # construct model
    lookback = 30
    model = GraphArb(logdir='.', random_seed=123, lookback=lookback, device='cpu', d_model=64, lstm_hidden=64, gnn_layers=1, topk=16, edge_dim=1)
    model.is_trainable = True

    # training args
    num_epochs = 3
    batchsize = 64
    optimizer_name = 'Adam'
    optimizer_opts = {'lr': 1e-3}

    # call train_test.train
    rets_full, turnover, short_prop, weights, assets_to_trade = train_test.train(
        model = model,
        preprocess = preprocess_cumsum,
        data_train = arr_small,
        data_dev = None,
        log_dev_progress = True,
        log_dev_progress_freq = 1,
        num_epochs = num_epochs,
        lr = 1e-3,
        batchsize = batchsize,
        optimizer_name = optimizer_name,
        optimizer_opts = optimizer_opts,
        save_params = False,
        output_path = os.path.join(repo_root, 'examples', 'grapharb_train_output'),
        model_tag = 'grapharb-test',
        lookback = lookback,
        device = 'cpu',
        parallelize = False,
    )

    print('Training completed. sample returns shape', np.array(rets_full).shape)
    outp = os.path.join(repo_root, 'examples', 'grapharb_train_output')
    os.makedirs(outp, exist_ok=True)
    np.save(os.path.join(outp, 'train_rets.npy'), rets_full)
    np.save(os.path.join(outp, 'train_weights.npy'), weights)
    print('Saved training outputs to', outp)


if __name__ == '__main__':
    main()
