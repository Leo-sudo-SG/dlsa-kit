"""
Modified training script for GNN-enhanced statistical arbitrage models.

This script extends run_train_test.py to handle graph construction and loading
for GNNTransformer models.
"""

import argparse
import datetime
import json
import logging
import gzip
import os
import pathlib
import pickle
import re
import shutil
import socket
import time
import io
import zipfile
import urllib.request

import yaml
import petname
import numpy as np
import pandas as pd
import torch

from train_test import test, estimate
from data import perturb
from preprocess import *
from models import *
from models.graph_data_loader import GraphDataLoader
from utils import initialize_logging, nploadp, import_string, get_free_gpu_ids, send_twilio_message

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False


def configure_logging(app_name: str, run_id: str = None, logdir: str = "logs", debug=False):
    """Configure logging for the run."""
    debugtag = "-debug" if debug else ""
    run_id = str(run_id)
    username = os.path.split(os.path.expanduser("~"))[-1]
    hostname = socket.gethostname().replace(".stanford.edu", "")
    if not os.path.isdir(logdir):
        os.mkdir(logdir)
    starttimestr = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logtag = petname.Generate(2)
    
    fh = logging.FileHandler(f"{logdir}/{app_name}{debugtag}_{run_id}_{logtag}_{username}_{hostname}_{starttimestr}.log")
    ch = logging.StreamHandler()
    formatter = logging.Formatter(f"[%(asctime)s] Run-{run_id} - %(levelname)s - %(message)s", '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logging.getLogger('').handlers = []
    logging.getLogger('').addHandler(fh)
    logging.getLogger('').addHandler(ch)
    logging.info(f"STARTED LOGGING FROM CHILD")
    return username, hostname, logtag, starttimestr


def load_graph_for_model(config: dict, factor_model: str, n_factors: int, 
                         window_idx: int = 0) -> dict:
    """
    Load or construct graph for GNN model.
    
    Args:
        config: Configuration dictionary
        factor_model: Factor model name ('IPCA', 'PCA', 'FamaFrench')
        n_factors: Number of factors
        window_idx: Training window index
        
    Returns:
        graph_data: Dictionary with 'edge_index', 'edge_weight', 'num_nodes'
    """
    # Extract graph parameters from config
    model_config = config['model']
    graph_method = model_config.get('graph_method', 'knn')
    k = model_config.get('graph_k_neighbors', 10)
    threshold = model_config.get('graph_threshold', 0.7)
    similarity_metric = model_config.get('graph_similarity_metric', 'cosine')
    include_self = model_config.get('graph_include_self', True)
    
    # IPCA parameters
    ipca_params = config.get('ipca_params', {})
    initial_months = ipca_params.get('initial_months', 420)
    window = ipca_params.get('window', 240)
    reestimation_freq = ipca_params.get('reestimation_freq', 12)
    cap = config.get('cap_proportion', 0.01)
    
    # Initialize graph data loader
    use_cache = config.get('graph_cache', True)
    graph_loader = GraphDataLoader(use_cache=use_cache)
    
    try:
        if factor_model == 'IPCA':
            # Load graph based on IPCA factor loadings
            graph_data = graph_loader.get_graph_for_period(
                n_factors=n_factors,
                window_idx=window_idx,
                initial_months=initial_months,
                window=window,
                reestimation_freq=reestimation_freq,
                cap=cap,
                graph_method=graph_method,
                k=k,
                threshold=threshold,
                similarity_metric=similarity_metric,
                include_self=include_self,
                return_weights=True
            )
        else:
            # For PCA/FamaFrench, use correlation-based graph as fallback
            logging.warning(f"Factor model {factor_model} not fully supported for graph construction. "
                          f"Using correlation-based graph.")
            # This would require loading residuals - simplified for now
            # In practice, you'd load the residuals and call get_simple_graph_from_residuals
            raise NotImplementedError(f"Graph construction for {factor_model} not yet implemented")
        
        logging.info(f"Loaded graph for {factor_model} {n_factors} factors, window {window_idx}: "
                    f"{graph_data['num_nodes']} nodes, {graph_data['edge_index'].shape[1]} edges")
        return graph_data
        
    except Exception as e:
        logging.error(f"Error loading graph: {e}")
        logging.warning("Falling back to empty graph (no spatial connections)")
        # Return empty graph
        return {
            'edge_index': torch.empty((2, 0), dtype=torch.long),
            'edge_weight': torch.empty(0, dtype=torch.float),
            'num_nodes': 0
        }


def run(config: dict,
        run_id: str = None,
        gpu_device_ids: list = None,
        notification_phone_number: str = None):
    """
    Run GNN-enhanced statistical arbitrage model training and testing.
    
    Args:
        config: Configuration dictionary
        run_id: Run identifier for logging
        gpu_device_ids: GPU device IDs to use
        notification_phone_number: Phone number for completion notifications
    """
    model_name = config['model_name']
    results_tag = config['results_tag']
    debug = config['debug']
    username, hostname, log_tag, starttime = configure_logging(model_name, run_id=run_id, debug=debug) \
                                            if run_id else initialize_logging(model_name, debug=debug)
    
    try:
        logging.info(f"Config: \n{json.dumps(config, indent=2, sort_keys=False)}")
        
        results_filename = f"results_{log_tag}_{results_tag}"
        factor_models = config['factor_models']
        cap = config['cap_proportion']
        use_residual_weights = config['use_residual_weights']
        objective = config['objective']
        
        # Check if this is a GNN model
        is_gnn_model = model_name == 'GNNTransformer'
        
        if is_gnn_model:
            logging.info("Running GNN-enhanced model with graph construction")
        
        # Set up data paths (same as original)
        filepaths = []
        residual_weightsNames = []
        datanames = []
        results_dict = {}
        
        # IPCA data paths
        ipcadir = "ipca_normalized"
        ipcartag = "IPCA_DailyOOSresiduals"
        ipcamtag = "IPCA_DailyMatrixOOSresiduals"
        for factor in factor_models["IPCA"]:
            im = 420  # initial months
            w = 20 * 12  # window size
            filepaths += [f"residuals/{ipcadir}/{ipcartag}_{factor}_factors_{im}_initialMonths_{w}_window_12_reestimationFreq_{cap}_cap.npy"]
            datanames += ['IPCA' + str(factor)]
            residual_weightsNames += [f"residuals/{ipcadir}/{ipcamtag}_{factor}_factors_{im}_initialMonths_{w}_window_12_reestimationFreq_{cap}_cap.npy"]
        
        # PCA data paths
        pcadir = "pca"
        pcartag = "PCA_DailyOOSresiduals"
        pcamtag = "PCA_DailyMatrixOOSresiduals"
        for factor in factor_models.get("PCA", []):
            im = 420
            w = 20 * 12
            filepaths += [f"residuals/{pcadir}/{pcartag}_{factor}_factors_{im}_initialMonths_{w}_window_{cap}_cap.npy"]
            datanames += ['PCA' + str(factor)]
            residual_weightsNames += [f"residuals/{pcadir}/{pcamtag}_{factor}_factors_{im}_initialMonths_{w}_window_{cap}_cap.npy"]
        
        # Fama-French data paths
        ffdir = "famafrench"
        fftag = "DailyFamaFrench_OOSresiduals"
        for factor in factor_models.get("FamaFrench", []):
            iy = 1998
            rw = 60
            filepaths += [f"residuals/{ffdir}/{fftag}_{factor}_factors_{iy}_initialOOSYear_{rw}_rollingWindow_{cap}_Cap.npy"]
            datanames += ['FamaFrench' + str(factor)]
            residual_weightsNames += [None]  # Fama-French doesn't have matrix residuals
        
        # Load dates
        daily_dates_file = f"residuals/{ipcadir}/{ipcartag}_1_factors_{im}_initialMonths_{w}_window_12_reestimationFreq_{cap}_cap.npy"
        if not os.path.isfile(daily_dates_file) and os.path.exists(daily_dates_file + ".gz"):
            logging.info("Unzipping daily dates file")
            # Unzip the .gz file first
            with gzip.open(daily_dates_file + ".gz", 'rb') as f_in:
                with open(daily_dates_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        
        Data1 = np.load(daily_dates_file, allow_pickle=True)
        
        # Create dates (simplified - in practice would load from IPCA)
        daily_dates = pd.date_range(start='1998-01-01', periods=Data1.shape[0], freq='D')
        
        # GPU selection
        if gpu_device_ids is None:
            gpu_device_ids = get_free_gpu_ids()
            logging.info(f"Auto-selected GPUs: {gpu_device_ids}")
        
        device = f'cuda:{gpu_device_ids[0]}' if len(gpu_device_ids) > 0 else 'cpu'
        parallelize = len(gpu_device_ids) > 1
        
        # Get preprocessing function
        preprocess = import_string(f"preprocess.{config['preprocess_func']}")
        
        # Training parameters
        num_epochs = config['num_epochs']
        batch_size = config['batch_size']
        retrain_freq = config['retrain_freq']
        rolling_retrain = config['rolling_retrain']
        force_retrain = config['force_retrain']
        length_training = config['length_training']
        early_stopping = config['early_stopping']
        
        # Market frictions
        market_frictions = config.get('market_frictions', False)
        trans_cost = config.get('trans_cost', 0) if market_frictions else 0
        hold_cost = config.get('hold_cost', 0) if market_frictions else 0
        
        # Model import
        model_class = import_string(f"models.{model_name}")
        
        # Run for each dataset
        for filepath, dataname, residual_weights_name in zip(filepaths, datanames, residual_weightsNames):
            logging.info(f"\n{'='*80}\nProcessing {dataname}\n{'='*80}")
            
            # Load residuals with proper handling for nested directories and .gz files
            logging.info(f"Loading residuals from: {filepath}")
            
            # Handle nested directory structure and .gz files (same as original run_train_test.py)
            if not os.path.isfile(filepath) and os.path.exists(filepath + ".gz"):
                logging.info("Found .gz file")
                # Remove any trailing .npy directory that shouldn't be there
                if os.path.isdir(filepath):
                    # filepath is a directory, so use the nested file inside it
                    nested_filepath = os.path.join(filepath, os.path.basename(filepath))
                    if os.path.isfile(nested_filepath):
                        filepath = nested_filepath
                    else:
                        # Directory exists but nested file doesn't, unzip the .gz
                        with gzip.open(filepath + ".gz", 'rb') as f_in:
                            with open(filepath, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                else:
                    # filepath doesn't exist, unzip the .gz
                    with gzip.open(filepath + ".gz", 'rb') as f_in:
                        with open(filepath, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
            elif os.path.isdir(filepath):
                # If filepath is a directory, use the nested file
                nested_filepath = os.path.join(filepath, os.path.basename(filepath))
                if os.path.isfile(nested_filepath):
                    filepath = nested_filepath
            
            if not os.path.exists(filepath):
                logging.warning(f"File not found: {filepath}")
                continue
            
            Data = np.load(filepath, allow_pickle=True).astype(np.float32)
            logging.info(f"Residuals loaded: shape {Data.shape}")
            Data[np.isnan(Data)] = 0
            
            # Apply perturbation if configured
            if 'perturbation' in config and config['perturbation']:
                logging.info(f"Before perturbing residuals: std: {np.std(Data[Data != 0]):0.4f}")
                Data = perturb(Data, config['perturbation'])
                logging.info(f"After perturbing residuals: std: {np.std(Data[Data != 0]):0.4f}")
            
            # Load residual weights if needed
            residual_weights = None
            if use_residual_weights and residual_weights_name:
                logging.info(f"Loading residual weights from: {residual_weights_name}")
                rw_path = residual_weights_name
                
                # Handle nested directory or .gz for residual weights
                if os.path.isdir(rw_path):
                    nested_rw = os.path.join(rw_path, os.path.basename(rw_path))
                    if os.path.isfile(nested_rw):
                        rw_path = nested_rw
                    elif os.path.exists(rw_path + ".gz"):
                        with gzip.open(rw_path + ".gz", 'rb') as f_in:
                            nested_rw = os.path.join(rw_path, os.path.basename(rw_path))
                            with open(nested_rw, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        rw_path = nested_rw
                
                if not os.path.isfile(rw_path) and os.path.exists(rw_path + ".gz"):
                    with gzip.open(rw_path + ".gz", 'rb') as f_in:
                        with open(rw_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                
                if os.path.isfile(rw_path):
                    residual_weights = np.load(rw_path, allow_pickle=True)
                    logging.info(f"Residual weights loaded: shape {residual_weights.shape}")
                else:
                    logging.warning(f"Residual weights not found: {residual_weights_name}")
            
            # Create output directory
            output_path = os.path.join(os.getcwd(), 'results', model_name, dataname)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # Extract factor number from dataname
            factor_match = re.search(r'(\d+)$', dataname)
            n_factors = int(factor_match.group(1)) if factor_match else 5
            
            # Determine factor model type
            if 'IPCA' in dataname:
                factor_model_type = 'IPCA'
            elif 'PCA' in dataname:
                factor_model_type = 'PCA'
            elif 'FamaFrench' in dataname:
                factor_model_type = 'FamaFrench'
            else:
                factor_model_type = 'Unknown'
            
            # For GNN models, build graph from residuals before training
            if is_gnn_model:
                logging.info("Building graph from residual correlations...")
                
                # Get graph parameters from config
                graph_params = config.get('graph_params', {})
                graph_method = graph_params.get('method', 'correlation')
                graph_type = graph_params.get('graph_type', 'knn')
                k_neighbors = graph_params.get('k_neighbors', 10)
                graph_threshold = graph_params.get('threshold', 0.3)
                graph_lookback = graph_params.get('lookback', 250)
                min_obs = graph_params.get('min_obs', 30)
                include_self = graph_params.get('include_self', True)
                
                # Build graph from the training portion of residuals
                from models.graph_utils import construct_graph_from_residuals
                
                # Use initial training window for graph construction
                graph_residuals = Data[:length_training, :]
                
                graph_data = construct_graph_from_residuals(
                    residuals=graph_residuals,
                    method=graph_method,
                    graph_type=graph_type,
                    k=k_neighbors,
                    threshold=graph_threshold,
                    include_self=include_self,
                    min_obs=min_obs,
                    return_weights=True
                )
                
                logging.info(f"Graph constructed: {graph_data['num_nodes']} nodes, "
                           f"{graph_data['edge_index'].shape[1]} edges")
                
                # Store graph in config for model initialization
                config['model']['edge_index'] = graph_data['edge_index']
                config['model']['edge_weight'] = graph_data.get('edge_weight')
            
            # Run test or estimate
            mode = config['mode']
            if mode == 'test':
                returns, sharpe, mean_ret, std_ret, turnovers, short_proportions = test(
                    Data=Data,
                    daily_dates=daily_dates,
                    model=model_class,
                    preprocess=preprocess,
                    config=config,
                    residual_weights=residual_weights,
                    log_dev_progress_freq=50,
                    num_epochs=num_epochs,
                    batchsize=batch_size,
                    early_stopping=early_stopping,
                    save_params=True,
                    device=device,
                    output_path=output_path,
                    model_tag=dataname,
                    lookback=config['model']['lookback'],
                    retrain_freq=retrain_freq,
                    length_training=length_training,
                    rolling_retrain=rolling_retrain,
                    parallelize=parallelize,
                    device_ids=gpu_device_ids,
                    trans_cost=trans_cost,
                    hold_cost=hold_cost,
                    force_retrain=force_retrain,
                    objective=objective
                )
            elif mode == 'estimate':
                returns, sharpe, mean_ret, std_ret, turnovers, short_proportions = estimate(
                    Data=Data,
                    daily_dates=daily_dates,
                    model=model_class,
                    preprocess=preprocess,
                    config=config,
                    residual_weights=residual_weights,
                    log_dev_progress_freq=50,
                    num_epochs=num_epochs,
                    batchsize=batch_size,
                    early_stopping=early_stopping,
                    save_params=True,
                    device=device,
                    output_path=output_path,
                    model_tag=dataname,
                    lookback=config['model']['lookback'],
                    length_training=length_training,
                    test_size=retrain_freq,
                    parallelize=parallelize,
                    device_ids=gpu_device_ids,
                    trans_cost=trans_cost,
                    hold_cost=hold_cost,
                    force_retrain=force_retrain,
                    objective=objective,
                    estimate_start_idx=0
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
            # Store results
            results_dict[dataname] = {
                'sharpe': sharpe,
                'mean_return': mean_ret,
                'std_return': std_ret,
                'returns': returns,
                'turnovers': turnovers,
                'short_proportions': short_proportions
            }
            
            logging.info(f"Completed {dataname}: Sharpe={sharpe:.4f}, Return={mean_ret:.4f}, Std={std_ret:.4f}")
        
        # Save all results
        results_path = os.path.join(os.getcwd(), 'results', model_name, f'{results_filename}.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(results_dict, f)
        logging.info(f"Saved results to {results_path}")
        
        # Send notification if configured
        if notification_phone_number:
            try:
                send_twilio_message(
                    notification_phone_number,
                    f"GNN training complete: {model_name} - {results_tag}"
                )
            except Exception as e:
                logging.warning(f"Could not send notification: {e}")
        
        logging.info("=" * 80)
        logging.info("ALL RUNS COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Error during execution: {e}", exc_info=True)
        if notification_phone_number:
            try:
                send_twilio_message(
                    notification_phone_number,
                    f"GNN training FAILED: {model_name} - {str(e)[:100]}"
                )
            except:
                pass
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GNN-enhanced statistical arbitrage model')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('-g', '--gpus', type=str, default=None,
                       help='Comma-separated list of GPU IDs (e.g., "0,1,2")')
    parser.add_argument('-p', '--phone', type=str, default=None,
                       help='Phone number for completion notifications')
    parser.add_argument('-r', '--run_id', type=str, default=None,
                       help='Run ID for logging')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Set logging level
    if config.get('debug', False):
        logging.getLogger('').setLevel(logging.DEBUG)
    else:
        logging.getLogger('').setLevel(logging.INFO)
    
    # Run
    run(config, run_id=args.run_id, gpu_device_ids=gpu_ids, 
        notification_phone_number=args.phone)

