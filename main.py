"""
Main Execution Script for LENS

Usage:
    python main.py --dataset 1_ALOI --device 0
    python main.py --dataset all --device 1
"""
import random
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

from core.filtering import density_estimation, approximate_filter_size, sparsity_induced_filtering
from core.penalizing import instability_induced_penalty
from core.train import train_autoencoder, inference
from core.weight_update import update_sample_weights, weighted_sampling
from utils.consensus import compute_consensus
from models.FCAE import FCAE
from utils.data_loader import load_npz_data
from utils.metrics import calculate_metrics


def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)


def get_dataset_config(config, dataset_name):
    if not dataset_name.endswith('.npz'):
        dataset_name = f"{dataset_name}.npz"
    
    for ds in config['datasets']:
        if ds['name'] == dataset_name:
            return ds
    
    return None

def run_single_trial(X_all, y_true, config, global_config, seed, device):

    # Initialize
    n_samples = len(X_all)
    weights = np.ones(n_samples) / n_samples
    k_prev = np.ones(n_samples)
    
    all_errors = []
    all_densities = []
    all_betas = []
    
    # Set seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch.cuda.device_count())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    for t in range(global_config['num_models']):
        sampled_indices = weighted_sampling(weights, n_samples)
        X_sampled = X_all[sampled_indices]
        
        model = FCAE(
            input_dim=X_all.shape[1],
            hidden_dim=config['hidden_dim'],
            latent_dim=config['latent_dim']
        ).to(device)
        
        model = train_autoencoder(
            model, X_sampled, device,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['learning_rate']
        )
        
        errors, latent = inference(model, X_all, device)
        
        relative_density, full_log_density = density_estimation(
            latent, device,
            h_k=config['kde_bandwidth'],
            k_subsample=global_config['kde_subsample_size'],
            chunk_size=global_config['kde_gpu_chunk_size']
        )
        
        k_t = np.exp(full_log_density) / (np.max(np.exp(full_log_density)) + 1e-10)
        
        sigma = global_config['SIGMA_FIRST'] if t == 0 else global_config['SIGMA_REST']
        p, stats = approximate_filter_size(errors, lmbda=sigma)
        
        mask = sparsity_induced_filtering(full_log_density, p)
        
        center_log_density = instability_induced_penalty(
            full_log_density, mask, latent,
            h_k=config['kde_bandwidth'],
            quantile=config['meanshift_quantile'],
            seed=seed,
            max_samples=global_config['meanshift_n_samples'],
            use_gpu=global_config['use_gpu_kde'],
            gpu_method="GPU",
            device=device,
            kde_gpu_func=density_estimation,
            chunk_size=global_config['kde_gpu_chunk_size']
        )
        
        final_mask = sparsity_induced_filtering(center_log_density, p)
        
        weights, beta = update_sample_weights(
            weights, errors, k_t, final_mask, k_prev, tau=config['tau']
        )
        k_prev = k_t.copy()
        
        all_errors.append(errors)
        all_densities.append(k_t)
        all_betas.append(beta)
    
    final_score = compute_consensus(all_errors, all_densities, all_betas)
    
    metrics = calculate_metrics(y_true, final_score)
    
    return metrics['auroc'], metrics['aucpr']


def run_experiment(dataset_name, config, global_config, device):

    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    data_path = os.path.join("datasets", dataset_name)
    if not os.path.exists(data_path):
        print(f"Dataset file not found: {data_path}")
        return None
    
    X_tensor, y_tensor = load_npz_data(data_path)
    X_all = X_tensor.to(device)
    y_true = y_tensor.numpy()
    
    print(f"Num Pts: {len(X_all)} | Dim: {X_all.shape[1]} | % Anomaly: {y_true.sum()} ({y_true.sum()/len(y_true)*100:.2f}%)")
    
    aurocs = []
    auprcs = []
    
    for trial_idx, seed in enumerate(global_config['seeds'], 1):
        print(f"\nTrial {trial_idx}/{len(global_config['seeds'])} (Seed {seed})", end=' ')
        
        auroc, auprc = run_single_trial(
            X_all, y_true, config, global_config, seed, device
        )
        
        aurocs.append(auroc)
        auprcs.append(auprc)
        
        print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    
    auroc_mean = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    auprc_mean = np.mean(auprcs)
    auprc_std = np.std(auprcs)
    
    print(f"FINAL RESULTS:")
    print(f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}")
    print(f"AUPRC: {auprc_mean:.4f} ± {auprc_std:.4f}")
    
    results = {
        'dataset': dataset_name,
        'auroc_mean': float(auroc_mean),
        'auroc_std': float(auroc_std),
        'auprc_mean': float(auprc_mean),
        'auprc_std': float(auprc_std)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='LENS'
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help='Dataset name (e.g., "1_ALOI" or "all")'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.json',
        help='Path to config.json (default: config.json)'
    )
    # --device 인자 추가
    parser.add_argument(
        '--device', 
        type=str, 
        default='0',
        help='GPU device number (default: "0")'
    )
    args = parser.parse_args()
    
    # Load config
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
    
    full_config = load_config(args.config)
    global_config = full_config['global_settings']
    
    # Device 할당 로직 수정
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")
        
    print(f"Device: {device}")
    print(f"Seeds: {global_config['seeds']}")
    print(f"Ensemble size: {global_config['num_models']}")
    print(f"{'='*80}")
    
    # Dataset selection
    if args.dataset.lower() == 'all':
        # Run all datasets
        print(f"\nRunning all datasets ({len(full_config['datasets'])} total)\n")
        
        all_results = []
        
        for idx, ds_config in enumerate(full_config['datasets'], 1):
            print(f"Progress: {idx}/{len(full_config['datasets'])}")
            
            results = run_experiment(
                ds_config['name'],
                ds_config,
                global_config,
                device
            )
            
            if results is not None:
                all_results.append(results)
        
        # Save summary
        os.makedirs("results", exist_ok=True)
        output_path = "results/all_results.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"All results saved to: {output_path}")
        
    else:
        # Run single dataset
        ds_config = get_dataset_config(full_config, args.dataset)
        
        if ds_config is None:
            print(f"\nError: Dataset '{args.dataset}' not found in config.json")
            print(f"\nAvailable datasets:")
            for ds in full_config['datasets']:
                print(f"  - {ds['name'].replace('.npz', '')}")
            sys.exit(1)
        
        results = run_experiment(
            ds_config['name'],
            ds_config,
            global_config,
            device
        )
        
        if results is not None:
            # Save result
            os.makedirs("results", exist_ok=True)
            dataset_id = args.dataset.replace('.npz', '')
            output_path = f"results/{dataset_id}_results.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()