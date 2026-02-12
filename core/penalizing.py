import numpy as np
import torch
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.neighbors import KernelDensity
from sklearn.metrics.pairwise import euclidean_distances


def mean_shift_clustering(latent_active, quantile, h_k, seed, max_samples):

    n_active = len(latent_active)
    
    # Step 1: Subsample for clustering
    if n_active > max_samples:
        subsample_idx = np.random.choice(n_active, max_samples, replace=False)
        clustering_samples = latent_active[subsample_idx]
    else:
        clustering_samples = latent_active
    
    # Step 2: Bandwidth estimation
    estimated_bw = estimate_bandwidth(
        clustering_samples,
        quantile=quantile,
        n_samples=len(clustering_samples),
        random_state=seed
    )
    
    # Step 3: MeanShift clustering
    final_bw = estimated_bw if (estimated_bw and estimated_bw > 1e-10) else h_k
    meanshift_model = MeanShift(bandwidth=final_bw, bin_seeding=True).fit(clustering_samples)
    
    cluster_centers = meanshift_model.cluster_centers_
    
    # Step 4: Assign samples to nearest center
    center_distances = euclidean_distances(latent_active, cluster_centers)
    cluster_labels = np.argmin(center_distances, axis=1)
    
    return cluster_centers, cluster_labels


def center_density_estimation(cluster_centers, latent_full, h_k, use_gpu, 
                            gpu_method, device, kde_gpu_func, chunk_size):
   
    n_clusters = len(cluster_centers)
    
    if use_gpu and gpu_method == "GPU":
        if isinstance(latent_full, np.ndarray):
            latent_tensor = torch.FloatTensor(latent_full).to(device)
        else:
            latent_tensor = latent_full
        
        centers_tensor = torch.FloatTensor(cluster_centers).to(device)
        concatenated = torch.cat([latent_tensor, centers_tensor], dim=0)

        _, all_log_densities_np = kde_gpu_func(
            concatenated,
            device=device,
            h_k=h_k,
            chunk_size=chunk_size
        )
        
        # 마지막 n_clusters개 추출
        center_log_densities = all_log_densities_np[-n_clusters:]
        
    else:
        if isinstance(latent_full, torch.Tensor):
            latent_np = latent_full.cpu().numpy()
        else:
            latent_np = latent_full
        
        kde_model = KernelDensity(kernel='gaussian', bandwidth=h_k)
        kde_model.fit(latent_np)
        center_log_densities = kde_model.score_samples(cluster_centers)
    
    return center_log_densities


def penalty_calculation(full_log_density, mask, cluster_labels, center_log_densities):
    
    # Step 1: Initialize center density
    center_log_density = np.copy(full_log_density)
    
    # Step 2: Replace active sample densities with center densities
    
    active_indices = np.where(mask)[0] 
    
    for active_idx, assigned_cluster in enumerate(cluster_labels):
        global_idx = active_indices[active_idx]
        center_log_density[global_idx] = center_log_densities[assigned_cluster]
    
    return center_log_density


def instability_induced_penalty(
    full_log_density,
    mask,
    latent_full,
    h_k,
    quantile,
    seed,
    max_samples,
    use_gpu,
    gpu_method,
    device,
    kde_gpu_func,
    chunk_size
):

    if isinstance(latent_full, torch.Tensor):
        latent_np = latent_full.cpu().numpy()
    else:
        latent_np = latent_full
    
    latent_active = latent_np[mask]
    
    if len(latent_active) == 0:
        return np.copy(full_log_density)
    
    try:
        # Step 1: MeanShift clustering
        cluster_centers, cluster_labels = mean_shift_clustering(
            latent_active=latent_active,
            quantile=quantile,
            h_k=h_k,
            seed=seed,
            max_samples=max_samples
        )
        
        # Check if clustering succeeded
        if len(cluster_centers) == 0:
            return np.copy(full_log_density)
        
        # Step 2: Center density estimation
        center_log_densities = center_density_estimation(
            cluster_centers=cluster_centers,
            latent_full=latent_full,
            h_k=h_k,
            use_gpu=use_gpu,
            gpu_method=gpu_method,
            device=device,
            kde_gpu_func=kde_gpu_func,
            chunk_size=chunk_size
        )
        
        # Step 3: Penalty calculation
        center_log_density = penalty_calculation(
            full_log_density=full_log_density,
            mask=mask,
            cluster_labels=cluster_labels,
            center_log_densities=center_log_densities
        )
        
        return center_log_density
        
    except Exception as e:
        return np.copy(full_log_density)