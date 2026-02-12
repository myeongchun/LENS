import torch
import numpy as np

#density estimation

def density_estimation(latent_tensor, device, h_k=0.1, k_subsample=2000, chunk_size=50000, epsilon=1e-10):

    latent_tensor = latent_tensor.to(device)
    n_samples = latent_tensor.shape[0]
    
    # 1. Subsampling for kernel centers
    if n_samples > k_subsample:
        subsample_idx = torch.randperm(n_samples, device=device)[:k_subsample]
        kernel_centers = latent_tensor[subsample_idx]
    else:
        kernel_centers = latent_tensor
    
    log_densities = []
    
    # 2. Chunk-wise computation
    for i in range(0, n_samples, chunk_size):
        chunk_end = min(i + chunk_size, n_samples)
        chunk = latent_tensor[i:chunk_end]

        chunk_norm = torch.sum(chunk ** 2, dim=1, keepdim=True)
        center_norm = torch.sum(kernel_centers ** 2, dim=1, keepdim=True).T
        cross_term = torch.mm(chunk, kernel_centers.T)
        
        dist_sq = chunk_norm + center_norm - 2 * cross_term
        
        # Apply Gaussian Kernel
        kernel_vals = torch.exp(-dist_sq / (2 * (h_k ** 2)))
        density = torch.mean(kernel_vals, dim=1)
        
        log_density_chunk = torch.log(density + epsilon)
        log_densities.append(log_density_chunk.cpu())
    
    full_log_density = torch.cat(log_densities, dim=0)

    # 3. Latent density calculation
    max_log = torch.max(full_log_density)
    relative_density = torch.exp(full_log_density - max_log)
    
    return relative_density.numpy(), full_log_density.numpy()

#gaussian distribution fitting

def approximate_filter_size(current_errors, lmbda, epsilon=1e-10):

    # Step 1: Log-transform the errors
    log_errors = np.log(current_errors + epsilon)
    
    # Step 2: Calculate statistics
    mu_log = np.mean(log_errors)
    sigma_log = np.std(log_errors, ddof=1)
    
    # Step 3: Determine threshold
    threshold_log = mu_log + lmbda * sigma_log
    threshold_val = np.exp(threshold_log)
    
    # Step 4: Calculate the filtering size p
    p = int(np.sum(current_errors > threshold_val))
    
    stats = {
        'mu_log': mu_log,
        'sigma_log': sigma_log,
        'threshold': threshold_val
    }

    return p, stats

#density-based filtering

def sparsity_induced_filtering(relative_density, p):

    n_samples = len(relative_density)
    
    # Step 1: Order by density (Ascending)
    sorted_indices = np.argsort(relative_density)
    
    # Step 2: Select p samples with the lowest density
    filtered_indices = sorted_indices[:p]
    
    # Step 3: Generate boolean mask (True: Keep, False: Filtered)
    mask = np.ones(n_samples, dtype=bool)
    mask[filtered_indices] = False
    
    return mask