import numpy as np


def update_sample_weights(weights, errors, k_t, mask, k_prev, tau, epsilon=1e-10):
    
    # Step 1: Normalize errors
    errors_kept = errors[mask]
    max_kept_error = np.max(errors_kept) if len(errors_kept) > 0 else np.max(errors)
    
    if max_kept_error <= epsilon:
        max_kept_error = epsilon
    
    d_norm = np.clip(errors / (max_kept_error + epsilon), 0.0, 1.0)
    
    # Step 2: Calculate epsilon_t
    epsilon_t = np.clip(np.sum(weights * d_norm), epsilon, 1.0 - epsilon)
    
    # Step 3: Calculate beta
    beta_e = np.sqrt(epsilon_t / (1.0 - epsilon_t))

    factor_error = np.power(beta_e + epsilon, 1.0 - d_norm)

    delta_k = np.abs(k_prev - k_t)

    factor_stable = np.exp(-delta_k / (tau + epsilon))

    new_weights = weights * factor_error * factor_stable
    new_weights[~mask] = 0.0
    sum_weights = np.sum(new_weights)

    if sum_weights > epsilon:
        new_weights = new_weights / sum_weights
    else:
        new_weights = np.ones(len(weights)) / len(weights)
    
    return new_weights, beta_e


def weighted_sampling(weights, n_samples, epsilon=1e-10):
    
    p_norm = weights / (np.sum(weights) + epsilon)

    sampled_indices = np.random.choice(
        n_samples, 
        size=n_samples, 
        replace=True, 
        p=p_norm
    )
    
    return sampled_indices