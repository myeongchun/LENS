import numpy as np

def compute_consensus(all_errors, all_densities, all_betas, epsilon=1e-10):

    errors_stack = np.stack(all_errors, axis=1)
    densities_stack = np.stack(all_densities, axis=1)
    
    betas_array = np.array(all_betas)
    betas_array = np.clip(betas_array, epsilon, 1.0 - epsilon)
    alpha_raw = np.log(1.0 / betas_array)
    alphas = alpha_raw / (np.sum(alpha_raw) + epsilon)
    
    mean_k = np.mean(densities_stack, axis=1, keepdims=True)
    Ratio_Inv = np.clip(
        (mean_k + epsilon) / (densities_stack + epsilon), 
        0.1, 
        10.0
    )
    
    W_raw = alphas * Ratio_Inv
    W_norm = W_raw / (np.sum(W_raw, axis=1, keepdims=True) + epsilon)
    final_score = np.sum(W_norm * errors_stack, axis=1)
    #final_score = np.sum(alphas * errors_stack, axis=1)
    
    return final_score
