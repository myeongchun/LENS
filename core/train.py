import torch
import torch.nn as nn


def train_autoencoder(model, X_sampled, device, epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X_sampled), device=device)
        
        for i in range(0, len(X_sampled), batch_size):
            batch_indices = perm[i:i + batch_size]
            X_batch = X_sampled[batch_indices]
            
            recon, _ = model(X_batch)
            loss = criterion(recon, X_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model


def inference(model, X_all, device):

    model.eval()
    with torch.no_grad():
        recon, latent = model(X_all)
        errors = torch.mean((X_all - recon)**2, dim=1).cpu().numpy()
    
    return errors, latent