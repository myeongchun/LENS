import torch
import torch.nn as nn

class FCAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(FCAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), 
            nn.Sigmoid()
        )
    
    def forward(self, x): 
        z = self.encode(x)
        return self.decoder(z), z
    
    def encode(self, x): 
        return self.encoder(x)

    def weight_reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()