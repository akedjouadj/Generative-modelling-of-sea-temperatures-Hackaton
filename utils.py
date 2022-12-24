import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

device = 'cpu' 

'''def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

    Args:
    model: A PyTorch model instance that represents a 
      time-dependent score-based model.
    x: A mini-batch of training data.    
    marginal_prob_std: A function that gives the standard deviation of 
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(100, device=x.device) * (1. - eps) + eps  
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None] # on génère des données pertubées x(t), de loi P0t(x(t) sachant x(0)) pour toutes les valeurs de t
    score = model(perturbed_x, random_t) # on évalue le vecteur de score, shape (64, 1, 64, 6)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3))) 
    return loss'''


'''def fit(score_model, data_loader, marginal_prob_std, n_epochs = 50, batch_size = 64, lr = 1e-4):
    
    optimizer = Adam(score_model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        avg_loss = 0.
        num_items = 0
        for x in data_loader:
            x = x.to(device)    
            loss = loss_fn(score_model, x, marginal_prob_std)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
            # Print the averaged training loss so far.
        print('epoch : ', epoch, 'Average Loss: {:5f}'.format(avg_loss / num_items))
    torch.save(score_model.state_dict(), 'parameters/parameters_.pth')
    return score_model'''


def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff,
                           noise, 
                           batch_size=16, 
                           num_steps=500, 
                           device='cpu', 
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    num_steps: The number of sampling steps. 
      Equivalent to the number of discretized time steps.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    eps: The smallest time step for numerical stability.

    Returns:
    Samples.    
    """
    t = torch.ones(batch_size, device=device)
    # on simule batch_size vecteurs de 6 gaussiennes de paramètre sigma_t pout t=T=1, temps final
    #init_x = torch.tensor(noise)* marginal_prob_std(t)[:, None]
    init_x = torch.randn(batch_size, 6, device=device) * marginal_prob_std(t)[:, None]
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    x = init_x
    with torch.no_grad():
        for i, time_step in enumerate(time_steps):      
            batch_time_step = torch.ones(1, device=device) * time_step
            g = diffusion_coeff(batch_time_step)
            mean_x = x + (g**2) * score_model(x, batch_time_step)[0,0,:,:] * step_size
            x = mean_x + torch.sqrt(step_size) * g * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
    return mean_x


class ScoreNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, embed_dim=6):

        super().__init__()

        self.dense1 = nn.Sequential(
        nn.Linear(in_features=embed_dim, out_features=32),
        nn.LeakyReLU(negative_slope=0.2)
        )

        self.dense2 = nn.Sequential(
        nn.Linear(in_features=32, out_features=64),
        nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dense3 = nn.Sequential(
        nn.Linear(in_features=64, out_features=128),
        nn.LeakyReLU(negative_slope=0.2)
        )

        ##### Decoding

        self.dense4 = nn.Sequential(
        nn.Linear(in_features=128, out_features=64),
        nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dense5 = nn.Sequential(
        nn.Linear(in_features=64, out_features=32),
        nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.dense6 = nn.Sequential(
        nn.Linear(in_features=32, out_features=6),
        nn.LeakyReLU(negative_slope=0.2)
        )
        
        self.marginal_prob_std = marginal_prob_std
        



    def forward(self, x, t): 
        """output : shape = x.shape
        self.marginal_prob_std(t)[:, None, None, None] : shpae = (len(t), 1, 1, 1)"""

        output = self.dense1(x) 
        output = self.dense2(output) 
        output = self.dense3(output) 
        output = self.dense4(output) 
        output = self.dense5(output) 
        output = self.dense6(output)

        
        # Normalize output
        output = output / self.marginal_prob_std(t)[:, None, None, None]
        return output
