from .utilities import *
from torch.autograd import grad
import torch.nn as nn
import torch

class CausalNeuralSurvivalClusteringTorch(nn.Module):

  def __init__(self, inputdim, layers = [100, 100, 100], act = 'ReLU',
               layers_surv = [100], representation = 50, 
               k = 3, dropout = 0., optimizer = "Adam", multihead = False):
    super(CausalNeuralSurvivalClusteringTorch, self).__init__()
    self.input_dim = inputdim
    self.k = k # Number clusters
    self.representation = representation # Latent input for clusters (centroid representation)
    self.dropout = dropout
    self.optimizer = optimizer

    # Assign points to cluster
    self.profile = create_representation(inputdim, layers + [self.k], act, self.dropout, last = nn.LogSoftmax(dim = 1)) # Assign each point to a cluster
    self.treatment = create_representation(inputdim, layers + [1], act, self.dropout, last = nn.Sigmoid())
    
    # Cluster of treatment responses
    self.latent = nn.ParameterList([nn.Parameter(torch.randn((1, self.representation))) for _ in range(self.k)])
    self.outcome = nn.ModuleList([create_representation_positive(1 + self.representation, layers_surv + [2]) for _ in range(self.k)]) if multihead else \
                   create_representation_positive(1 + self.representation, layers_surv + [2]) # Response under both
    
    self.forward = self.forward_multihead if multihead else self.forward_singlehead

  def forward_singlehead(self, x, horizon):
    # Compute proba cluster len(x) * 2 * k
    log_alphas = torch.zeros((len(x), 1, self.k), requires_grad = True).float().to(x.device) if (self.k == 1) else \
                 self.profile(x).unsqueeze(1) # Temperature for spasity
    tau_outcome = [horizon.clone().detach().requires_grad_(True).unsqueeze(1) for _ in range(self.k)] # Requires independent clusters
    
    latent = torch.cat([self.latent[i].repeat_interleave(len(x), dim = 0) for i in range(self.k)], 0)
    tau = torch.cat(tau_outcome, 0)
    logOutcome = tau * self.outcome(torch.cat((latent, tau), 1)) # Outcome at time t for both
    log_sr = - logOutcome.unsqueeze(-1) # cluster * len(x), 2, 1

    log_sr = torch.cat(torch.split(log_sr, len(x)), -1)  # Dim: Point * [Untreat, Treat] * Cluster
    return log_alphas, log_sr, tau_outcome

  def forward_multihead(self, x, horizon):
    # Compute proba cluster len(x) * 2 * k
    log_alphas = torch.zeros((len(x), 1, self.k), requires_grad = True).float().to(x.device) if (self.k == 1) else \
                 self.profile(x).unsqueeze(1) # Temperature for spasity

    # For each treatment regime
    log_sr = []
    tau_outcome = [horizon.clone().detach().requires_grad_(True).unsqueeze(1) for _ in range(self.k)] # Requires independent clusters
    self.latent._size = self.k
    for outcome, latent, tau in zip(self.outcome, self.latent, tau_outcome):
      latent = latent.repeat_interleave(len(x), dim = 0) # Shape: len(x)

      # Compute survival distribution for each distribution 
      logOutcome = tau * outcome(torch.cat((latent, tau), 1)) # Outcome at time t for both
      log_sr.append(- logOutcome.unsqueeze(-1)) # len(x), 2, 1

    log_sr = torch.cat(log_sr, -1)  # Dim: Point * [Untreat, Treat] * Cluster
    return log_alphas, log_sr, tau_outcome
  
  def propensity(self, x):
    return self.treatment(x)[:, 0]
  
  def gradient(self, outcomes, horizon, e, a):
    # Avoid computation of all gradients by focusing on the one used later
    return torch.cat(grad([- outcomes[:, regime][(a == regime) & (e == 1)].sum() for regime in a.unique()], horizon, create_graph = True), 1).clamp_(1e-10)
