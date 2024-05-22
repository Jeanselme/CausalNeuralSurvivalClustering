import torch


def total_loss(model, x, t, e, a, correct = True):
  """
    Compute the log likelihood assocaited with data (IPTW if correct is True)
  """
  # Go through network
  log_alphas, log_sr, taus = model.forward(x, t)
  log_hr = model.gradient(log_sr, taus, e, a).log() 
  log_sr = log_alphas + log_sr

  if correct:
    weights = model.propensity(x).detach().clone().clamp_(0.01, 0.99)
    weights[a == 0] = 1 - weights[a == 0]
    weights = 1. / weights
  else:
    weights = torch.ones(len(x))
  
  # Weighted factual likelihood
  error = 0
  for regime in [0, 1]:
    error -= (weights[(e == 0) & (a == regime)] * torch.logsumexp(log_sr[:, regime][(e == 0) & (a == regime)], dim = 1)).sum() # Sum over all cluster and then across patient
    error -= (weights[(e == 1) & (a == regime)] * torch.logsumexp(log_sr[:, regime][(e == 1) & (a == regime)] + log_hr[(e == 1) & (a == regime)], dim = 1)).sum() # LogSumExp over all cluster then across patient

  return error / torch.sum(weights)

def propensity_loss(model, x, t, e, a, correct = True, gamma = 0):
  propensity = model.propensity(x)
  return torch.nn.BCELoss()(propensity, a.double())