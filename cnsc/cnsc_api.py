from cnsc.cnsc import CausalNeuralSurvivalClusteringTorch
from cnsc.losses import *
from cnsc.utilities import train_cnsc

import torch
import numpy as np
from tqdm import tqdm

class CausalNeuralSurvivalClustering:

  def __init__(self, cuda = torch.cuda.is_available(), correct = True, **params):
    """
    Initialise model 

    Args:
        cuda (bool, optional): Use cuda (0: No, 1: Yes for all, 2: Push each (slower but allow larger data)). Defaults to torch.cuda.is_available().
        correct (bool, optional): Correct the loss with IPTW. Defaults to True.
        **params: Parameters for the model. See cnsc/CausalNeuralSurvivalClusteringTorch for more details.
    """
    self.params = params
    self.fitted = False
    self.cuda = cuda
    self.correct = correct

  def _gen_torch_model(self, inputdim, optimizer):
    model = CausalNeuralSurvivalClusteringTorch(inputdim, 
                                     **self.params,
                                     optimizer = optimizer).double()
    if self.cuda > 0:
      model = model.cuda()
    return model

  def fit(self, x, t, e, a, vsize = 0.15, val_data = None,
          optimizer = "Adam", random_state = 100, **args):
    """
    Fit the model

    Args:
        x (Numpy float array: n * # cov)): Covariates.
        t (Numpy float array: n): Times of event.
        e (Numpy bool array: n): Indicator event (0 is censored).
        a (Numpy bool array: n): Treatment indicator.
        vsize (float, optional): Spli train data. Defaults to 0.15.
        val_data ((x, t, e, a), optional): Set of data to use for validation. Defaults use vsize subset of training.
        optimizer (str, optional): Training optimiser. Defaults to "Adam".
        random_state (int, optional): Random seed. Defaults to 100.
    """
    processed_data = self._preprocess_training_data(x, t, e, a,
                                                   vsize, val_data,
                                                   random_state)
    x_train, t_train, e_train, a_train, x_val, t_val, e_val, a_val = processed_data

    model = self._gen_torch_model(x_train.size(1), optimizer)
    if self.correct:
      model = train_cnsc(model,
                          x_train, t_train, e_train, a_train,
                          x_val, t_val, e_val, a_val, cuda = self.cuda == 2, loss_f = propensity_loss,
                          **args)
    model = train_cnsc(model,
                      x_train, t_train, e_train, a_train,
                      x_val, t_val, e_val, a_val, cuda = self.cuda == 2, 
                      loss_f = total_loss, correct = self.correct,
                      **args)

    self.torch_model = model.eval()
    self.fitted = True
    return self    

  def compute_nll(self, x, t, e, a):
    """
      Compute the negative log-likelihood associated with data
    """
    if not self.fitted:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `_eval_nll`.")
    processed_data = self._preprocess_training_data(x, t, e, a, 0, None, 0)
    _, _, _, _, x_val, t_val, e_val, a_val = processed_data
    if self.cuda == 2:
      x_val, t_val, e_val, a_val = x_val.cuda(), t_val.cuda(), e_val.cuda(), a_val.cuda()

    loss = total_loss(self.torch_model, x_val, t_val, e_val, a_val, correct = self.correct)
    return loss.item()

  def predict_propensity(self, x):
    """
      Predict the propensity of treatment for covariates
    """
    x = self._preprocess_test_data(x)
    return self.torch_model.propensity(x).detach().cpu().numpy()

  def predict_survival(self, x, t, a = 1):
    """
    Predict survival function for the covariates at the time t under treatment regime a.

    Args:
        x (Numpy array): Covariates.
        t (Float or list): Time to evaluate (does not match x dimensions).
        a (bool, optional): Treatment regime. Defaults to 1.

    Returns:
        Numpy array #points * #evaluation time: Survival at different horizons.
    """
    x = self._preprocess_test_data(x)
    a = torch.tensor(a).int().repeat((x.shape[0], 1)).to(x.device) if isinstance(a, int) else torch.tensor(a).int()
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      scores = []
      for t_ in t:
        t_ = torch.DoubleTensor([t_] * len(x)).to(x.device)
        log_alphas, log_sr, _ = self.torch_model(x, t_)
        scores.append(np.take_along_axis(torch.exp(log_alphas + log_sr).sum(2).detach().cpu().numpy(), 
                                         a.cpu().numpy().reshape((-1, 1)), axis = 1))
      return np.concatenate(scores, axis = 1)
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_survival`.")

  def predict_alphas(self, x):
    """
    Predict the cluster assignmnent associated with the covariates.

    Args:
        x (Numpy float array): Covariates.

    Returns:
        Numby array #points * #clusters: Proability of being in each cluster.
    """
    x = self._preprocess_test_data(x)
    if self.fitted:
      log_alphas, _, _ = self.torch_model(x, torch.zeros(len(x), dtype = torch.double).to(x.device))
      return log_alphas[:, 0, :].exp().detach().cpu().numpy() # In this model alpha is independent of risk
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `predict_alphas`.")

  def treatment_effect_cluster(self, t):
    """
    Compute the treatment effect for each cluster

    Args:
        t (Float or list of float): Time of evaluation.

    Returns:
        Numpy array #clusters * #evaluation: Treatment effect.
    """
    if not isinstance(t, list):
      t = [t]
    if self.fitted:
      t = torch.tensor(t).double()
      x = torch.zeros(len(t), self.torch_model.input_dim, dtype = torch.double) # + 1 for treatment
      # Push on the right device
      if self.cuda > 0:
        x, t = x.cuda(), t.cuda()

      _, log_sr, _ = self.torch_model(x, t)
      return (log_sr[:, 1, :].exp() - log_sr[:, 0, :].exp()).detach().cpu().numpy()
    else:
      raise Exception("The model has not been fitted yet. Please fit the " +
                      "model using the `fit` method on some training data " +
                      "before calling `treatment_effect_cluster`.")

  def feature_importance(self, x, t, e, a, n = 100):
    """
    Perform permutation test to compute the feature importance.
    Note that this appraoch does not take into the correlation strucutre betweeen covariates.

    Args:
        n (int, optional): Number of permutations. Defaults to 100.

    Returns:
        (Dict, Dict): Mean and std associated with each covariate.
    """
    global_nll = self.compute_nll(x, t, e, a)
    permutation = np.arange(len(x))
    performances = {j: [] for j in range(x.shape[1])}
    for _ in tqdm(range(n)):
      np.random.shuffle(permutation)
      for j in performances:
        x_permuted = x.copy()
        x_permuted[:, j] = x_permuted[:, j][permutation]
        performances[j].append(self.compute_nll(x_permuted, t, e, a))
    # If positive difference: global_nll > random_nll: global_ll < random_ll (you expect negative values)
    return {j: np.mean((global_nll - np.array(performances[j]))/abs(global_nll)) for j in performances}, \
           {j: 1.96 * np.std((global_nll - np.array(performances[j]))/abs(global_nll)) / np.sqrt(n) for j in performances}

  def _preprocess_training_data(self, x, t, e, a, vsize, val_data, random_state):
    idx = list(range(x.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    x_train, t_train, e_train, a_train = x[idx], t[idx], e[idx], a[idx]

    x_train = torch.from_numpy(x_train).double()
    t_train = torch.from_numpy(t_train).double()
    e_train = torch.from_numpy(e_train).double()
    a_train = torch.from_numpy(a_train).int()

    if val_data is None:

      vsize = int(vsize*x_train.shape[0])
      x_val, t_val, e_val, a_val = x_train[-vsize:], t_train[-vsize:], e_train[-vsize:], a_train[-vsize:]
      x_train, t_train, e_train, a_train = x_train[:-vsize], t_train[:-vsize], e_train[:-vsize], a_train[:-vsize]

    else:

      x_val, t_val, e_val, a_val = val_data

      x_val = torch.from_numpy(x_val).double()
      t_val = torch.from_numpy(t_val).double()
      e_val = torch.from_numpy(e_val).double()
      a_val = torch.from_numpy(a_val).int()

    if self.cuda == 1:
      x_train, t_train, e_train, a_train = x_train.cuda(), t_train.cuda(), e_train.cuda(), a_train.cuda()
      x_val, t_val, e_val, a_val = x_val.cuda(), t_val.cuda(), e_val.cuda(), a_val.cuda()

    return (x_train, t_train, e_train, a_train,
            x_val, t_val, e_val, a_val)

  def _preprocess_test_data(self, x):
    data = torch.from_numpy(x)
    if self.cuda:
      data = data.cuda()
    return data
