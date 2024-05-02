from cnsc.losses import total_loss
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy

class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features, bias = False):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_weight)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.log_weight)
            bound = np.sqrt(1 / np.sqrt(fan_in))
            nn.init.uniform_(self.bias, -bound, bound)
        self.log_weight.data.abs_().sqrt_()

    def forward(self, input):
        if self.bias is not None:
            return nn.functional.linear(input, self.log_weight ** 2, self.bias)
        else:
            return nn.functional.linear(input, self.log_weight ** 2)


def create_representation_positive(inputdim, layers, last = nn.Softplus()):
    modules = []
    act = nn.Tanh()
    
    prevdim = inputdim
    for hidden in layers:
        modules.append(PositiveLinear(prevdim, hidden, bias=True))
        modules.append(act)
        prevdim = hidden

    if last is not None:
        modules[-1] = last

    return nn.Sequential(*modules)

def create_representation(inputdim, layers, activation, dropout = 0., last = None):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(act)
        prevdim = hidden

    if last is not None:
        modules[-1] = last

    return nn.Sequential(*modules)

def get_optimizer(models, lr, optimizer, **kwargs):
    parameters = list(models.parameters())

    if optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr, **kwargs)
    elif optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=lr, **kwargs)
    elif optimizer == 'RMSProp':
        return torch.optim.RMSprop(parameters, lr=lr, **kwargs)
    else:
        raise NotImplementedError('Optimizer '+optimizer+' is not implemented')

def train_cnsc(model,
              x_train, t_train, e_train, m_train,
              x_valid, t_valid, e_valid, m_valid,
              n_iter = 1000, lr = 1e-3, weight_decay = 0.001, gamma = 0,
              bs = 100, patience_max = 10, cuda = False, 
              loss_f = total_loss, correct = True):
    optimizer = get_optimizer(model, lr, model.optimizer, weight_decay = weight_decay)
    patience, best_loss = 0, np.inf
    best_param = deepcopy(model.state_dict())
    
    nbatches = int(x_train.shape[0]/bs) + 1
    index = np.arange(len(x_train))
    t_bar = tqdm(range(n_iter))
    for i in t_bar:
        np.random.shuffle(index)
        model.train()

        train_loss = 0
        # Train survival model
        for j in range(nbatches):
            xb = x_train[index[j*bs:(j+1)*bs]]
            tb = t_train[index[j*bs:(j+1)*bs]]
            ab = m_train[index[j*bs:(j+1)*bs]]
            eb = e_train[index[j*bs:(j+1)*bs]]
            
            if xb.shape[0] == 0:
                continue

            if cuda:
                xb, tb, eb, ab = xb.cuda(), tb.cuda(), eb.cuda(), ab.cuda()

            optimizer.zero_grad()
            loss = loss_f(model,
                              xb,
                              tb,
                              eb,
                              ab, 
                              correct, gamma) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / nbatches

        model.eval()
        xb, tb, eb, ab = x_valid, t_valid, e_valid, m_valid
        if cuda:
            xb, tb, eb, ab = xb.cuda(), tb.cuda(), eb.cuda(), ab.cuda()

        valid_loss = loss_f(model,
                                xb,
                                tb,
                                eb,
                                ab,
                                correct, gamma).item() 
        mssg = "Loss: {:.3f} - Training: {:.3f} ".format(valid_loss, train_loss)
        t_bar.set_description(mssg)
        if valid_loss < best_loss:
            patience = 0
            best_loss = valid_loss
            best_param = deepcopy(model.state_dict())
        elif patience == patience_max:
            break
        else:
            patience += 1
    model.load_state_dict(best_param)
    return model
