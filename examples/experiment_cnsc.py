
# # Comparsion models for competing risks
# In this script we train the different models for competing risks
import sys
from cnsc import datasets
from experiment import *

random_seed = 42

# Open dataset
dataset = sys.argv[1] # FRAMINGHAM, SEER, METABRIC
path = sys.argv[2]
fold = None
if len(sys.argv) > 3:
    fold = int(sys.argv[3])
print("Script running experiments on ", dataset)
x, a, t, e, covariates = datasets.load_dataset(dataset, path = path) 

# Hyperparameters and evaluations
max_epochs = 1000
grid_search = 50
layers = [[50] * (j + 1) for j in range(3)]

for k in range(1, 7):
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers_surv': layers,
        'representation': [10, 25, 50],
        'k': [k],
        'layers' : layers,
        'act': ['Tanh'],

        'correct': [True]
    }
    CNSCExperiment.create(param_grid, fold = fold, n_iter = grid_search, path = 'Results_cnsc/{}_cnsc+k={}'.format(dataset, k), random_seed = random_seed).train(x, t, e, a)

# CNSC Competing risk
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'layers_surv': layers,
    'representation': [10, 25, 50],
    'k': [2, 3, 4, 5],
    'layers' : layers,
    'act': ['Tanh'],
    'correct' : [True]
}
CNSCExperiment.create(param_grid, fold = fold, n_iter = grid_search, path = 'Results_ntc/{}_cnsc'.format(dataset), random_seed = random_seed).train(x, t, e, a)

param_grid['correct'] = [False]
CNSCExperiment.create(param_grid, fold = fold, n_iter = grid_search, path = 'Results_cnsc/{}_cnsc+uncorrect'.format(dataset), random_seed = random_seed).train(x, t, e, a)


# CMHE
param_grid = {
    'epochs': [max_epochs],
    'learning_rate' : [1e-3, 1e-4],
    'batch': [100, 250],

    'layers': layers,
    'k': [1, 2, 3],
    'g': [1, 2, 3],
}
CMHEExperiment.create(param_grid, fold = fold, n_iter = grid_search, path = 'Results_cnsc/{}_cmhe'.format(dataset), random_seed = random_seed).train(x, t, e, a)

param_grid['k'] = [2]
param_grid['g'] = [2]
CMHEExperiment.create(param_grid, fold = fold, n_iter = grid_search, path = 'Results_cnsc/{}_cmhe_kg'.format(dataset), random_seed = random_seed).train(x, t, e, a)

