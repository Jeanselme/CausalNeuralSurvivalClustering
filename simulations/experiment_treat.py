# Description: Script to experiment with different treatment rates

import sys
from experiment import *
from generate import *

# Open dataset
random_seed = 42
mode = sys.argv[1]

try:
    if int(mode) == 0:
        mode = 'rand' 
    elif int(mode) == 1:
        mode = 'obs'
except: pass

os.makedirs("Results/treat/", exist_ok=True)

print("Script running experiments on generated data with seed =", random_seed)
for percentage_treatment in [0.25, 0.75]:
    x, a, t, e, _ = generate(random_seed, mode = mode, percentage_treatment = percentage_treatment) 

    # Normalise data
    x, t, e, a = StandardScaler().fit_transform(x.values).astype(float),\
                t.values.astype(float), e.values.astype(int), a.values.astype(int)

    # Hyperparameters and evaluations
    max_epochs = 1000
    grid_search = 100
    layers = [[50] * j for j in range(4)]

    # Clustering
    param_grid = {
        'n_clusters': [3]
    }
    KmeansExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_kmeans'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)

    # VirtualTwins
    param_grid = {
        'n_clusters': [3],
        'max_depth': [3, 5, None],
        'min_samples_split': [6, 12, 24],
    }
    VirtualTwinsExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_twins'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)

    # NTC Competing risk
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers_surv': layers,
        'representation': [10, 25, 50],
        'k': [3],
        'layers_assignment' : layers,
        'layers_treat': layers,
        'act': ['Tanh'],
        'correct' : [True]
    }
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_ntc'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)

    param_grid['correct'] = [False]
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_ntc+uncorrect'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)


    # CMHE
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [1],
        'g': [3],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_cmhe+g'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [3],
        'g': [1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}={}+{}_cmhe+k'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [3 // 2 + 1],
        'g': [3 // 2 + 1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/treat/{}+{}={}_cmhe+gk'.format(mode, random_seed, percentage_treatment)).train(x, t, e, a)