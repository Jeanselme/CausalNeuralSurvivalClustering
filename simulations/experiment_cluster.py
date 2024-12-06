# Description: Script to experiment with different number of clusters

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
    else:
        mode = 'inf'
except: pass

os.makedirs("Results/cluster/", exist_ok=True)

print("Script running experiments on generated data with seed =", random_seed)

centers_list = [([0, 2.25], [-2.25, -1], [2.25, -1]),
                ([0, 2.25], [-2.25, -1], [2.25, -1], [-3, 3], [4, 4])]

for centers in centers_list:
    x, a, t, e, _ = generate(random_seed, mode = mode, centers = centers) 

    # Normalise data
    x, t, e, a = StandardScaler().fit_transform(x.values).astype(float),\
                t.values.astype(float), e.values.astype(int), a.values.astype(int)

    # Hyperparameters and evaluations
    max_epochs = 1000
    grid_search = 100
    layers = [[50] * j for j in range(4)]

    # Clustering
    param_grid = {
        'n_clusters': [len(centers)]
    }
    KmeansExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_kmeans'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # VirtualTwins
    param_grid = {
        'n_clusters': [len(centers)],
        'max_depth': [3, 5, None],
        'min_samples_split': [6, 12, 24],
    }
    VirtualTwinsExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_twins'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # NTC Competing risk
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers_surv': layers,
        'representation': [10, 25, 50],
        'k': [len(centers)],
        'layers_assignment' : layers,
        'layers_treat': layers,
        'act': ['Tanh'],
        'correct' : [True]
    }
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_ntc'.format(mode, random_seed, len(centers))).train(x, t, e, a)
    if mode == 'obs' and (len(centers) == 3):
        for size in [2, 5, 6]:
            param_grid['k'] = [size]
            CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_ntc_k={}'.format(mode, random_seed, len(centers), size)).train(x, t, e, a)
        param_grid['k'] = [len(centers)]

    param_grid['correct'] = [False]
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_ntc+uncorrect'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # Parametrise for logistic regression on both treatment and assignment
    param_grid['correct'] = [True]
    param_grid['layers_assignment'] = [[]]
    param_grid['layers_treat'] = [[]]
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}+{}={}_ntc+linear'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # CMHE
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [1],
        'g': [len(centers)],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_cmhe+g'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [len(centers)],
        'g': [1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_cmhe+k'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [len(centers) // 2 + 1],
        'g': [len(centers) // 2 + 1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results/cluster/{}={}+{}_cmhe+gk'.format(mode, random_seed, len(centers))).train(x, t, e, a)