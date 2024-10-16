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
except: pass

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
    layers = [[50] * (j + 1) for j in range(3)]

    # Clustering
    param_grid = {
        'n_clusters': [len(centers)]
    }
    KmeansExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_kmeans'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # VirtualTwins
    param_grid = {
        'n_clusters': [len(centers)],
        'max_depth': [3, 5, None],
        'min_samples_split': [6, 12, 24],
    }
    VirtualTwinsExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_twins'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    # NTC Competing risk
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers_surv': layers,
        'representation': [10, 25, 50],
        'k': [len(centers)],
        'layers' : layers,
        'act': ['Tanh'],
        'gamma': [0],
        'correct' : [True]
    }
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_ntc'.format(mode, random_seed, len(centers))).train(x, t, e, a)
    if mode == 'obs' and (len(centers) == 3):
        for size in [2, 4]:
            param_grid['k'] = [size]
            CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_ntc_k={}'.format(mode, random_seed, len(centers), size)).train(x, t, e, a)
        param_grid['k'] = [len(centers)]

    param_grid['gamma'] = [0]
    param_grid['correct'] = [False]
    CNSCExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_ntc+uncorrect'.format(mode, random_seed, len(centers))).train(x, t, e, a)


    # CMHE
    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [1],
        'g': [len(centers)],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_cmhe+g'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [len(centers)],
        'g': [1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_cmhe+k'.format(mode, random_seed, len(centers))).train(x, t, e, a)

    param_grid = {
        'epochs': [max_epochs],
        'learning_rate' : [1e-3, 1e-4],
        'batch': [100, 250],

        'layers': layers,
        'k': [len(centers) // 2 + 1],
        'g': [len(centers) // 2 + 1],
    }
    CMHEExperiment.create(param_grid, n_iter = grid_search, path = 'Results_ntc/generate_{}={}+{}_cmhe+gk'.format(mode, random_seed, len(centers))).train(x, t, e, a)