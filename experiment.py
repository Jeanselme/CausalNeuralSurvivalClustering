from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, train_test_split
import pandas as pd
import numpy as np
import pickle
import torch
import os
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location = 'cpu')
        else: 
            return super().find_class(module, name)

def from_surv_to_t(pred, times):
    """
        Interpolate pred for predictions at time

        Pred: Horizon * Patients
    """
    from scipy.interpolate import interp1d
    res = []
    for i in pred.columns:
        res.append(interp1d(pred.index, pred[i].values, fill_value = (1, pred[i].values[-1]), bounds_error = False)(times))
    return np.vstack(res)

class ToyExperiment():

    def train(self, *args):
        print("Toy Experiment - Results already saved")

class Experiment():

    def __init__(self, hyper_grid = None, n_iter = 100, fold = None, k = 5, 
                random_seed = 0, times = 100, path = 'results', save = True):
        self.hyper_grid = list(ParameterSampler(hyper_grid, n_iter = n_iter, random_state = random_seed) if hyper_grid is not None else [{}])
        self.random_seed = random_seed
        self.times = times
        self.k = k
        
        # Allows to reload a previous model
        self.all_fold = fold
        self.iter, self.fold = 0, 0
        self.best_hyper = {}
        self.best_model = {}
        self.best_nll = None

        self.path = path
        self.tosave = save

    @classmethod
    def create(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
                random_seed = 0, times = 100, path = 'results', force = False, save = True):
        if not(force):
            path = path if fold is None else path + '_{}'.format(fold)
            if os.path.isfile(path + '.csv'):
                return ToyExperiment()
            elif os.path.isfile(path + '.pickle'):
                print('Loading previous copy')
                try:
                    return cls.load(path+ '.pickle')
                except Exception as e:
                    print('ERROR: Reinitalizing object')
                    os.remove(path + '.pickle')
                    pass
                
        return cls(hyper_grid, n_iter, fold, k, random_seed, times, path, save)

    @staticmethod
    def load(path):
        file = open(path, 'rb')
        if torch.cuda.is_available():
            se = pickle.load(file)
            return se
        else:
            se = CPU_Unpickler(file).load()
            for model in se.best_model:
                if type(se.best_model[model]) is dict:
                    for m in se.best_model[model]:
                        se.best_model[model][m].cuda = False
                else:
                    se.best_model[model].cuda = False
            return se
        
    @classmethod
    def merge(cls, hyper_grid = None, n_iter = 100, fold = None, k = 5,
            random_seed = 0, times = 100, path = 'results', save = True):
        merged = cls(hyper_grid, n_iter, fold, k, random_seed, times, path, save)
        for i in range(k):
            path_i = path + '_{}.pickle'.format(i)
            if os.path.isfile(path_i):
                merged.best_model[i] = cls.load(path_i).best_model[i]
                merged.best_hyper[i] = cls.load(path_i).best_hyper[i]
            else:
                print('Fold {} has not been computed yet'.format(i))
        merged.fold = k # Nothing to run
        return merged

    @staticmethod
    def save(obj):
        with open(obj.path + '.pickle', 'wb') as output:
            try:
                pickle.dump(obj, output)
            except Exception as e:
                print('Unable to save object')
                
    def save_results(self, x, times):
        predictions = pd.DataFrame(np.nan, index = self.fold_assignment.index, columns = pd.MultiIndex.from_product([['treated', 'untreated'], self.times]))
        clusters = []

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            for a, treat in enumerate(['untreated', 'treated']):
                predictions.loc[index, treat] = self._predict_(self.best_model[i], x[index], times, a)
            # Propagate
            predictions = predictions.T.ffill().T
            clusters.append(pd.DataFrame(self._predict_cluster_(self.best_model[i], x[index]), index = index))
        clusters = pd.concat(clusters, axis = 0).loc[predictions.dropna().index]
        clusters.columns = pd.MultiIndex.from_product([['Assignment'], clusters.columns])

        if self.tosave:
            fold_assignment = self.fold_assignment.copy().to_frame()
            fold_assignment.columns = pd.MultiIndex.from_product([['Use'], ['Fold']])
            pd.concat([predictions, fold_assignment, clusters], axis = 1).to_csv(self.path + '.csv')

        return predictions, clusters

    def train(self, x, t, e, a):
        """
            Cross validation model

            Args:
                x (Dataframe n * d): Observed covariates
                t (Dataframe n): Time of censoring or event
                e (Dataframe n): Event indicator


            Returns:
                (Dict, Dict): Dict of fitted model and Dict of observed performances
        """
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)

        self.fold_assignment = pd.Series(np.nan, index = range(len(x)))
        if self.k == 1:
            kf = ShuffleSplit(n_splits = self.k, random_state = self.random_seed, test_size = 0.2)
        else:
            kf = StratifiedKFold(n_splits = self.k, random_state = self.random_seed, shuffle = True)

        # First initialization
        if self.best_nll is None:
            self.best_nll = np.inf
        for i, (train_index, test_index) in enumerate(kf.split(x, e + 2 * a)): # To ensure to split on both censoring and treatment
            self.fold_assignment[test_index] = i
            if i < self.fold: continue # When reload: start last point
            if not(self.all_fold is None) and (self.all_fold != i): continue
            print('Fold {}'.format(i))
            
            train_index, dev_index = train_test_split(train_index, test_size = 0.2, random_state = self.random_seed, stratify = e[train_index])
            dev_index, val_index   = train_test_split(dev_index,   test_size = 0.5, random_state = self.random_seed, stratify = e[dev_index])
            
            x_train, x_dev, x_val = x[train_index], x[dev_index], x[val_index]
            t_train, t_dev, t_val = t[train_index], t[dev_index], t[val_index]
            e_train, e_dev, e_val = e[train_index], e[dev_index], e[val_index]
            a_train, a_dev, a_val = a[train_index], a[dev_index], a[val_index]

            # Train on subset one domain
            ## Grid search best params
            for j, hyper in enumerate(self.hyper_grid):
                if j < self.iter: continue # When reload: start last point
                np.random.seed(self.random_seed)
                torch.manual_seed(self.random_seed)

                model = self._fit_(x_train, t_train, e_train, a_train, x_val, t_val, e_val, a_val, hyper.copy())
                nll = self._nll_(model, x_dev, t_dev, e_dev, a_dev, hyper.copy())
                if nll < self.best_nll:
                    self.best_hyper[i] = hyper
                    self.best_model[i] = model
                    self.best_nll = nll
                self.iter = j + 1
                self.save(self)
            self.fold, self.iter = i + 1, 0
            self.best_nll = np.inf
            self.save(self)

        if self.all_fold is None:
            self.save(self)
            return self.save_results(x, self.times)

    def _fit_(self, *params):
        raise NotImplementedError()

    def _nll_(self, *params):
        raise NotImplementedError()
    
    def likelihood(self, x, t, e, a):
        """
            Compute the nll over the different folds
            Data must match original index
        """
        x = self.scaler.transform(x)
        nll_fold = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            nll_fold[i] = self._nll_(model, x[index], t[index], e[index], a[index], self.best_hyper[i])

        return nll_fold

    def importance(self, x, t, e, a, **params):
        return None
    
class CMHEExperiment(Experiment):
    
    def _predict_cluster_(self, model, x):
        """
            Compute assignment of all points
        """
        cox = model.predict_latent_z(x)
        treatment = model.predict_latent_phi(x)
        return np.concatenate([cox.T * treatment[:, t] for t in range(treatment.shape[1])], 0).T

    def _fit_(self, x, t, e, a, x_val, t_val, e_val, a_val, hyperparameter):
        from auton_survival.models.cmhe import DeepCoxMixturesHeterogenousEffects

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        k = hyperparameter.pop('k')
        g = hyperparameter.pop('g')

        model = DeepCoxMixturesHeterogenousEffects(k = k, g = g, **hyperparameter) # Enforce the same number of cluster and treatment response
        model.fit(x, t, e, a, iters = epochs, batch_size = batch,
                learning_rate = lr, val_data = (x_val, t_val, e_val, a_val))
        return model

    def _nll_(self, model, x, t, e, a, *train):
        from auton_survival.models.cmhe.cmhe_utilities import test_step
        _, _, _, _, x, t, e, a = model._preprocess_training_data(x, t, e, a, vsize = 0, val_data = None, random_seed = 0)
        return test_step(model.torch_model[0], x, t, e, a, model.torch_model[1])

    def _predict_(self, model, x, times, a):
        a = [a] * len(x) if isinstance(a, int) else a 
        return model.predict_survival(x, np.array(a), times.tolist())
    
    def clusters(self, t):
        te_cluster = {}
        for i in self.best_model:
            model = self.best_model[i]
            te_cluster[i] = pd.DataFrame(model.predict_clusters(t.tolist(), a = 1) - model.predict_clusters(t.tolist(), a = 0)).ffill().to_numpy()

        return te_cluster

class CNSCExperiment(Experiment):

    def _predict_cluster_(self, model, x):
        return model.predict_alphas(x)
    
    def __preprocess__(self, t, save = False):
        if save:
            self.max_t = t.max()
        return t / self.max_t
    
    def save_results(self, x, times):
        return super().save_results(x, self.__preprocess__(times))

    def train(self, x, t, e, a):
        self.times = np.linspace(t.min(), t.max(), self.times) if isinstance(self.times, int) else self.times
        t_norm = self.__preprocess__(t, True)
        return super().train(x, t_norm, e, a)

    def _fit_(self, x, t, e, a, x_val, t_val, e_val, a_val, hyperparameter):  
        from cnsc import CausalNeuralSurvivalClustering

        epochs = hyperparameter.pop('epochs', 1000)
        batch = hyperparameter.pop('batch', 250)
        lr = hyperparameter.pop('learning_rate', 0.001)

        model = CausalNeuralSurvivalClustering(**hyperparameter)
        model.fit(x, t, e, a, n_iter = epochs, bs = batch, 
                lr = lr, val_data = (x_val, t_val, e_val, a_val))
        return model

    def _predict_(self, model, x, times, a):
        return model.predict_survival(x, times.tolist(), a)
    
    def _nll_(self, model, x, t, e, a, hyperparameter, *train):
        return model.compute_nll(x, t, e, a)

    def likelihood(self, x, t, e, a):
        t_norm = self.__preprocess__(t)
        return super().likelihood(x, t_norm, e, a)
    
    def importance(self, x, t, e, a, **params):
        """
            Compute the permutation importance of the different features
        """
        x = self.scaler.transform(x)
        t_norm = self.__preprocess__(t)
        importance = {}

        for i in self.best_model:
            index = self.fold_assignment[self.fold_assignment == i].index
            model = self.best_model[i]
            importance[i] = model.feature_importance(x[index], t_norm[index], e[index], a[index], **params)

        return importance
    
    def clusters(self, t):
        te_cluster = {}
        t_norm = self.__preprocess__(t)
        for i in self.best_model:
            model = self.best_model[i]
            te_cluster[i] = model.treatment_effect_cluster(t_norm.tolist())

        return te_cluster

class KmeansExperiment(Experiment):

    def _fit_(self, x, t, e, a, x_val, t_val, e_val, a_val, hyperparameter):  
        from alternatives import TEKMeans
        model = TEKMeans(**hyperparameter)
        model.fit(x, t, e, a)
        return model

    def _predict_(self, model, x, times, a):
        return model.predict_survival(x, times.tolist(), a)
    
    def clusters(self, t):
        te_cluster = {}
        for i in self.best_model:
            model = self.best_model[i]
            te_cluster[i] = model.treatment_effect_cluster(t.tolist())

        return te_cluster

    def _predict_cluster_(self, model, x):
        return model.predict_alphas(x)

    def _predict_(self, model, x, times, a):
        return model.predict_survival(x, times.tolist(), a)
    
    def _nll_(self, model, x, t, e, a, hyperparameter, *train):
        return model.compute_nll(x, t, e, a)
    
class VirtualTwinsExperiment(KmeansExperiment):

    def _fit_(self, x, t, e, a, x_val, t_val, e_val, a_val, hyperparameter):  
        from alternatives import VirtualTwins
        model = VirtualTwins(**hyperparameter)
        model.fit(x, t, e, a)
        return model