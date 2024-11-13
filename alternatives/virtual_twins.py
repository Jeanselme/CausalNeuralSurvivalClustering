from sklearn.cluster import KMeans
from sksurv.tree import SurvivalTree

import numpy as np
import pandas as pd

class VirtualTwins:

    def __init__(self, **params):
        self.cluster = params.pop('n_clusters')
        self.params = params
        self.fitted = False

    def interpolate(self, survival, times):
        predictions = pd.DataFrame([np.nan] * len(times), index = times)

        survival = pd.concat([survival, predictions]).sort_index(kind = 'stable').bfill().ffill()
        survival = survival[~survival.index.duplicated(keep='first')]

        return survival.loc[times].values.T

    def fit(self, x, t, e, a, random_state = 100, **args):
        # Fit a survival tree on each group
        self.treated = SurvivalTree(**self.params, random_state = random_state).fit(x[a == 1], np.array(list(zip(e[a == 1], t[a == 1])), dtype = [('e', bool), ('t', float)]))
        self.untreated = SurvivalTree(**self.params, random_state = random_state).fit(x[a == 0], np.array(list(zip(e[a == 0], t[a == 0])), dtype = [('e', bool), ('t', float)]))

        # Cluster estimated treatment effect
        treated = pd.DataFrame(self.treated.predict_survival_function(x, return_array = True), columns = self.treated.unique_times_)
        untreated = pd.DataFrame(self.untreated.predict_survival_function(x, return_array = True), columns = self.untreated.unique_times_)

        self.times = np.linspace(0, np.max(t), 100)
        treatment_effect = self.interpolate(treated.T, self.times) - self.interpolate(untreated.T, self.times)
        self.kmeans = KMeans(self.cluster, random_state = random_state).fit(treatment_effect)

        self.save_clusters = {c: {
                                    'treated': treated[self.kmeans.labels_ == c].mean(axis = 0),
                                    'untreated': untreated[self.kmeans.labels_ == c].mean(axis = 0)
                                } for c in range(self.cluster)}
        
        self.fitted = True
        return self    

    def compute_nll(self, x, t, e, a):
        if not self.fitted:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `_eval_nll`.")
        # Minus because wants to maximise the c-index
        return - self.treated.score(x[a == 1], np.array(list(zip(e[a == 1], t[a == 1])), dtype = [('e', bool), ('t', float)])) - self.untreated.score(x[a == 0], np.array(list(zip(e[a == 0], t[a == 0])), dtype = [('e', bool), ('t', float)]))

    def predict_survival(self, x, t, a = 1):
        a = np.array([a] * len(x)).astype(int) if isinstance(a, int) else np.array(a).astype(int)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            treated = self.interpolate(pd.DataFrame(self.treated.predict_survival_function(x, return_array = True), columns = self.treated.unique_times_).T, t)
            untreated = self.interpolate(pd.DataFrame(self.untreated.predict_survival_function(x, return_array = True), columns = self.untreated.unique_times_).T, t)
            treated[a == 0] = untreated[a == 0]
            return treated
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_survival`.")

    def predict_alphas(self, x):
        if self.fitted:
            treated = pd.DataFrame(self.treated.predict_survival_function(x, return_array = True), columns = self.treated.unique_times_).T
            untreated = pd.DataFrame(self.untreated.predict_survival_function(x, return_array = True), columns = self.untreated.unique_times_).T
            treatment_effect = self.interpolate(treated, self.times) - self.interpolate(untreated, self.times)
            results = np.zeros((len(x), self.cluster))
            results[np.arange(len(x)), self.kmeans.predict(treatment_effect)] = 1
            return results
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                        "model using the `fit` method on some training data " +
                        "before calling `predict_alphas`.")
    
    def treatment_effect_cluster(self, t):
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            t = np.array(t)
            return np.vstack([self.interpolate(self.save_clusters[c]['treated'], t) - self.interpolate(self.save_clusters[c]['untreated'], t)
                                   for c in self.save_clusters]).T
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `treatment_effect_cluster`.")