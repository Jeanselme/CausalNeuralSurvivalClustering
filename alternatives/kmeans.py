from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter

import numpy as np

class TEKMeans:

    def __init__(self, **params):
        self.params = params
        self.fitted = False

    def fit(self, x, t, e, a, random_state = 100, **args):
        # Cluster data given covariates
        self.kmeans = KMeans(**self.params, random_state = random_state).fit(x)
        assignment = self.kmeans.predict(x)

        # Non parametric estimate of the treatmente effect in each group
        self.treatment_effect = {}
        for c in np.unique(assignment):
            selection = assignment == c
            treated = selection & (a == 1)
            treated_kmf = KaplanMeierFitter().fit(t[treated], e[treated], label = "treated")

            untreated = selection & (a == 0)
            untreated_kmf = KaplanMeierFitter().fit(t[untreated], e[untreated], label = "untreated")

            self.treatment_effect[c] = {
                1: treated_kmf,
                0: untreated_kmf
            }
        
        self.fitted = True
        return self    

    def compute_nll(self, x, t, e, a, gamma = 0):
        if not self.fitted:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `_eval_nll`.")
        print("Not implemented")
        return 0

    def predict_survival(self, x, t, a = 1):
        a = np.array([a] * len(x)).astype(int) if isinstance(a, int) else np.array(a).astype(int)
        if not isinstance(t, list):
            t = [t]
        if self.fitted:
            scores = []
            assignment = self.kmeans.predict(x)
            outcomes = {c: 
                            {1: self.treatment_effect[c][1].survival_function_at_times(t),
                             0: self.treatment_effect[c][0].survival_function_at_times(t)}
                for c in self.treatment_effect}
            for t_ in t:
                scores.append(np.array([outcomes[assignment_][a_][t_] for assignment_, a_ in zip(assignment, a)]))
            return np.vstack(scores).T
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict_survival`.")

    def predict_alphas(self, x):
        if self.fitted:
            results = np.zeros((len(x), len(self.treatment_effect)))
            results[np.arange(len(x)), self.kmeans.predict(x)] = 1
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
            return np.vstack([self.treatment_effect[c][1].survival_function_at_times(t) 
                                   - self.treatment_effect[c][0].survival_function_at_times(t) 
                                   for c in range(len(self.treatment_effect))]).T
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `treatment_effect_cluster`.")