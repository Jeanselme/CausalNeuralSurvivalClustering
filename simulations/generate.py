import numpy as np
import pandas as pd
from scipy.stats import gompertz, bernoulli
from sklearn.datasets import make_blobs

shape = {
        -2: lambda d, p, x: d / 2 + np.abs((p * x)[:, :5]).sum(1) / 5,
        -1: lambda d, p, x: d / 2 + np.abs((p * x)[:, 5:]).sum(1) / 5,
        0 : lambda p, x: np.abs((p * x)[:, 5:]).sum(1) / 5,
        1 : lambda d, p, x: d / 2 + ((p * x)[:, 5:] ** 2).sum(1) / 5,
        2 : lambda d, p, x: d / 2 + ((p * x)[:, :5] ** 2).sum(1) / 5
    }

def generate(random_seed = 42, size = 3000, mode = 'rand', centers = ([0, 2.25], [-2.25, -1], [2.25, -1]), homogenous = False, proportions = None, percentage_treatment = 0.5):
    """
        Generate two clusters with different survival profiles
        Concatenate clusters and covariates for X

        homogenous - Draw additional covariates from normal distribution
        proportions - Change proportions of the differtent clusters of interst

        return x, t, e, betas, z
    """
    # Set seed for the experiment
    np.random.seed(random_seed)

    # Compute proportions
    if proportions is not None:
        assert len(proportions) == len(centers), "Proportions must be the same length as centers"
        size_group = (np.array(proportions) * size).astype(int)
    else:
        size_group = size

    # Data - Generate blobs
    x, z = make_blobs(n_samples = size_group, n_features = 2, centers = centers, random_state = random_seed)
    if homogenous:
        x = np.column_stack([x] + [np.random.normal(size = size) for _ in range(8)]) 
    else:
        x_additional, _ = make_blobs(n_samples = size, n_features = 8, centers = 4, center_box = (-1, 1), random_state = random_seed + 1)
        x = np.column_stack([x] + [x_additional]) 

    # Generate parameters for each gompretz cause specific hazards
    parameters = {event: np.array([np.random.normal(size = 10) for _ in np.unique(z)]) for event in [-1, -2, 1, 2]}
    cluster_shape = {event: np.abs(np.random.normal(size = len(np.unique(z)))) for event in [-1, -2, 1, 2]}

    # Generate the parameters
    sh1 = shape[1](cluster_shape[1][z], parameters[1][z], x)
    sh2 = shape[2](cluster_shape[2][z], parameters[2][z], x)
    sc1 = shape[-1](cluster_shape[-1][z], parameters[-1][z], x)
    sc2 = shape[-2](cluster_shape[-2][z], parameters[-2][z], x)

    # Generate the event for both clusters
    outcomes = pd.DataFrame({'untreated': gompertz.rvs(sh1, sc1), 
                             'treated': gompertz.rvs(sh2, sc2)})

    # Create model censoring
    censoring_beta = np.random.normal(size = 10)
    censoring = gompertz.rvs(shape[0](censoring_beta, x), 2)

    # Assign the outcomes following Bernoulli draw 
    # DO LAST TO ENSURE THAT ONLY TREATMENT IS DIFFERENT WHEN GENERATE WITH A GIVEN SEED
    if mode == 'rand':
        assignment_dig = np.array([percentage_treatment] * size)
        outcomes['treatment'] = bernoulli.rvs(assignment_dig)
    else:
        # Treatment is proportional to treatment response
        assignment = (x ** 2).sum(1)
        assignment_dig = np.digitize(assignment, bins = np.quantile(assignment, np.linspace(0, 1, 100))) / 100
        if mode == "inf":
            percentage_treatment = np.array([0.25, 0.5, 0.75])
            percentage_treatment = percentage_treatment[z]
        assignment_dig = np.clip(assignment_dig * 2 * percentage_treatment, a_min = 0.1, a_max = 0.9)
        outcomes['treatment'] = bernoulli.rvs(assignment_dig) # Draw from uniform with proba percentage

    # Assign outcomes
    outcomes['duration'] = outcomes['treatment'] * outcomes['treated'] + (1 - outcomes['treatment']) * outcomes['untreated']
    outcomes['event'] = censoring >= outcomes['duration']
    outcomes['duration'] = outcomes['event'] * outcomes['duration'] + (1 - outcomes['event']) * censoring
    outcomes['cluster'] = z

    return pd.DataFrame(x), outcomes['treatment'], outcomes['duration'], outcomes['event'], (cluster_shape, parameters, outcomes, assignment_dig)

def compute_cif(x, z, cluster_shape, betas, times):
    # As we know each cause specific hazard, we can model the associated gompretz
    shape_x = {event: shape[event](cluster_shape[event][z], betas[event][z], x.values) for event in [-1, -2, 1, 2]}

    cif_untreat = pd.DataFrame(np.vstack([gompertz.cdf(times, sh, sc) for sh, sc in zip(shape_x[1], shape_x[-1])]), columns = times)
    cif_treat = pd.DataFrame(np.vstack([gompertz.cdf(times, sh, sc) for sh, sc in zip(shape_x[2], shape_x[-2])]), columns = times)
    return pd.concat([cif_untreat, cif_treat], keys=['untreated','treated'], axis = 1)


def generate_linear(random_seed = 42, size = 3000, mode = 'rand'):
    """
        Generate two clusters with different survival profiles
        Concatenate clusters and covariates for X

        return x, t, e, betas, z
    """
    # Set seed for the experiment
    np.random.seed(random_seed)

    # Data - Generate 
    x, z = make_blobs(n_samples = size, n_features = 2, centers = ([0, 2.25], [-2.25, -1], [2.25, -1]), random_state = random_seed)
    x_additional, _ = make_blobs(n_samples = size, n_features = 8, centers = 4, center_box = (-1, 1), random_state = random_seed + 1)
    x = np.column_stack([x] + [x_additional]) 

    # Generate parameters for each gompretz cause specific hazards
    parameters = {event: np.array([np.random.normal(size = 10) for _ in np.unique(z)]) for event in [-1, -2, 1, 2]}
    cluster_shape = {event: np.abs(np.random.normal(size = 3)) for event in [-1, -2, 1, 2]}
    treatment = np.array([0.5, 1, 1.5]) # Treatment effect

    # Generate the parameters
    sh1 = shape[1](cluster_shape[1][z], parameters[1][z], x)
    sc1 = shape[-1](cluster_shape[-1][z], parameters[-1][z], x)

    # Generate the event for both clusters
    outcomes = pd.DataFrame({'untreated': gompertz.rvs(sh1, sc1), 
                             'treated': gompertz.rvs(sh1 * treatment[z], sc1)})

    # Create model censoring
    censoring_beta = np.random.normal(size = 10)
    censoring = gompertz.rvs(shape[0](censoring_beta, x), 2)

    # Assign the outcomes following Bernoulli draw 
    # DO LAST TO ENSURE THAT ONLY TREATMENT IS DIFFERENT WHEN GENERATE WITH A GIVEN SEED
    if mode == 'rand':
        assignment_dig = np.array([0.25] * size)
        outcomes['treatment'] = bernoulli.rvs(assignment_dig)
    else:
        # Treatment is proportional to treatment response
        assignment = (x ** 2).sum(1)
        assignment_dig = np.digitize(assignment, bins = np.quantile(assignment, np.linspace(0, 1, 100))) / 100
        assignment_dig = np.clip(assignment_dig * 0.5, a_min = 0.1, a_max = 0.9)
        outcomes['treatment'] = bernoulli.rvs(assignment_dig) # Draw from uniform with proba percentage

    # Assign outcomes
    outcomes['duration'] = outcomes['treatment'] * outcomes['treated'] + (1 - outcomes['treatment']) * outcomes['untreated']
    outcomes['event'] = censoring >= outcomes['duration']
    outcomes['duration'] = outcomes['event'] * outcomes['duration'] + (1 - outcomes['event']) * censoring
    outcomes['cluster'] = z

    return pd.DataFrame(x), outcomes['treatment'], outcomes['duration'], outcomes['event'], (cluster_shape, parameters, outcomes, assignment_dig)

def compute_cif_linear(x, z, cluster_shape, betas, times):
    # As we know each cause specific hazard, we can model the associated gompretz
    shape_x = {event: shape[event](cluster_shape[event][z], betas[event][z], x.values) for event in [-1, 1]}
    treatment = np.array([0.5, 1, 2]) # Treatment effect

    cif_untreat = pd.DataFrame(np.vstack([gompertz.cdf(times, sh, sc) for sh, sc in zip(shape_x[1], shape_x[-1])]), columns = times)
    cif_treat = pd.DataFrame(np.vstack([gompertz.cdf(times, sh * treat, sc) for sh, sc, treat in zip(shape_x[1], shape_x[-1], treatment[z])]), columns = times)
    return pd.concat([cif_untreat, cif_treat], keys=['untreated','treated'], axis = 1)