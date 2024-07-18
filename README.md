# Causal Neural Survival Clustering
This repository allows to use the model and reproduce the results introduced in [Causal Neural Survival Clustering]().
This models aim to uncover subgroups of patients with different treatment effects.
Each patient is assigned to a subgroup characterised by two neural networks modelling survival under treatment and control regimes.


## Model
The model consists of three neural networks for estimating the survival distributions: M models the cumulative incidence function under both treatment regimes given a latent representation for each cluster, G assigns each patient to the different subgroups, and W computes the probability to receive treatment to adjust the likelihood for observational study.

![Model](./images/ntc.png)

## How to use ?
To use the model, one needs to execute:
```python
from cnsc import CausalNeuralSurvivalClustering
model = CausalNeuralSurvivalClustering()
model.fit(x, t, e, a)
model.predict_risk(x, risk = 1)
```
With `x` the covarites, `t` the time of end of follow-up, `e` the associated cause (the model allows for competing risks) and `a` the assigned treatment.

A full example with analysis is provided in `examples/Causal Neural Survival Clustering on METABRIC Dataset.ipynb` using a publicly available dataset for reproducibility.

## Reproduce paper's results
To reproduce the paper's results:

0. Clone the repository with dependencies: `git clone git@github.com:Jeanselme/CausalNeuralSurvivalClustering.git --recursive`.
1. Create a conda environment with all necessary libraries `pycox`, `lifelines`, `pysurvival`.
2. Add path `export PYTHONPATH="$PWD:$PWD"`.
3. Run `examples/experiment_cnsc.py SEER`.
5. Analysis using `examples/Analysis CNSC.ipynb`.

## Compare to a new method
Adding a new method consists in adding a child to `Experiment` in `experiment.py` with functions to compute the nll and fit the model.
Then, add the method in `examples/experiment_cnsc.py` and follow the previous point. 

# Setup
## Structure
We followed the same architecture than the [DeepSurvivalMachines](https://github.com/autonlab/DeepSurvivalMachines) repository with the model in `cnsc/` - only the api should be used to test the model. Examples are provided in `examples/`. 

## Clone
```
git clone git@github.com:Jeanselme/CausalNeuralSurvivalClustering.git
```

## Requirements
The model relies on `pytorch >= 2.0`, `numpy` and `tqdm`.  
To run the set of experiments `auton_survival`, `pycox`, `lifelines`, `pysurvival` are necessary.