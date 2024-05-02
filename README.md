# Causal Neural Survival Clustering
This repository allows to reproduce the results in [Causal Neural Survival Clustering]()
A neural network approach to identifying treatments effect subgroups, leveraging monotone neural networks to model the cumulative incidence function of each group.

## Model
The model consists in two neural networks for estimating the survival distributions: one models the cumulative incidence function and the other the balance to ensure that they add up to one. These models rely on a latent parameters that represent each cluster and the **assigned treatment**. An additional model aims to assigns each point to one of these distributions. Finally to tackle the problem of non-random treatment assigment, the embedding is penalised by a Wasserstein distance between the different treatments' assignment distributions.

![Model](./images/ntc.png)

## How to use ?
To use the model, one needs to execute:
```python
from ntc import NeuralTreatmentCluster
model = NeuralTreatmentCluster(mask = mask)
model.fit(x, t, e)
model.predict_risk(x, risk = 1)
```
With `x` the covarites, `t` the time of end of follow-up, `e` the associated cause (the model allows for competing risks) and `mask` the covariates dimensions corresponding to the assigned treatment.

A full example with analysis is provided in `examples/Causal Neural Survival Clustering on METABRIC Dataset.ipynb`.

## Reproduce paper's results
To reproduce the paper's results:

0. Clone the repository with dependencies: `git clone git@github.com:Jeanselme/NeuralTreatment.git --recursive`.
1. Create a conda environment with all necessary libraries `pycox`, `lifelines`, `pysurvival`.
2. Add path `export PYTHONPATH="$PWD:$PWD/DeepSurvivalMachines:$PYTHONPATH"`.
3. Run `examples/experiment_ntc.py SEER`.
5. Analysis using `examples/Analysis NTC.ipynb`.

## Compare to a new method
Adding a new method consists in adding a child to `Experiment` in `experiment.py` with functions to compute the nll and fit the model.
Then, add the method in `examples/experiment_treatment_effect.py` and follow the previous point. 
`TODOs` have been added to make the addition of a new method easier.

# Setup
## Structure
We followed the same architecture than the [DeepSurvivalMachines](https://github.com/autonlab/DeepSurvivalMachines) repository with the model in `ntc/` - only the api should be used to test the model. Examples are provided in `examples/`. 

## Clone
```
git clone git@github.com:Jeanselme/NeuralTreatment.git --recursive
```

## Requirements
The model relies on `DeepSurvivalMachines`, `pytorch >= 2.0`, `numpy` and `tqdm`.  
To run the set of experiments `pycox`, `lifelines`, `pysurvival` are necessary.