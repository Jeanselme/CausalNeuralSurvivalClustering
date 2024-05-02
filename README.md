# Causal Neural Survival Clustering
This repository allows to reproduce the results in [Causal Neural Survival Clustering]()
A micture of treatments effects parametrised by monotone neural networks to model the cumulative incidence function of each group under treatment and control regimes.

## Model
The model consists of three neural networks for estimating the survival distributions: M models the cumulative incidence function under both treatmen regimes, C assigns each patient to the different clusters, and W computes the likelihood to receive treatment to adjust the likelihood.  

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

A full example with analysis is provided in `examples/Causal Neural Survival Clustering on Seer Dataset.ipynb`.

## Reproduce paper's results
To reproduce the paper's results:

0. Clone the repository with dependencies: `git clone git@github.com:Jeanselme/NeuralTreatment.git --recursive`.
1. Create a conda environment with all necessary libraries `pycox`, `lifelines`, `pysurvival`.
2. Add path `export PYTHONPATH="$PWD:$PWD/DeepSurvivalMachines:$PYTHONPATH"`.
3. Run `examples/experiment_cnsc.py SEER`.
5. Analysis using `examples/Analysis CNSC.ipynb`.

## Compare to a new method
Adding a new method consists in adding a child to `Experiment` in `experiment.py` with functions to compute the nll and fit the model.
Then, add the method in `examples/experiment_cnsc.py` and follow the previous point. 

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