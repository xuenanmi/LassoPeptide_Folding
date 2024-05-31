# Lasso Folding

This repository contains all the source code for analyzing the lasso folding molecular dynamics simulation data of 20 different lasso peptides.

## Features

- Featurization of MD trajectory data
- Construction and validation of Markov State Models (MSM)
- Thermodynamic and kinetic analysis
- Identification and clustering of kinetic folding pathways

## Repository Structure

### FeatureCalculation_and_MSM

Contains scripts to:
- Featurize MD trajectory data using all pair-wise residue-residue distance <br>
  `feature_calculation.py`, calculate all pair-wise residue-residue distances of all trajectory data
  
- Construct and validate Markov State Models for each lasso peptideue distances <br>
  `optimize_MSM_hyperparameter.py`, perform a grid search to optimize the hyperparameter of MSM based on VAMP2 score <br> and construct MSM using each hyperparameter combinations <br>
  `MSM_reweighted_population_analysis.py`,  evaluate the stationary distribution from the MSM and its deviation from the original distribution <br>
  `MSM_population_of_cluster.py`, evaluate the stationary distribution of each cluster <br>
  `bootstrapping_MSM.py`, estimate relative errors through bootstrapping, 80% of trajectories are randomly selected for each bootstrap sample <br>

- Construct and validate Markov State Models for each lasso peptide

### Thermodynamics_Kinetics

Contains scripts to perform thermodynamic and kinetic analysis of the 20 lasso peptides.

### Kinetic_pathways_analysis

Contains scripts to identify and cluster kinetic folding pathways.



# Authors:
Xuenan Mi,
xmi4@illinois.edu
