
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
- Featurize MD trajectory data using all pair-wise residue-residue distances  
  - `feature_calculation.py`: Calculate all pair-wise residue-residue distances of all trajectory data.
- Construct and validate Markov State Models for each lasso peptide  
  - `optimize_MSM_hyperparameter.py`: Perform a grid search to optimize the hyperparameter of MSM based on VAMP2 score and construct MSM using each hyperparameter combination.  
  - `MSM_reweighted_population_analysis.py`: Evaluate the stationary distribution from the MSM and its deviation from the original distribution.  
  - `MSM_population_of_cluster.py`: Evaluate the stationary distribution of each cluster.  
  - `bootstrapping_MSM.py`: Estimate relative errors through bootstrapping. 80% of trajectories are randomly selected for each bootstrap sample, and MSM is constructed for each sample.

### Thermodynamics_Kinetics

Contains scripts to perform thermodynamic and kinetic analysis of the 20 lasso peptides.
- Thermodynamics analysis  
  - `NativeContacts_calculation.py`: Calculate the fraction of native contacts of each microstate (cluster).  
  - `thermodynamics_analysis.py`: Calculate the percentage of lasso-like topology formation (Native Contacts > 0.8).
- Kinetic analysis  
  - `kinetics_analysis.py`: Estimate the transition timescale from unfolded states to folded states by MFPT (mean first passage time).
- Plot MSM-weighted free energy landscape  
  - `Free_energy_landscape_plot.py`: Project MD trajectory data on tIC 1 and tIC2, and project the average native contact value of each microstate on the landscape.

### Kinetic_pathways_analysis

Contains scripts to identify and cluster kinetic folding pathways.
- `kinetic_pathway_from_TPT.py`: Obtain all kinetic pathways from unfolded to folded state based on TPT (Transition Path Theory).  
- `project_pathway_on_tICA_space.py`: Project each pathway on low dimensional subspace (tICA space).  
- `train_VAE_cluster_pathways.py`: Train VAE model to learn the latent space of each pathway.  
- `optimize_cluster_number_by_silhouette_analysis.py`: Optimize the cluster number used for performing k-means clustering on VAE latent space of all pathways.  
- `plot_pathways_on_latent_space.py`: Visualize the clustering results of all pathways on VAE latent space and plot train-validation loss of VAE model.

## Dependency
To set up the environment for this project, use the provided `environment.yml` file. This file contains all necessary dependencies.

## Authors

- **Xuenan Mi** - [xmi4@illinois.edu](mailto:xmi4@illinois.edu)

## Contributing

Contributions are welcome! Please create an issue to discuss the changes you wish to make.

## License


