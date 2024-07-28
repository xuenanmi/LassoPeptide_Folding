# Test Example
This folder contains code for analyzing kinetic pathwas of lasso peptide microcinJ25

-Step1:
 - `project_pathway_on_tICA_space.py`: Project kinetic pathways on low-dimensional space and obtain 1D vector of each pathway

-Step2:
 - `train_VAE_cluster_pathways.py`: Train VAE using 1D vector of each pathway, and obtain the latent space of all pathways

-Step3:
 - `optimize_cluster_number_by_silhouette_analysis.py`: optimize the cluster numbers of KMeans

-Step4:
 - `plot_pathways_on_latent_space.py`: Perform Kmeans of latent space of all pathways using the optimized cluster numbers
