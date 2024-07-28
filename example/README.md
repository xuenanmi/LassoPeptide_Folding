# Test Example: Kinetic Pathway Analysis of Lasso Peptide MicrocinJ25
This repository contains code for analyzing the kinetic pathways of the lasso peptide MicrocinJ25. The analysis is performed in several steps as outlined below.

-Step1:
 - `project_pathway_on_tICA_space.py`: Project kinetic pathways on low-dimensional space and obtain 1D vector for each pathway.

-Step2:
 - `train_VAE_cluster_pathways.py`: Train VAE using 1D vector of the pathways to obtain the latent space representation of all pathways.
   
-Step3:
 - `optimize_cluster_number_by_silhouette_analysis.py`: Optimize the number of clusters for KMeans clustering using silhouette analysis to ensure the best fit for the data.
   
-Step4:
 - `plot_pathways_on_latent_space.py`: Perform KMeans clustering on the latent space of all pathways using the optimized number of clusters and visualize the results.
