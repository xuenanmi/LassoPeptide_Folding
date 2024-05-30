import numpy as np
import glob
import matplotlib.pyplot as plt
import mdtraj as md
import os
import pickle
import pyemma
import random


# Lasso peptide names
lasso_names = ['acinetodin', 'astexin-1', 'benenodin-1', 'brevunsin', 'capistruin', 'caulonodin-V', 'caulosegnin-I',
               'caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25', 'rubrivinodin', 'sphaericin',
               'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 'xanthomonin-I', 'xanthomonin-II']

# Corresponding parameters for each peptide
lag_times = [250, 300, 300, 300, 250, 300, 250, 300, 300, 300, 300, 300, 350, 300, 300, 350, 300, 300, 250, 250]
cluster_numbers = [400, 100, 100, 400, 100, 200, 500, 100, 400, 200, 400, 400, 300, 400, 700, 200, 100, 200, 300, 400]
tic_dims = [4, 8, 8, 6, 10, 8, 8, 10, 8, 8, 10, 10, 10, 8, 6, 10, 6, 6, 10, 10]

# Loop over each peptide to process MSM and plot
for i in range(len(lasso_names)):
    peptide = lasso_names[i]

    # Construct file paths
    cluster_file = f"{peptide}/MSM/MSM-cluster_kmeans_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl"
    msm_file = f"{peptide}/MSM/MSM-MSM_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl"

    # Load dtrajs and MSM model
    dtrajs_ref = pickle.load(open(cluster_file, 'rb'))
    dtrajs_com = np.concatenate(dtrajs_ref)
    msm_ref = pickle.load(open(msm_file, 'rb'))

    # Get the first left eigenvector of the MSM
    eigen_ref = msm_ref.eigenvectors_left()[0]

    # Calculate total population
    total_pop = np.sum([np.count_nonzero(dtrajs_com == index) for index in msm_ref.active_set])

    # Calculate unweighted and weighted populations
    unweighted_popu = np.array([np.count_nonzero(dtrajs_com == index) / total_pop for index in msm_ref.active_set])
    weighted_popu = eigen_ref

    # Plot the populations
    fig, axs = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
    axs.plot(np.log10(unweighted_popu), np.log10(weighted_popu), 'o', color='green')
    axs.plot([-5, 0], [-5, 0], color='black', linestyle='--')
    axs.set_xlim(-5, 0)
    axs.set_ylim(-5, 0)
    axs.set_xticks(range(-5, 1, 1))
    axs.set_yticks(range(-5, 1, 1))
    axs.set_xlabel('Unweighted Population', fontsize=30)
    axs.set_ylabel('MSM Population', fontsize=30)
    axs.tick_params(axis='both', which='major', labelsize=22)
    plt.savefig(f"{peptide}/raw_msm.png", transparent=True, dpi=500)
    plt.close()

    print(f"Processed {peptide}")

