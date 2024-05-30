import glob
import mdtraj as md
import matplotlib.pyplot as plt
import os
import pickle
import pyemma
import numpy as np

# List of lasso peptides
lasso_names = [
    'acinetodin', 'astexin-1', 'benenodin-1', 'brevunsin', 'capistruin', 'caulonodin-V', 
    'caulosegnin-I', 'caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25', 
    'rubrivinodin', 'sphaericin', 'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 
    'xanthomonin-I', 'xanthomonin-II'
]

# Corresponding parameters for each peptide
lag_times = [250, 300, 300, 300, 250, 300, 250, 300, 300, 300, 300, 300, 350, 300, 300, 350, 300, 300, 250, 250]
cluster_numbers = [400, 100, 100, 400, 100, 200, 500, 100, 400, 200, 400, 400, 300, 400, 700, 200, 100, 200, 300, 400]
tic_dims = [4, 8, 8, 6, 10, 8, 8, 10, 8, 8, 10, 10, 10, 8, 6, 10, 6, 6, 10, 10]

# Loop over each lasso peptide
for m in range(len(lasso_names)):
    peptide = lasso_names[m]
    cluster_num = cluster_numbers[m]
    all_eigen_ref = []

    # Loop over each bootstrap sample
    for j in range(200):
        msm_file = f"{peptide}/bootstrapping/bt_80_{j}_msm.pkl"
        msm = pickle.load(open(msm_file, 'rb'))
        eigen_ref = msm.eigenvectors_left()[0]

        # Handle missing states
        if len(eigen_ref) == cluster_num:
            all_eigen_ref.append(eigen_ref)
        else:
            missing_index = [i for i in range(msm.active_set[0], msm.active_set[-1] + 1) if i not in msm.active_set]
            for missing_idx in missing_index:
                eigen_ref = np.insert(eigen_ref, missing_idx, 0)
            all_eigen_ref.append(eigen_ref)

    # Convert to numpy array and calculate mean and standard deviation
    all_eigen_ref = np.vstack(all_eigen_ref)
    all_mean_array = np.mean(all_eigen_ref, axis=0)
    all_sd_array = np.std(all_eigen_ref, axis=0)
    indices = list(range(cluster_num))

    # Plotting the results
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    axs.errorbar(indices, all_mean_array, yerr=all_sd_array, fmt='o', markersize=5)
    axs.set_xlim(0, cluster_num)
    axs.set_ylim(0, 0.2)
    axs.set_xlabel('State index', fontsize=30)
    axs.set_ylabel('MSM Population', fontsize=30)
    axs.set_xticks(np.arange(0, cluster_num + 1, step=cluster_num // 5))
    axs.set_xticklabels(np.arange(0, cluster_num + 1, step=cluster_num // 5), fontsize=22)
    axs.set_yticks(np.linspace(0, 0.2, 6))
    axs.set_yticklabels(np.linspace(0, 0.2, 6), fontsize=22)

    plt.savefig(f"{peptide}/MSM_population_with_bt.png", transparent=True, dpi=500)
    plt.close()

    print(f"Processed {peptide}")

