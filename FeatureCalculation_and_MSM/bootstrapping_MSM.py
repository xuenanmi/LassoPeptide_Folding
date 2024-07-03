import numpy as np
import glob
import mdtraj as md
import math
import os
import pickle
import pyemma
import random

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
msm_lag_times = [500, 800, 1250, 400, 600, 600, 550, 800, 750, 600, 1000, 300, 800, 400, 650, 600, 400, 550, 700, 500]

# Ensure the bootstrapping directory exists for each peptide
for peptide in lasso_names:
    os.makedirs(f"{peptide}/bootstrapping", exist_ok=True)

# Loop over each lasso peptide
for i, peptide in enumerate(lasso_names):
    totSD = []
    fileSD = []

    # Load the distance matrices
    for file in sorted(glob.glob(f"{peptide}/*folded/*dist.npy")):
        totSD.append(np.load(file))
        fileSD.append(file)

    # Load the MSM clusters
    msm_file = f"{peptide}/MSM/MSM-cluster_kmeans_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl"
    dtrajs = pickle.load(open(msm_file, 'rb'))

    N = int(len(fileSD) * 0.8)  # 80% of the data for bootstrapping

    # Perform bootstrapping 200 times
    for j in range(200):
        index = list(range(len(fileSD)))
        random.shuffle(index)
        bt_files = [fileSD[k] for k in index[:N]]
        bt_kmeans = [dtrajs[k] for k in index[:N]]

        # Estimate the Markov model
        msm = pyemma.msm.estimate_markov_model(bt_kmeans, lag=msm_lag_times[i])

        # Save the bootstrapped data
        with open(f"{peptide}/bootstrapping/bt_80_{j}_files.pkl", 'wb') as f:
            pickle.dump(bt_files, f)
        with open(f"{peptide}/bootstrapping/bt_80_{j}_clusters.pkl", 'wb') as f:
            pickle.dump(bt_kmeans, f)
        with open(f"{peptide}/bootstrapping/bt_80_{j}_msm.pkl", 'wb') as f:
            pickle.dump(msm, f)

    print(f"Processed {peptide}")
