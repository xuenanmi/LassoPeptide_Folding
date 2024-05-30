import numpy as np
import pyemma
import pickle
import glob
import os
import pandas as pd

lasso_names = ['acinetodin', 'astexin-1', 'benenodin-1', 'brevunsin', 'capistruin', 'caulonodin-V', 'caulosegnin-I', 
               'caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25', 'rubrivinodin', 'sphaericin', 
               'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 'xanthomonin-I', 'xanthomonin-II'] 

tic_lag_times = [250, 300, 300, 300, 250, 300, 250, 300, 300, 300, 300, 300, 350, 300, 300, 350, 300, 300, 250, 250]
msm_lag_times = [500, 800, 1250, 400, 600, 600, 550, 800, 750, 600, 1000, 300, 800, 400, 650, 600, 400, 550, 700, 500]

# Grid search hyperparameters of Markov State Model (MSM)
tic_dims = [2, 4, 6, 8, 10]
clusters = [100, 200, 300, 400, 500, 700]

# Function to build MSM and evaluate VAMP2 score
def calculate_msm(peptide, index):
    tica_lag_time = tic_lag_times[index]
    msm_lag_time = msm_lag_times[index]
    VAMP2_scores = []
    totdist = []

    # Load all distance files for the peptide
    for file in sorted(glob.glob(peptide + '/*folded/*dist.npy')):
        #print(file)
        dist = np.load(file)
        totdist.append(dist)
    print(f'Total distance files for {peptide}: {len(totdist)}')

    # Iterate over different cluster numbers and TIC dimensions
    for cluster_number in clusters:
        for tic_dim in tic_dims:
            # Perform TICA (Time-lagged Independent Component Analysis)
            tica = pyemma.coordinates.tica(totdist, lag=tica_lag_time, dim=tic_dim)
            data_tica = tica.get_output()

            # Perform clustering using k-means
            cluster_kmeans = pyemma.coordinates.cluster_kmeans(data_tica, k=cluster_number, max_iter=200, stride=5)
            dtrajs = cluster_kmeans.dtrajs

            # Save clustering results
            dtrajs_file = f'{peptide}/MSM/MSM-cluster_kmeans_C_{cluster_number}_lt_{tica_lag_time}_ticdim_{tic_dim}.pkl'
            os.makedirs(os.path.dirname(dtrajs_file), exist_ok=True)
            with open(dtrajs_file, 'wb') as f:
                pickle.dump(cluster_kmeans.dtrajs, f)

            dtrajs_com = np.concatenate(dtrajs)

            # Estimate Markov State Model (MSM)
            msm = pyemma.msm.estimate_markov_model(dtrajs, lag=msm_lag_time)
            
            # Save MSM model
            msm_file = f'{peptide}/MSM/MSM_MSM_C_{cluster_number}_lt_{tica_lag_time}_ticdim_{tic_dim}.pkl'
            with open(msm_file, 'wb') as f:
                pickle.dump(msm, f)

            # Calculate population shift and VAMP2 score
            eigen_vector = msm.eigenvectors_left()[0]
            total_pop = np.sum([np.count_nonzero(dtrajs_com == idx) for idx in msm.active_set])

            popu_shift = [eigen_vector[j] / (np.count_nonzero(dtrajs_com == idx) / total_pop) 
                          for j, idx in enumerate(msm.active_set)]

            score = msm.score_cv(dtrajs, score_method="VAMP2", score_k=6)
            score_file = f'{peptide}/MSM/MSM-score_C_{cluster_number}_lt_{tica_lag_time}_ticdim_{tic_dim}.pkl'
            with open(score_file, 'wb') as f:
                pickle.dump(score, f)

            VAMP2_scores.append([np.mean(score), np.max(popu_shift), tic_dim, cluster_number, peptide])

            # Clean up to avoid memory issues
            del msm, dtrajs, cluster_kmeans, tica, data_tica

    # Sort VAMP2 scores and save to CSV
    sorted_VAMP2_scores = sorted(VAMP2_scores, key=lambda x: x[0], reverse=True)
    scores_df = pd.DataFrame(sorted_VAMP2_scores, columns=['VAMP2_score', 'popu_shift', 'tic_dim', 'cluster_number', 'peptide'])
    scores_df.to_csv(peptide + '/MSM_score.csv', index=False)

    print(f'Best VAMP2 score for {peptide}: {sorted_VAMP2_scores[0]}')
    del totdist

# Iterate over each peptide and calculate MSM
for index, peptide in enumerate(lasso_names):
    calculate_msm(peptide, index)

