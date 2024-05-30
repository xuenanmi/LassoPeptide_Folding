import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
import pickle
import pyemma

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

# Loop over each peptide
for i in range(len(lasso_names)):
    # Load the MSM model
    msm_path = f"{lasso_names[i]}/MSM/MSM-MSM_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl"
    msm = pickle.load(open(msm_path, 'rb'))

    # Load the native contacts per cluster
    Native_contacts_per_cluster = np.load(f"{lasso_names[i]}/Native_contacts_per_cluster.npy")
    
    # Define unfolded and folded states based on native contacts
    unfolded_state = list(np.where(Native_contacts_per_cluster < 0.1)[0])
    num = int(cluster_numbers[i] * 0.05)  
    folded_state = list(np.argpartition(Native_contacts_per_cluster, -num)[-num:])
    
    # Print the number of unfolded and folded states
    print(f"{lasso_names[i]}: {len(unfolded_state)} unfolded states, {len(folded_state)} folded states")
    
    # Perform Transition Path Theory (TPT) analysis
    TPT = pyemma.msm.tpt(msm, unfolded_state, folded_state)
    print('TPT analysis done')
    
    # Calculate pathways and their flux
    pathways = TPT.pathways(fraction=1.0, maxiter=10000)
    flux = pathways[1]
    
    # Calculate the relative flux and cumulative relative flux
    flux_res = [100 * f / TPT.total_flux for f in flux]
    cumulative_flux_res = np.cumsum(flux_res)
    len_flux = len(cumulative_flux_res)
    
    # Plot the cumulative flux
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    axs.scatter(range(1, len_flux + 1), cumulative_flux_res, s=15, color='dodgerblue')
    axs.plot(range(1, len_flux + 1), cumulative_flux_res, color='dodgerblue')
    axs.set_yticks([0, 20, 40, 60, 80, 100])
    axs.tick_params(axis='y', labelsize=22)
    axs.set_xlabel('Number of pathways', fontsize=24)
    axs.set_ylabel('Relative Flux (percent)', fontsize=24)
    plt.savefig(f"{lasso_names[i]}/Flux_plot.png", dpi=300, transparent=True)
    plt.close()
    
    # Print the total flux percentage of the pathways
    print(f"Total flux percentage for {lasso_names[i]}: {np.sum(pathways[1]) / TPT.total_flux * 100:.2f}%")
    
    # Calculate and save the committor probabilities
    committor = TPT.committor
    np.save(f"{lasso_names[i]}/TPT_pathways.npy", pathways[0])
    np.save(f"{lasso_names[i]}/TPT_pathways_flux.npy", pathways[1])
    np.save(f"{lasso_names[i]}/TPT_committor.npy", committor)

