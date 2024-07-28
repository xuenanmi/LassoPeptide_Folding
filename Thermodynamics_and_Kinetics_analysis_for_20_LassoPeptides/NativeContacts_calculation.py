import numpy as np
import mdtraj as md
from itertools import combinations
import matplotlib.pyplot as plt
import glob
import pickle
from multiprocessing import Pool

# Function to calculate the fraction of native contacts (from mdtraj)
def q_func(traj, native):
    """
    Compute the fraction of native contacts according the definition from
    Best, Hummer and Eaton [1]
    
    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    native : md.Trajectory
        The 'native state'. This can be an entire trajecory, or just a single frame.
        Only the first conformation is used
        
    Returns
    -------
    q : np.array, shape=(len(traj),)
        The fraction of native contacts in each frame of `traj`
        
    References
    ----------
    ..[1] Best, Hummer, and Eaton, "Native contacts determine protein folding
          mechanisms in atomistic simulations" PNAS (2013)
    """
    BETA_CONST = 50  # 1/nm
    LAMBDA_CONST = 1.8
    NATIVE_CUTOFF = 0.45  # nanometers
    
    # Get the indices of all of the heavy atoms
    heavy = native.topology.select_atom_indices('heavy')
    # Get the pairs of heavy atoms which are farther than 3 residues apart
    heavy_pairs = np.array(
        [(i, j) for (i, j) in combinations(heavy, 2)
            if abs(native.topology.atom(i).residue.index - 
                   native.topology.atom(j).residue.index) > 3]
    )
    
    # Compute the distances between these pairs in the native state
    heavy_pairs_distances = md.compute_distances(native[0], heavy_pairs)[0]
    # Get the pairs such that the distance is less than NATIVE_CUTOFF
    native_contacts = heavy_pairs[heavy_pairs_distances < NATIVE_CUTOFF]
    
    # Compute these distances for the whole trajectory
    r = md.compute_distances(traj, native_contacts)
    # Compute them for just the native state
    r0 = md.compute_distances(native[0], native_contacts)
    
    q = np.mean(1.0 / (1 + np.exp(BETA_CONST * (r - LAMBDA_CONST * r0))), axis=1)
    return q



def cal_Native_Contacts(i):
    """
    Calculate native contacts for each frame and native contacts of each cluster.
    Parameters:
    i (int): Index of the lasso peptide to process.

    Returns:
    None
    """
    Native_contacts_res = [] 
    native = md.load_pdb(f"{lasso_names[i]}/{lasso_names[i]}_folded_starting_point.pdb")
    
    # Load the file list from a pickled file
    files = pickle.load(open(f"{lasso_names[i]}/file_list.pkl", 'rb'))
    print(f"Processing {lasso_names[i]} with {len(files)} files")
    
    for file in sorted(files):
        traj = md.load(file, top=f'/home/xuenan/storage/Research/Lasso/FAH_Lasso/Lasso-20-structure-nocap-new/equl/{lasso_names[i]}-HMR-strip.prmtop')
        q = q_func(traj, native)
        Native_contacts_res.append(q)
    
    # Save the native contacts of each frame to a file
    np.save(f"{lasso_names[i]}/Native_contacts_res.npy", Native_contacts_res)

    # Load the MSM model
    msm = pickle.load(open(f"{lasso_names[i]}/MSM/MSM_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl", 'rb'))
    connected_sets = msm.discrete_trajectories_active
    print(f"Number of connected sets: {len(connected_sets)}")

    # Create a dictionary to store the native contacts per cluster
    dic = {new_list: [] for new_list in range(cluster_numbers[i])}
    for l in range(cluster_numbers[i]):
        for j in range(len(connected_sets)):
            for k in range(len(connected_sets[j])):
                if connected_sets[j][k] == l:
                    dic[l].append([j, k])
    
    # Calculate the mean native contact for each cluster
    cluster_Q = []
    for k in range(cluster_numbers[i]):
        tmp = [Native_contacts_res[j[0]][j[1]] for j in dic[k]]
        cluster_Q.append(np.mean(tmp))
    print(f"{lasso_names[i]}, {cluster_numbers[i]}, {len(cluster_Q)}")
    
    # Save the native contacts per cluster
    np.save(f"{lasso_names[i]}/Native_contacts_per_cluster.npy", cluster_Q)
    
    # Plot and save the native contacts per cluster
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    axs.scatter(range(cluster_numbers[i]), sorted(cluster_Q), s=20, color='black')
    plt.ylim(0, 1.0)
    plt.ylabel('Native Contacts', fontsize=24)
    plt.savefig(f"{lasso_names[i]}/Native_contacts_per_cluster_plot.png", dpi=300, transparent=True)
    plt.close()

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
# Create a multiprocessing pool and process the peptides in parallel
p = Pool(20)
p.map(cal_Native_Contacts, range(20))
p.close()
p.join()
                                                       
