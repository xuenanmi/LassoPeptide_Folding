import numpy as np
import os
import glob
import pyemma
from tqdm import trange

## Function comes from JCTC paper "An Efficient Path Classification Algorithm Based on Variational Autoencoder to Identify 
## Metastable Path Channels for Complex Conformational Changes", 2023, 19, 14, 4728–4742,  J. Chem. Theory Comput.
def microstates_distribution(reassign_trajs, num_clusters, idx1, idx2, x_initial, x_end,  y_initial, y_end, 
                             num_slices, output_dir="microstates_distribution_tica"):
    """
    Calculate the distribution of microstates in a discretized tICA space.

    Parameters:
    reassign_trajs: list of reassigned tICA conformations for each microstate.
    num_clusters: number of microstates.
    idx1: index of the first embedding coordinate.
    idx2: index of the second embedding coordinate.
    x_initial: minimum value of the first embedding coordinate.
    x_end: maximum value of the first embedding coordinate.
    y_initial: minimum value of the second embedding coordinate.
    y_end: maximum value of the second embedding coordinate.
    num_slices: number of bins to discretize along each coordinate.
    output_dir: output directory for the microstate distribution.
    """
    os.makedirs(output_dir)
    xdelta = (x_end - x_initial) / num_slices
    ydelta = (y_end - y_initial) / num_slices
    for i in range(num_clusters):
        dist = np.zeros((num_slices, num_slices))
        print("For No {} state, in total, there are {} frames;".format(i, len(reassign_trajs[i])))
        for j in range(0, len(reassign_trajs[i]), 1):
            x = (reassign_trajs[i][j, idx1]-x_initial) // xdelta
            y = (reassign_trajs[i][j, idx2]-y_initial) // ydelta
            if 0 < x < num_slices and 0 < y< num_slices:
                dist[num_slices-int(y)-1, int(x)] += 1
        dist = dist / (len(reassign_trajs[i]))
        np.save(output_dir + "/%03d_state_distribution.npy"%(i), dist)

        print("No {} state is completed for distribution calculation;".format(i))

# List of lasso peptides
lasso_names = ['microcinJ25']

# Corresponding parameters for each peptide
lag_time = [300]
cluster_number = [400]
tic_dim = [10]

for m, peptide in enumerate(lasso_names):
    # Load features of each trajectory
    totdist = [] 
    for file in sorted(glob.glob(f"{peptide}/*folded/*dist.npy")):
        distI = np.load(file)
        totdist.append(distI)
    print(len(totdist))
    tic = pyemma.coordinates.tica(totdist,lag= lag_time[m], dim=tic_dim[m])
    data_tic = tic.get_output()
    
    data_tic_connected = np.concatenate(data_tic)
    
    ## Import the microstates based clustered trajectories
    ## The trajectories should have list format
    ctrajs = np.load(f"{peptide}/MSM/MSM-cluster_kmeans_C_{cluster_number[m]}_lt_{lag_time[m]}_ticdim_{tic_dim[m]}.pkl", allow_pickle= True)
    
    
    ## Set the reassign step, number of microstates, dimensionality of tICA
    reassign_trajs = {}
    for i in trange(len(ctrajs)):
        for j in range(0, len(ctrajs[i])):
            try:
                reassign_trajs[int(ctrajs[i][j])] = np.vstack((reassign_trajs[int(ctrajs[i][j])], data_tic[i][j]))
            except KeyError:
                reassign_trajs[int(ctrajs[i][j])] = data_tic[i][j]
    np.save(f"{peptide}/reassign_tica2micro.npy", reassign_trajs)
    
    # Project all frames in each microstate on tIC0-tIC1 subspace
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=0, idx2=1, x_initial=np.floor(np.min(data_tic_connected[:,0])), x_end=np.ceil(np.max(data_tic_connected[:,0])), y_initial=np.floor(np.min(data_tic_connected[:,1])), y_end=np.ceil(np.max(data_tic_connected[:,1])), num_slices=50, output_dir= f"{peptide}/microstates_distribution_tica01")
    # Project all frames in each microstate on tIC0-tIC2 subspace
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=0, idx2=2, x_initial=np.floor(np.min(data_tic_connected[:,0])), x_end=np.ceil(np.max(data_tic_connected[:,0])), y_initial=np.floor(np.min(data_tic_connected[:,2])), y_end=np.ceil(np.max(data_tic_connected[:,2])), num_slices=50, output_dir= f"{peptide}/microstates_distribution_tica02")
    # Project all frames in each microstate on tIC1-tIC2 subspace
    microstates_distribution(reassign_trajs=reassign_trajs, num_clusters= cluster_number[m], idx1=1, idx2=2, x_initial=np.floor(np.min(data_tic_connected[:,1])), x_end=np.ceil(np.max(data_tic_connected[:,1])), y_initial=np.floor(np.min(data_tic_connected[:,2])), y_end=np.ceil(np.max(data_tic_connected[:,2])), num_slices=50, output_dir= f"{peptide}/microstates_distribution_tica12")
    
    
    ## Load the pathways identified by Transition Path Theory, each pathway is a sequence of state indexes
    paths = np.load(f"{peptide}/microcinJ25_TPT_5000_pathways.npy", allow_pickle=True)
    ## Input number of pathways with largest flux to embed
    num_pathways = 5001
    ## The directories of state distribution (in each pair of collective variables space)
    dirc1 = 'microstates_distribution_tica01'
    dirc2 = 'microstates_distribution_tica02'
    dirc3 = 'microstates_distribution_tica12'
    num_slices = 50
    os.makedirs(f"{peptide}/tpt_path_distribution")

    for i in range(0, num_pathways):
        f = paths[i]
        dist = np.zeros((3*num_slices, num_slices))
        temp = 0
        for j in range(len(f)):
            mat1 = np.load(f"{peptide}/"+dirc1+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
            mat2 = np.load(f"{peptide}/"+dirc2+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
            mat3 = np.load(f"{peptide}/"+dirc3+"/%03d_state_distribution.npy"%int(f[j]), allow_pickle=True)
    
            dist[:num_slices] = dist[:num_slices] + mat1 
            dist[num_slices:2*num_slices] = dist[num_slices:2*num_slices] + mat2 
            dist[2*num_slices:3*num_slices] = dist[2*num_slices:3*num_slices] + mat3 
    
        print("No {} transition pathway is calculated as distribution;".format(i))
        np.save(f"{peptide}/tpt_path_distribution/No_%05d_path_distribution.npy"%i, dist)    
        
