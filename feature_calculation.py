import mdtraj as md
import numpy as np
import glob

# generate the features of 20 lasso peptides
lasso_names = ['acinetodin', 'astexin-1', 'benenodin-1', 'brevunsin', 'capistruin', 'caulonodin-V', 'caulosegnin-I',
               'caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25', 'rubrivinodin', 'sphaericin',
               'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 'xanthomonin-I', 'xanthomonin-II']

# Function to calculate residue-residue distances for each trajectory
def calculate_distances(file_path, top_path):
    trajectories = glob.glob(file_path)
    for traj_file in trajectories:
        # Load trajectory and corresponding topology
        t = md.load(traj_file, top=top_path)
        # Compute residue-residue distances using the alpha-carbon atoms
        dist = md.compute_contacts(t, contacts='all', scheme='ca')
        # Extract distances
        distances = dist[0]
        # Generate file names
        filename = traj_file.replace('.xtc', '').replace('.dcd', '').replace('/folded', '').replace('/unfolded', '')
        # Save distances to numpy file
        np.save(filename + '_ca_dist.npy', distances)

# Calculate residue-residue distances for folded trajectories
for peptide in lasso_names:
    folded_traj_path = peptide + '/folded/*.xtc'
    topology_path = '/home/xuenan/storage/Research/Lasso/FAH_Lasso/Lasso-20-structure-nocap-new/equl/' + peptide + '-HMR-strip.prmtop'
    calculate_distances(folded_traj_path, topology_path)
    print(peptide)

# Calculate residue-residue distances for unfolded trajectories
for peptide in lasso_names:
    unfolded_traj_path = peptide + '/unfolded/*.xtc'
    topology_path = '/home/xuenan/storage/Research/Lasso/FAH_Lasso/Lasso-20-structure-nocap-new/equl/' + peptide + '-HMR-strip.prmtop'
    calculate_distances(unfolded_traj_path, topology_path)
    print(peptide)

