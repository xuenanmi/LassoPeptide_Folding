import numpy as np
import mdtraj as md
import glob
import os
import matplotlib.pyplot as plt
import pickle
import pyemma
import seaborn as sns

# Define color schemes
purple = ['#6247aa', '#815ac0', '#a06cd5', '#b185db', '#d2b7e5']
blue = ['#2c7da0', '#468faf', '#61a5c2', '#89c2d9', '#a9d6e5']
green = ['#718355', '#87986a', '#97a97c', '#a3b18a', '#cfe1b9']
orange = ["#ffb700", "#ffc300", "#ffd000", "#ffdd00", "#ffea00"]
red = ['#f25c54', '#f27059', '#f4845f', '#f79d65', '#f7b267']
larger = ['#f7b267'] * 5
colors = purple + blue + green + orange + red + larger

def plot_tica(data, weights, NC_list, cluster_num, cluster_tic1, cluster_tic2, idx, title_, figname):
    """
    Plot TICA (Time-lagged Independent Component Analysis) with native contacts.

    Parameters
    ----------
    data : np.ndarray
        TICA transformed data.
    weights : np.ndarray
        Weights of the data points.
    NC_list : list
        List of native contacts per cluster.
    cluster_num : int
        Number of clusters.
    cluster_tic1 : list
        TICA component 1 for clusters.
    cluster_tic2 : list
        TICA component 2 for clusters.
    idx : int
        Index of the cluster with maximum native contacts.
    title_ : str
        Title of the plot.
    figname : str
        Filename to save the plot.

    Returns
    -------
    None
    """
    x_data = data[:, 0]
    y_data = data[:, 1]
    x_hist_lim_low, y_hist_lim_low = np.min(x_data) - 0.5, np.min(y_data) - 0.5
    x_hist_lim_high, y_hist_lim_high = np.max(x_data) + 0.5, np.max(y_data) + 0.5

    x_lim_low, y_lim_low = int(np.min(x_data)) - 1, int(np.min(y_data)) - 1
    x_lim_high, y_lim_high = int(np.max(x_data)) + 1, int(np.max(y_data)) + 1

    # Compute 2D histogram
    hist = np.histogram2d(x_data, y_data, bins=[x_bins, y_bins],
                          range=[[x_hist_lim_low, x_hist_lim_high], [y_hist_lim_low, y_hist_lim_high]],
                          density=True, weights=weights)

    x_bin_size, y_bin_size = xedge[1] - xedge[0], yedge[1] - yedge[0]
    free_energy = -R * T * np.log(prob_density * x_bin_size * y_bin_size)
    delta_free_energy = free_energy - np.min(free_energy)

    xx = [(xedge[i] + xedge[i + 1]) / 2 for i in range(len(xedge) - 1)]
    yy = [(yedge[i] + yedge[i + 1]) / 2 for i in range(len(yedge) - 1)]
    # MSM weighted free energy landscape  
    fig, axs = plt.subplots(1, 1, figsize=(10, 7))
    cd = axs.contourf(xx, yy, delta_free_energy.T, np.linspace(0, Max_energy, Max_energy * 5 + 1), vmin=0.0, vmax=Max_energy, colors=colors)
    cbar = fig.colorbar(cd, ticks=range(Max_energy + 1))
    cbar.ax.set_yticklabels(range(Max_energy + 1), fontsize=20)
    cbar.set_label("Free Energy (kcal/mol)", size=20)
    
    sc = axs.scatter(cluster_tic1, cluster_tic2, s=50, c=NC_list, vmin=0, vmax=1, cmap='Greys')
    # Highlight the frame which has maximum native contacts (most close to folded structure)
    axs.plot(cluster_tic1[idx], cluster_tic2[idx], color='#EF233C', markeredgecolor='black', marker='^', markersize=20)
    
    axs.set_xticks(np.arange(int(x_lim_low), int(x_lim_high) + 1, 1))
    axs.set_xticklabels(np.arange(int(x_lim_low), int(x_lim_high) + 1, 1), fontsize=20)
    axs.set_yticks(np.arange(int(y_lim_low), int(y_lim_high) + 1, 1))
    axs.set_yticklabels(np.arange(int(y_lim_low), int(y_lim_high) + 1, 1), fontsize=20)

    for axis in ['top', 'bottom', 'left', 'right']:
        axs.spines[axis].set_linewidth(2)
    axs.tick_params(width=2)
    
    plt.title(title_, fontsize=30)
    plt.xlabel(x_key, fontsize=24)
    plt.ylabel(y_key, fontsize=24)
    plt.tight_layout()
    plt.savefig(figname, dpi=500, transparent=True)
    plt.close()

def process_lasso_peptide(lasso_name, lag_time, cluster_number, tic_dim, x_bins, y_bins, R, T, Max_energy, x_key, y_key, colors):
    """
    Process a single lasso peptide for TICA and native contacts plotting.

    Parameters
    ----------
    lasso_name : str
        Name of the lasso peptide.
    lag_time : int
        Lag time for TICA.
    cluster_number : int
        Number of clusters for MSM.
    tic_dim : int
        Number of TICA dimensions.
    x_bins : int
        Number of bins in x-direction for histogram.
    y_bins : int
        Number of bins in y-direction for histogram.
    R : float
        Gas constant.
    T : float
        Temperature.
    Max_energy : int
        Maximum free energy for plotting.
    x_key : str
        Label for x-axis.
    y_key : str
        Label for y-axis.
    colors : list
        List of colors for plotting.

    Returns
    -------
    None
    """
    print(lasso_name)
    NC = np.load(lasso_name + '/Native_contacts_per_cluster.npy')
    NC_list = list(np.around(NC, 2))
    idx = np.argmax(NC_list)
    
    totdist = []
    for file in sorted(glob.glob(lasso_name + '/*folded/*dist.npy')):
        distI = np.load(file)
        totdist.append(distI)
    print(len(totdist))
    
    tic = pyemma.coordinates.tica(totdist, lag=lag_time, dim=tic_dim)
    data_tic = tic.get_output()
    data_tic_connected = np.concatenate(data_tic)
    
    folded = np.load(lasso_name + '/folded_starting_point.npy')
    folded_arr = folded.reshape(1, len(folded))
    tic_folded = tic.transform(folded_arr)
    # Load MSM 
    msm = pickle.load(open(lasso_name + '/MSM/MSM-MSM_C_' + str(cluster_number) + '_lt_' + str(lag_time) + '_ticdim_' + str(tic_dim) + '.pkl', 'rb'))
    # Get MSM weights of each frame
    weights = np.concatenate(msm.trajectory_weights())
    connected_sets = msm.discrete_trajectories_active
    # Create a dictionary to save the index of trajectory and frame belong to the cluster
    dic = {new_list: [] for new_list in range(cluster_number)}
    for l in range(cluster_number):
        for j in range(len(connected_sets)):
            for k in range(len(connected_sets[j])):
                if connected_sets[j][k] == l:
                    dic[l].append([j, k])
    # Calculate the average tIC1 and tIC2 value of each cluster
    cluster_tic1, cluster_tic2 = [], []
    for k in range(cluster_number):
        tic1, tic2 = [], []
        for j in dic[k]:
            tic1.append(data_tic[j[0]][j[1], 0])
            tic2.append(data_tic[j[0]][j[1], 1])
        cluster_tic1.append(np.mean(tic1))
        cluster_tic2.append(np.mean(tic2))

    plot_tica(data_tic_connected, weights, NC_list, cluster_number, cluster_tic1, cluster_tic2, idx, '', lasso_name + '/' + lasso_name + '-MSM-TICA-plot-with-NC-red-triangle-of-cluster.png')

# Main execution
if __name__ == '__main__':
    x_bins = 100
    y_bins = 100
    R = 0.001987  # Gas constant in kcal/(mol*K)
    T = 300  # Temperature in Kelvin
    Max_energy = 6
    x_key = 'tIC 1'
    y_key = 'tIC 2'

    lasso_names = [
       'acinetodin', 'astexin-1', 'benenodin-1', 'brevunsin', 'capistruin', 'caulonodin-V', 
       'caulosegnin-I', 'caulosegnin-II', 'chaxapeptin', 'citrocin', 'klebsidin', 'microcinJ25', 
       'rubrivinodin', 'sphaericin', 'sphingopyxin-I', 'streptomonomicin', 'subterisin', 'ubonodin', 
       'xanthomonin-I', 'xanthomonin-II'
    ]

    lag_times = [250, 300, 300, 300, 250, 300, 250, 300, 300, 300,300, 300, 350, 300, 300, 350, 300, 300, 250, 250]
    cluster_numbers = [400, 100, 100, 400, 100, 200, 500, 100, 400, 200, 400, 400, 300, 400, 700, 200, 100, 200, 300, 400]
    tic_dims = [4, 8, 8, 6, 10, 8, 8, 10, 8, 8, 10, 10, 10, 8, 6,10, 6, 6, 10, 10]

    # Process each lasso peptide in parallel
    for i in range(len(lasso_names)):
        process_lasso_peptide(lasso_names[i], lag_times[i], cluster_numbers[i], tic_dims[i], x_bins, y_bins, R, T, Max_energy, x_key, y_key, colors)


