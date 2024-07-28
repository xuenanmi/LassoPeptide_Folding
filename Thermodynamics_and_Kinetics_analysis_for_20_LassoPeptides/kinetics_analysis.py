import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import pickle
import seaborn as sns
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
cluster_mfpt = [10, 7, 5, 20, 5, 5, 10, 5, 10, 5, 10, 10, 15, 10, 20, 5, 5, 10, 7, 10]

mfpt_avg, mfpt_std = [], []
df_all = pd.DataFrame()

for i in range(len(lasso_names)):
    # Load the MSM model and native contacts per cluster for the current peptide
    msm = pickle.load(open(f"{lasso_names[i]}/MSM/MSM_C_{cluster_numbers[i]}_lt_{lag_times[i]}_ticdim_{tic_dims[i]}.pkl", 'rb'))
    NativeContacts = np.load(f"{lasso_names[i]}/Native_contacts_per_cluster.npy")
    
    # Determine unfolded and folded states based on native contacts
    unfolded_state = list(np.argpartition(NativeContacts, cluster_mfpt[i])[:cluster_mfpt[i]])
    folded_state = list(np.argpartition(NativeContacts, -cluster_mfpt[i])[-cluster_mfpt[i]:])
    
    mfpt_ls = []
    
    # Perform bootstrapping for 100 iterations
    for j in range(100):
        msm = pickle.load(open(f"{lasso_names[i]}/bootstrapping/bt_80_{j}_msm.pkl", 'rb'))
        TPT = pyemma.msm.tpt(msm, unfolded_state, folded_state)
        mfpt_ls.append(TPT.mfpt / 10000)  # Convert to microseconds

    # Save the bootstrapped MFPT results
    pickle.dump(mfpt_ls, open(f"{lasso_names[i]}/bootstrapping_mfpt_ls_new_test.pkl", 'wb'))
    
    # Compute and store the mean and standard deviation of the MFPT
    mfpt_avg.append(np.mean(mfpt_ls))
    mfpt_std.append(np.std(mfpt_ls))
    
    # Create a DataFrame for the current peptide and append to the overall DataFrame
    df = pd.DataFrame({'mfpt': mfpt_ls, 'lasso_name': lasso_names[i]})
    df_all = pd.concat([df_all, df], ignore_index=True)

print(df_all)

# Plotting the results
f, axs = plt.subplots(figsize=(10, 7))
sns.barplot(y=df_all['lasso_name'], x=df_all['mfpt'], capsize=.3, errwidth=1, edgecolor='black')
axs.spines['bottom'].set_linewidth(2)
axs.spines['left'].set_linewidth(2)
axs.spines['top'].set_linewidth(0)
axs.spines['right'].set_linewidth(0)
plt.xlim(0, 700)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel("")
plt.xlabel(r"Transition time of Unfolded$\rightarrow$Folded ($\mu$s)", fontsize=20)
plt.tight_layout()
plt.savefig("lasso_mfpt_bt_wide_slide.png")
plt.close()

