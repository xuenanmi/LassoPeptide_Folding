import numpy as np
import pandas as pd
import glob
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

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

res_avg = []
df_all = pd.DataFrame()

for i in range(20):
    # Load native contacts array for the current peptide
    NativeContacts_array = np.load(f"{lasso_names[i]}/Native_contacts_res.npy", allow_pickle=True)
    NativeContacts = NativeContacts_array.tolist()
    
    res = []
    
    # Perform bootstrapping for 100 iterations
    for j in range(100):
        # Load bootstrapped files and MSM model
        files = pickle.load(open(f"{lasso_names[i]}/bootstrapping/bt_80_{j}_files.pkl", 'rb'))
        msm = pickle.load(open(f"{lasso_names[i]}/bootstrapping/bt_80_{j}_msm.pkl", 'rb'))
        
        all_files = sorted(glob.glob(f"{lasso_names[i]}/*folded/*dist.npy"))
        index = [all_files.index(m) for m in files]
        
        # Select the corresponding native contacts
        NC_selected = [NativeContacts[k] for k in index]
        
        # Flatten the list of native contacts
        NC_list = np.concatenate(NC_selected)
        
        # Create a boolean list for native contacts greater than 0.75
        boolean_list = NC_list > 0.8
        feat = list(map(int, boolean_list))
        
        # Calculate the weighted fraction of lasso-like topology
        weights = np.concatenate(msm.trajectory_weights())
        feat_w = np.dot(feat, weights) * 100
        res.append(feat_w)
    
    # Save the bootstrapped results
    pickle.dump(res, open(f"{lasso_names[i]}/bootstrapping_NC_0.8_ls.pkl", 'wb'))
    
    # Load the saved results for creating the DataFrame
    NC = pickle.load(open(f"{lasso_names[i]}/bootstrapping_NC_0.8_ls.pkl", 'rb'))
    df = pd.DataFrame({'NC': NC, 'lasso_name': lasso_names[i]})
    df_all = pd.concat([df_all, df], ignore_index=True)

print(df_all)

# Plot the results
f, axs = plt.subplots(figsize=(10, 7))
axs = sns.barplot(y=df_all["lasso_name"], x=df_all["NC"], capsize=.3, errwidth=1, edgecolor='black')
axs.spines['bottom'].set_linewidth(2)
axs.spines['left'].set_linewidth(2)
axs.spines['top'].set_linewidth(0)
axs.spines['right'].set_linewidth(0)
plt.xlim(0, 0.8)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylabel("")
plt.xlabel("Percentage of lasso-like topology formation (%)", fontsize=20)
plt.savefig("Fraction_LassoFormation_bt_NC_0.8_slide.png", bbox_inches='tight')

