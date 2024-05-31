import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Define the result directory
resultdir = 'results/train_nsamples15000_batchsize250_lr8e-05_c1'

# Load training and testing loss
train_score = np.load(os.path.join(resultdir, 'training_scores_save.npy'), allow_pickle=True)
test_score = np.load(os.path.join(resultdir, 'testing_scores_save.npy'), allow_pickle=True)
train_loss = [train_score.tolist()[i][0] for i in range(100)] 
test_loss = [test_score.tolist()[i][0] for i in range(100)]

# Plot training and testing loss
fig, ax = plt.subplots(dpi=300, figsize=(10, 7))
epochs = list(range(200))
ax.plot(epochs, train_loss, color='navy', linewidth=3, label='Training Loss')
ax.plot(epochs, test_loss, color='orange', linewidth=3, label='Testing Loss')

for spine in ax.spines.values():
    spine.set_linewidth(2.0)
plt.xlabel('Training Epochs', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.legend(loc="upper right", fontsize="x-large")
plt.savefig(os.path.join(resultdir, 'train_test.png'))
plt.close()



# Load latent space of each pathway and flux data
centers = np.load(os.path.join(resultdir, 'trained_hidden_vectors_nsamples15000_batchsize250_lr8e-05_c1.npy'))
flux = np.load('TPT_pathways_flux.npy')

# Define clustering parameters
n_clusters = 2    # optimized from silhouette_analysis
n_samples = 15000 # number of pathways

# Perform KMeans clustering
km_cluster = KMeans(n_clusters=n_clusters, n_init="auto", random_state=12)
km_cluster.fit(centers)

# Save clustering labels
np.savetxt(os.path.join(resultdir, f"paths_{n_clusters}_kmeans_clustering_labels.txt"), km_cluster.labels_)

# Calculate and print silhouette score
silhouette_avg = silhouette_score(centers, km_cluster.labels_)
print(f'Silhouette Score: {silhouette_avg}')

# Calculate and print cluster flux
km_lump = np.zeros(n_clusters)
for i in range(n_samples):
    km_lump[int(km_cluster.labels_[i])] += flux[i]
km_lump /= np.sum(km_lump)
print(f'Cluster Flux Distribution: {km_lump}')


# Plot KMeans clustering results
colors = [plt.cm.tab20c(1), plt.cm.Set1(7)]
fig, ax = plt.subplots(dpi=600, figsize=(7, 7))
for i in range(n_samples):
    ax.scatter(centers[i, 0], centers[i, 1], c=[colors[int(km_cluster.labels_[i])]])
for spine in ax.spines.values():
    spine.set_linewidth(2.0)
plt.xlabel('VAE Latent Dimension 1', fontsize=22)
plt.ylabel('VAE Latent Dimension 2', fontsize=22)
# plt.title(f'KMeans Clustering: nsamples {n_samples}, batchsize {batch_size}, lr {learning_rate}')
plt.savefig(os.path.join(resultdir, f"hidden_layer_{n_clusters}_kmeans_centers.png"), bbox_inches='tight')
