import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

cluster_labels = label
def calculate_entropy(embeddings, cluster_labels, K=4):

    cluster_centers = np.array([embeddings[cluster_labels == i].mean(axis=0) for i in np.unique(cluster_labels)])
    distances = cdist(embeddings, cluster_centers, metric='euclidean')

    sorted_indices = np.argsort(distances, axis=1)[:, :K]
    probabilities = np.exp(-distances) / np.sum(np.exp(-distances), axis=1, keepdims=True)

    entropies = []
    for i in range(len(embeddings)):
        closest_probs = probabilities[i, sorted_indices[i]]
        entropy = -np.sum(closest_probs * np.log(closest_probs + 1e-9))
        entropies.append(entropy)

    return np.array(entropies)

entropies = calculate_entropy(embeddings, cluster_labels)

def select_triplets_by_entropy(embeddings, entropies, cluster_labels, top_n=5000):

    triplets = []

    high_entropy_indices = np.argsort(entropies)[-top_n:]

    for idx in high_entropy_indices:
        anchor = embeddings[idx]

        same_cluster_indices = np.where(cluster_labels == cluster_labels[idx])[0]
        same_cluster_indices = same_cluster_indices[same_cluster_indices != idx]


        if len(same_cluster_indices) == 0:
            continue
        positive_idx = np.random.choice(same_cluster_indices)
        positive = embeddings[positive_idx]

        different_cluster_indices = np.where(cluster_labels != cluster_labels[idx])[0]
        negative_idx = np.random.choice(different_cluster_indices)
        negative = embeddings[negative_idx]

        triplets.append((anchor, positive, negative))

    return triplets

triplets = select_triplets_by_entropy(embeddings, entropies, cluster_labels)


