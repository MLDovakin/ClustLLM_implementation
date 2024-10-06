import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import entropy
from sklearn.cluster import AgglomerativeClustering


def clustering(df, embeddings, n_clusters=70):

    wards = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit(embeddings)
    label = ward.labels_
    centroids = wards.centroids_

    return clusters, centroids

def calculate_distances(embeddings, centroids):
    closest_clusters, distances = pairwise_distances_argmin_min(embeddings, centroids)
    return closest_clusters, distances

def get_entropy(embeddings, centroids):

    cluster_probs = np.exp(-np.linalg.norm(embeddings[:, None] - centroids, axis=2))
    cluster_probs /= cluster_probs.sum(axis=1, keepdims=True) 

    entropies = np.apply_along_axis(entropy, 1, cluster_probs)
    return entropies

def get_triplets(embeddings, clusters, centroids, entropies, n_triplets=5):
    triplets = []
    high_entropy_indices = np.argsort(entropies)[-n_triplets:]  

    # Выбираем объекты с наибольшей энтропией чтобы потом подать их на вход LLM
    for idx in high_entropy_indices:
        anchor = embeddings[idx]
        anchor_cluster = clusters[idx]

        candidate_clusters = [i for i in range(len(centroids)) if i != anchor_cluster]
        choices = np.random.choice(candidate_clusters, size=2, replace=False)

        c1 = centroids[choices[0]]
        c2 = centroids[choices[1]]

        triplets.append((anchor, c1, c2))
    return triplets

clusters, centroids = clustering(df, embeddings, n_clusters=70)
df['Сluster'] = clusters

closest_clusters, distances = calculate_distances(embeddings, centroids)

entropies = get_entropy(embeddings, centroids)

triplets = get_triplets(embeddings, clusters, centroids, entropies)

df['entropy'] = entropies

df.head(), triplets


