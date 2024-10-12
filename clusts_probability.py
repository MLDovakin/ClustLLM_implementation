cluster_labels = df['Кластер'].values

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

def select_triplets_by_entropy(embeddings, entropies, cluster_labels, y_min=20, y_max=80, top_n=20000):

    triplets = []

    entropy_thresh_min = np.percentile(entropies, y_min)
    entropy_thresh_max = np.percentile(entropies, y_max)

    entropy_thresh_ind = np.where((entropies >= entropy_thresh_min) & (entropies <= entropy_thresh_max))[0]

    high_entropy_indices = entropy_thresh_ind[np.argsort(entropies[entropy_thresh_ind])[-top_n:]]

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

y_min = 20
y_max = 80 
triplets = select_triplets_by_entropy(embeddings, entropies, cluster_labels, y_min, y_max)
