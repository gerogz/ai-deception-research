import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def apply_pca(data, n_components=2):
    """
    Apply Principal Component Analysis (PCA) to reduce dimensionality.

    Parameters:
        data (pd.DataFrame): Preprocessed numerical dataset.
        n_components (int): Number of principal components to keep.

    Returns:
        tuple: (Reduced data as pd.DataFrame, PCA model)
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)

    # Convert to DataFrame
    pca_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    return pca_df, pca

def find_optimal_clusters(data, max_clusters=10):
    """
    Use the Elbow Method and Silhouette Score to find the optimal number of clusters.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset.
        max_clusters (int): Maximum number of clusters to test.

    Returns:
        int: Optimal number of clusters.
    """
    distortions = []
    silhouette_scores = []
    K = range(2, max_clusters + 1)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    # Plot elbow method
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, distortions, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    # Plot silhouette scores
    plt.subplot(1, 2, 2)
    plt.plot(K, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')

    plt.show()

    optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    return optimal_k

def perform_clustering(data, n_clusters):
    """
    Perform K-Means clustering on the dataset.

    Parameters:
        data (pd.DataFrame): Preprocessed dataset.
        n_clusters (int): Number of clusters.

    Returns:
        tuple: (Cluster labels, K-Means model)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)

    return cluster_labels, kmeans
