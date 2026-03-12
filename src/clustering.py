import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import (
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_score,
)


# -------------------------------------------------------------------
# Distance functions for the custom K-Means implementation
# -------------------------------------------------------------------

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def cosine_distance(point1, point2):
    return cosine(point1, point2)


def manhattan_distance(point1, point2):
    return np.sum(np.abs(point1 - point2))


def gower_distance(point1, point2, categorical_column_index=0):
    """
    Simplified Gower-like distance for mixed data.
    In this project, Age is treated separately by position.
    """
    dissimilarity = 0
    p = len(point1)

    for i in range(p):
        if i == categorical_column_index:
            dissimilarity += (point1[i] != point2[i])
        else:
            dissimilarity += abs(point1[i] - point2[i])

    return dissimilarity / p


# -------------------------------------------------------------------
# K selection
# -------------------------------------------------------------------

def compute_elbow_sse(data, k_range, random_state=42):
    """
    Compute SSE values for the elbow method.
    """
    sse = []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(data)
        sse.append(model.inertia_)

    return list(k_range), sse


def compute_cluster_metrics(data, k_range, sample_fraction=0.05, random_state=42):
    """
    Compute Davies-Bouldin, Calinski-Harabasz, and Silhouette scores
    across different values of k.
    """
    db_scores = []
    ch_scores = []
    silhouette_scores = []

    # Random sample for silhouette to reduce runtime
    np.random.seed(random_state)
    sample_size = int(len(data) * sample_fraction)
    sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
    data_sample = data[sample_indices]

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            init="k-means++",
            max_iter=300,
            n_init=10
        )

        labels_full = model.fit_predict(data)
        db_scores.append(davies_bouldin_score(data, labels_full))
        ch_scores.append(calinski_harabasz_score(data, labels_full))

        labels_sample = KMeans(
            n_clusters=k,
            random_state=random_state,
            init="k-means++",
            max_iter=300,
            n_init=10
        ).fit_predict(data_sample)

        silhouette_scores.append(silhouette_score(data_sample, labels_sample))

    return pd.DataFrame({
        "k": list(k_range),
        "davies_bouldin_score": db_scores,
        "calinski_harabasz_score": ch_scores,
        "silhouette_score": silhouette_scores
    })


def plot_elbow(k_values, sse, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sse, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_cluster_metrics(metrics_df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(metrics_df["k"], metrics_df["davies_bouldin_score"], marker="o")
    axes[0].set_title("Davies-Bouldin Score")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Score")
    axes[0].grid(True)

    axes[1].plot(metrics_df["k"], metrics_df["calinski_harabasz_score"], marker="o")
    axes[1].set_title("Calinski-Harabasz Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Score")
    axes[1].grid(True)

    axes[2].plot(metrics_df["k"], metrics_df["silhouette_score"], marker="o")
    axes[2].set_title("Silhouette Score")
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Score")
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


# -------------------------------------------------------------------
# Custom K-Means
# -------------------------------------------------------------------

def custom_kmeans(data, k, distance_function, max_iter=50, tol=1e-4, random_state=42):
    """
    A simple K-Means implementation to compare behavior with scikit-learn.
    """
    np.random.seed(random_state)
    centroids = data[np.random.choice(data.shape[0], k, replace=False)].astype(float)

    start_time = time.time()

    def assign_clusters(data, centroids):
        labels = []
        for point in data:
            distances = [distance_function(point, centroid) for centroid in centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    def update_centroids(data, labels, k):
        new_centroids = []
        for i in range(k):
            cluster_points = data[labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(cluster_points.mean(axis=0).astype(float))
            else:
                new_centroids.append(data[np.random.choice(data.shape[0])].astype(float))
        return np.array(new_centroids)

    for iteration in range(max_iter):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)

        if np.allclose(centroids, new_centroids, atol=tol):
            elapsed_time = time.time() - start_time
            print(f"Custom K-Means converged in {iteration + 1} iterations.")
            print(f"Execution time: {elapsed_time:.4f} seconds")
            return labels, new_centroids

        centroids = new_centroids

    elapsed_time = time.time() - start_time
    print("Custom K-Means did not converge within the maximum number of iterations.")
    print(f"Execution time: {elapsed_time:.4f} seconds")

    return labels, centroids


# -------------------------------------------------------------------
# Scikit-learn K-Means
# -------------------------------------------------------------------

def run_kmeans(data, k, random_state=42, tol=1e-4):
    """
    Fit K-Means with scikit-learn and return labels and centroids.
    """
    start_time = time.time()

    model = KMeans(
        n_clusters=k,
        init="random",
        max_iter=300,
        tol=tol,
        random_state=random_state,
        n_init=10
    )
    model.fit(data)

    execution_time = time.time() - start_time

    print(f"Scikit-learn K-Means converged in {model.n_iter_} iterations.")
    print(f"Execution time: {execution_time:.4f} seconds")

    return model.labels_, model.cluster_centers_, model


def summarize_clusters(df_prepared, labels):
    """
    Add cluster labels to the dataset and return:
    - cluster counts
    - mean feature values by cluster
    """
    df_result = df_prepared.copy()
    df_result["Cluster"] = labels

    cluster_counts = df_result["Cluster"].value_counts().sort_index()
    cluster_profile = df_result.groupby("Cluster")[
        ["Age", "Marital_Status", "Gender", "City_Category_A", "City_Category_B", "City_Category_C"]
    ].mean()

    return df_result, cluster_counts, cluster_profile


def compute_final_metrics(data, labels, sample_fraction=0.05, random_state=42):
    """
    Compute clustering metrics for the final clustering solution.
    """
    db_score = davies_bouldin_score(data, labels)
    ch_score = calinski_harabasz_score(data, labels)

    np.random.seed(random_state)
    sample_size = int(len(data) * sample_fraction)
    sample_indices = np.random.choice(len(data), size=sample_size, replace=False)
    sampled_data = data[sample_indices]

    sampled_labels = KMeans(
        n_clusters=len(np.unique(labels)),
        random_state=random_state,
        n_init=10
    ).fit_predict(sampled_data)

    sil_score = silhouette_score(sampled_data, sampled_labels)

    metrics = {
        "davies_bouldin_score": db_score,
        "calinski_harabasz_score": ch_score,
        "silhouette_score": sil_score
    }

    return metrics

