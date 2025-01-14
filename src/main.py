import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='data.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)

    # Hyperparameters
    init = 'kmeans++'
    max_iter = 300

    # Perform fit
    kmeans = KMeans(n_classifiers, init, max_iter)
    learnt_clustering, learnt_centroids, learnt_labels = kmeans.fit(X)

    print(f'Silhouette Score for Clustering: {kmeans.silhouette(learnt_clustering, X)}')

    # Visualize
    visualize_cluster(learnt_clustering, learnt_centroids)

def visualize_cluster(clustering: list[np.ndarray], centroids: np.ndarray):
    from sklearn.decomposition import PCA as skpca

    # Cannot plot more than 6 clusters, not enough colors in matplotlib
    if(len(clustering) > 6):
        return

    # Combine clusters into a single array for PCA
    all_points = np.vstack(clustering)

    # Perform PCA to reduce to 2D
    pca = skpca(n_components=2)
    reduced_points = pca.fit_transform(all_points)

    # Plot
    plt.figure(figsize=(10, 8))

    colors = ['b','g','r','c','m','y','k']
    for i, cluster in enumerate(clustering):
        cluster_reduced = pca.transform(cluster)  # Transform the cluster to 2D
        plt.scatter(cluster_reduced[:, 0], cluster_reduced[:, 1], color=colors[i], label=f'Cluster {i + 1}', alpha=0.5)

    # Plot centroids
    centroid_reduced = pca.transform(np.vstack(centroids))
    plt.scatter(centroid_reduced[:, 0], centroid_reduced[:, 1], color='k', marker='x', s=200, label='Centroids')

    # Title, labels, etc.
    plt.title('Cluster Visualization with Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
