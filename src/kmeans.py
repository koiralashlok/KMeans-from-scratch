import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def is_cluster_empty(self, cluster):
            return not cluster or cluster[0].size <= 0

    def fit(self, X: np.ndarray):
        '''
        A clustering is a list of clusters
        Index in clustering corresponds to index in centroids
        '''
        self.initialize_centroids(X)
        iteration = 0
    
        def has_clustering_changed(old_clstr, new_clstr):
            # Empty old cluster implies 0th iteration, return true
            if self.is_cluster_empty(old_clstr):
                return True
            
            for i in range(0, self.n_clusters):
                # Change detected in the ith cluster
                if not np.array_equal(old_clstr[i], new_clstr[i]):
                    return True
            
            # No change detected
            return False

        # Initialize clustering lists
        clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        new_clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]

        while iteration < self.max_iter:
            # Calculate new centroids and clustering
            new_clustering, new_centroids = self.update_centroids(clustering, X)

            # If no change, return current clustering and centroids
            if not has_clustering_changed(clustering, new_clustering):
                return (clustering, self.centroids)
            
            # Update clustering and centroids
            self.centroids = new_centroids.copy()
            clustering = new_clustering.copy()
            iteration += 1

        return (clustering, self.centroids)

    def update_centroids(self, clustering: list[np.ndarray], X: np.ndarray):
        
        # Initialize to self.centroids to handle first call
        new_centroids: np.ndarray = self.centroids.copy()

        # In the first call, clustering will be empty
        if not self.is_cluster_empty(clustering):
            for i in range(0, self.n_clusters):
                new_centroids[i] = np.average(clustering[i], axis=0) # Average over row axis
        
        # Calculate distance from each point to each centroid
        dist_matrix = self.euclidean_distance(X, new_centroids)

        new_clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        for i, row in enumerate(dist_matrix):
            # Pick colummn corresponding to minimum distance
            cluster_index = np.argmin(row)

            # Add row to cluster corresponding to lowest distance
            new_clustering[cluster_index] = np.append(new_clustering[cluster_index], np.reshape(X[i], [1,X.shape[1]]), axis=0)

        # TODO maybe make a mask of cluster 1 being min, cluster 2 being min then select and append for faster?
        return (new_clustering, new_centroids)


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X: ndarray containing dataset
        :return: (n_cluster, d) ndarry with initial centroids
        """
        if self.init == 'random':
            self.centroids: np.ndarray = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            # KMeans++ initialization
            self.centroids[0] = X[np.random.choice(X.shape[0])]  # Randomly choose the first centroid
            for i in range(1, self.n_clusters):
                distances = self.euclidean_distance(X, self.centroids[:i])  # Calculate distances to existing centroids
                min_distances = np.min(distances, axis=1)  # Get the minimum distance for each point
                probabilities = min_distances / np.sum(min_distances)  # Normalize to create a probability distribution
                self.centroids[i] = X[np.random.choice(X.shape[0], p=probabilities)]  # Select next centroid based on probability

        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1: ndarray of first dataset (usually the entire dataset)
        :param X2: ndarray of second dataset (usually centroids)
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)) # TODO can actually drop sqrt


    # def silhouette(self, clustering: np.ndarray, X: np.ndarray):
    #     # TODO
    #     pass

    def silhouette_score(self, clustering: list[np.ndarray], X: np.ndarray) -> float:
        """
        Compute the silhouette score for the clustering.
        
        :param clustering: A list of ndarrays, where each ndarray contains the points of a cluster.
        :param X: A 2D ndarray of shape (n_samples, n_features) representing the dataset.
        :return: The average silhouette score for the clustering.
        """
        n_clusters = self.n_clusters
        n_samples = X.shape[0]
        
        # Assign cluster labels to each sample
        labels = np.zeros(n_samples, dtype=int)
        for cluster_index, cluster in enumerate(clustering):
            labels[np.isin(X, cluster)] = cluster_index
        
        total_silhouette = 0.0
        
        for i in range(n_samples):
            # Get the current point and its cluster
            current_point = X[i]
            current_label = labels[i]
            
            # Calculate intra-cluster distance
            same_cluster_points = clustering[current_label]
            if len(same_cluster_points) > 1:
                intra_distances = self.euclidean_distance(current_point.reshape(1, -1), same_cluster_points)
                a_i = np.mean(intra_distances)
            else:
                a_i = 0  # No other points in the same cluster

            # Calculate nearest cluster distance
            nearest_dist = np.inf
            for cluster_index, cluster in enumerate(clustering):
                if cluster_index != current_label:
                    inter_distances = self.euclidean_distance(current_point.reshape(1, -1), cluster)
                    nearest_dist = min(nearest_dist, np.mean(inter_distances))
            
            # Silhouette score for the current point
            if a_i == 0 and nearest_dist == 0:
                s_i = 0  # If both distances are zero, the silhouette score is undefined
            else:
                s_i = (nearest_dist - a_i) / max(a_i, nearest_dist)

            total_silhouette += s_i

        # Return the average silhouette score
        return total_silhouette / n_samples