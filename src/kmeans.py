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

    def is_clustering_empty(self, cluster):
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
            if self.is_clustering_empty(old_clstr):
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
            new_clustering, new_centroids, labels = self.update_centroids(clustering, X)

            # If no change, return current clustering and centroids
            if not has_clustering_changed(clustering, new_clustering):
                return (clustering, self.centroids, labels)
            
            # Update clustering and centroids
            self.centroids = new_centroids.copy()
            clustering = new_clustering.copy()
            iteration += 1

        return (clustering, self.centroids, labels)
    

    def update_centroids(self, clustering: list[np.ndarray], X: np.ndarray):
        
        # Initialize to self.centroids to handle first call
        new_centroids: np.ndarray = self.centroids.copy()
        labels = np.zeros(X.shape[0])

        # In the first call, clustering will be empty
        if not self.is_clustering_empty(clustering):
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
            labels[i] = cluster_index

        return (new_clustering, new_centroids, labels)


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X: ndarray containing dataset
        :return: (n_cluster, d) ndarry with initial centroids
        """
        if self.init == 'random':
            self.centroids: np.ndarray = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            # 1. Randomly select the first centroid from the data points
            self.centroids = np.zeros((self.n_clusters, X.shape[1]))
            self.centroids[0] = X[np.random.choice(X.shape[0])]

            for i in range(1, self.n_clusters):
                # 2. Compute distance from nearest, previously chosen centroid
                distances = self.euclidean_distance(X, self.centroids[:i])
                min_distances = np.min(distances, axis=1)  # distance from nearest centroid

                # Step 3: Select the next centroid proportional to distances
                probabilities = min_distances / min_distances.sum()
                cumulative_probabilities = np.cumsum(probabilities)
                random_value = np.random.rand()

                # Pick first value >= random_value
                next_centroid_index = np.searchsorted(cumulative_probabilities, random_value)
                self.centroids[i] = X[next_centroid_index]

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

        # Exploiut broadcasting
        X1_expanded = X1[:, np.newaxis, :]  # (num_rows_X1, 1, num_features)
        X2_expanded = X2[np.newaxis, :, :]  # (1, num_rows_X2, num_features)

        # Squared differences
        squared_diff = (X1_expanded - X2_expanded) ** 2  # (num_rows_X1, num_rows_X2, num_features)

        # Sum the squared differences along the last dimension and take the square root
        return np.sqrt(np.sum(squared_diff, axis=-1))  # (num_rows_X1, num_rows_X2)
    

    def silhouette(self, clustering: list, X: np.ndarray):
        score = 0
        num_pts = X.shape[0]

        is_cluster_empty = lambda cluster: len(cluster) == 0

        # For each point in X (same as each point in each cluster)
        for cluster_number, cluster in enumerate(clustering):
            
            if is_cluster_empty(cluster):
                continue

            dist_matrix = self.euclidean_distance(cluster, self.centroids)

            # Col vector with ith row = dist from ith point to its cluster rep
            a = dist_matrix[:,cluster_number]

            # Remove centroid of current cluster so that min is the second closest cluster
            dist_matrix = np.delete(dist_matrix, cluster_number, axis=1)
            b = np.min(dist_matrix, axis=1)

            # Add to the sum score
            score = score + np.sum((b-a)/b) # b is max after clustering!

        # Return the average score
        return score / num_pts
