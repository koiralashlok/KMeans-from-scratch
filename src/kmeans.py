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

    def fit(self, X: np.ndarray):
        '''
        A clustering is a list of clusters
        Index in clustering corresponds to index in centroids
        '''
        self.initialize_centroids(X)
        iteration = 0
    
        def has_clustering_changed(old_clstr, new_clstr):
            if not old_clstr or old_clstr[0].size <= 0:
                return True
            
            for i in range(0, self.n_clusters):
                if not np.array_equal(old_clstr[i], new_clstr[i]):
                    return True
                
            return False

        clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        new_clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        while iteration < self.max_iter:
            # Calculate new centroids and clustering
            new_clustering, new_centroids = self.update_centroids(clustering, X)

            # If no change, return
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
        def is_cluster_empty(clstr):
            clstr[0].size > 0

        if is_cluster_empty(clustering):
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
        :param X:
        :return:
        """
        if self.init == 'random':
            self.centroids: np.ndarray = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            # TODO
            pass
        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)) # can actually drop sqrt


    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # TODO
        pass