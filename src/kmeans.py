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
    
        def has_clustering_changed(new_centroids_):
            for i in range(0, self.n_clusters):
                if not np.array_equal(new_centroids_[i], self.centroids[i]):
                    return False
            return True

        clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        new_clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        while iteration < self.max_iter:
            # Calculate new centroids and clustering
            new_clustering, new_centroids = self.update_centroids(clustering, X)
                
            # If no change, return
            if has_clustering_changed(new_clustering):
            # if new_centroids == self.centroids:
                return clustering, self.centroids
            
            # Update clustering and centroids
            self.centroids = new_centroids.copy()
            clustering = new_clustering.copy()
            iteration += 1

        return clustering, self.centroids

    def update_centroids(self, clustering: list[np.ndarray], X: np.ndarray):
        
        # Initialize to self.centroids to handle first call
        new_centroids = self.centroids.copy()

        # In the first call, clustering will be empty
        if clustering[0].size > 0:
            for i in range(0, self.n_clusters):
                new_centroids[i] = np.average(clustering[i]) # TODO does this broadcast properly

        dist_matrix = self.euclidean_distance(X, new_centroids)
        new_clustering = [np.empty((0, 100)) for _ in range(self.n_clusters)]
        for i, row in enumerate(dist_matrix):
            # pick min col
            cluster_index = np.argmin(row)
            # add row to cluster corresponding to min col
            # new_clustering[cluster_index] is (num_pts, dims) but X[i] is (dims,) so reshape to (1,dims)
            np.append(new_clustering[cluster_index], np.reshape(X[i], [1,X.shape[1]]), axis=0)
        
        # TODO maybe make a mask of cluster 1 being min, cluster 2 being min then select and append for faster?
        return (new_centroids, new_clustering)


    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            self.centroids = X[np.random.choice(X.shape[0], size=self.n_clusters, replace=False)]
            print('random centroids')
            print(self.centroids)
        elif self.init == 'kmeans++':
            # your code
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
        # your code
        return np.sqrt(np.sum((X1[:, np.newaxis] - X2) ** 2, axis=2)) # can actually drop sqrt

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        pass