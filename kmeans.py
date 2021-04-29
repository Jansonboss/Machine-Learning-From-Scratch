import numpy as np
import random
from scipy.linalg.special_matrices import toeplitz
from scipy.spatial.distance import euclidean
from scipy.spatial import distance_matrix

def kmeans(X:np.ndarray, k:int, centroids=None, max_iter=30, tolerance=1e-2):

    if centroids == None: # select k random unique points from X as k inital centroids => x: (n,p)
        init_centroid_mask = random.sample(range(X.shape[0]), k)
        current_centroid = X[init_centroid_mask] # (k, p)
        # https://jbencook.com/pairwise-distance-in-numpy/z
        # distance = X[:,None,:] - init_centroid[None,:,:] = (K, 1, p) - (1, n, p) = (K, n, p)
        distance = distance_matrix(X, current_centroid)
        cluster_id = np.argmin(distance, axis=1) # assign data point to their closest cluster centroid
        
        for i in range(max_iter):
            previous_centroid = current_centroid
            # recompute the centroid with group mean
            current_centroid = np.zeros(previous_centroid.shape)
            for j in range(k):
                subset = X[cluster_id==j]
                centorid_j = np.sum(subset, axis=0)/len(subset)
                current_centroid[j, :] = centorid_j
            # recompute the distance matrix on new centroid
            distance = distance_matrix(X, current_centroid)
            cluster_id = np.argmin(distance, axis=1)
            if np.linalg.norm(current_centroid - previous_centroid) <= tolerance:
                return current_centroid, cluster_id
        # otherwise exceed the maxiumn iteration
        return current_centroid, cluster_id

        
    if centroids == "kmeans++":
        current_centroid = np.zeros((k, X.shape[1]))
        init_centroid_mask = random.sample(range(X.shape[0]))
        current_centroid[0] = X[init_centroid_mask]
        distance = distance_matrix(X, current_centroid)

        for i in range(1, k):
            prob = distance ** 2
            rand_index = np.random.choice(X.shape[0], size = 1, p = prob / np.sum(prob))
            centroids[i] = X[rand_index]
            if i == k - 1:
                break
            distances_new = distance_matrix(X, [centroids[i]])
            cluster_id = np.argmin(distance, axis=1) 
            if np.linalg.norm(current_centroid - distances_new) <= tolerance:
                return current_centroid, cluster_id
        # otherwise exceed the maxiumn iteration
        return current_centroid, cluster_id


