
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans

def hierarchical(num_clusters, dist_matrix):
    clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average').fit(dist_matrix)
    cluster_assignments = clustering.labels_
    return cluster_assignments

    # cluster based on distance metrics dbscane
def dbscan(dist_matrix):
    clustering = DBSCAN(eps=0.5, min_samples=5, metric='precomputed').fit(dist_matrix)
    cluster_assignments = clustering.labels_
    return cluster_assignments

def spectral_clustering(num_clusters, dist_matrix):
    # perform spectral clustering
    clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(dist_matrix)
    cluster_assignments = clustering.labels_
    return cluster_assignments

def kmeans(num_clusters, price_series):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(price_series)
    cluster_assignments = kmeans.labels_
    return cluster_assignments


def fuzzy_cmeans(num_clusters, price_series):
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        price_series.T,  
        c=num_clusters,  # Number of clusters
        m=2,  # Fuzziness parameter
        error=0.005,  # Stopping criterion
        maxiter=10000,  # Maximum number of iterations
        init=None  # Initialization method (None for random)
    )
    return np.argmax(u, axis=0)