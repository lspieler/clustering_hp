
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
import numpy as np
import skfuzzy as fuzz
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform
from distances import fastdtw

def hierarchical(num_clusters, dist_matrix):
    dist_matrix = squareform(dist_matrix)
    linkage_matrix = linkage(dist_matrix, method='average')
    labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return labels

    # cluster based on distance metrics dbscane
def dbscan(dist_matrix):
    clustering = DBSCAN(eps=0.5, min_samples=2, metric= 'precomputed').fit(dist_matrix)
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


def get_test_assignment(test_data, cluster_assignments, existing_data, distance_metric='euclidean'):
    # calcualte the medoids for each of the existing clusters
    medoids = []
    print(cluster_assignments)
    for i in range(len(np.unique(cluster_assignments))):
        cluster_data = existing_data[cluster_assignments -1 == i]
        medoid = np.mean(cluster_data, axis=0)
        medoids.append(medoid)
    
    print(medoids)
    # assign each test data to the closest medoid
    test_assignments = []
    for i in range(len(test_data)):
        if distance_metric == 'euclidean':
            test_assignments.append(np.argmin(np.linalg.norm(medoids - test_data[i], axis=1)))
        elif distance_metric == 'fastdtw':
            distances = []
            for medoid in medoids:
                distances.append(fastdtw(medoid, test_data[i])[0])
            test_assignments.append(np.argmin(distances))
        elif distance_metric == 'correlation':
            test_assignments.append(np.argmin(np.correlate(medoids, test_data[i])))
        elif distance_metric == 'magnitude':
            test_assignments.append(np.argmin(np.linalg.norm(np.abs(medoids - test_data[i]), axis=1)))
    return test_assignments

