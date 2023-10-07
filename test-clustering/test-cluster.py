from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor
from get_data import get_data
import glob
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from multiprocessing import freeze_support
import cProfile
import ot
from scipy.stats import gaussian_kde
import getopt, sys


def find_medoid(cluster):
    min_distance = float('inf')
    medoid = None
    
    for i, s1 in enumerate(cluster):
        current_distance = sum(fastdtw([s1], [s2], dist=euclidean)[0] for j, s2 in enumerate(cluster) if i != j)
        
        if current_distance < min_distance:
            min_distance = current_distance
            medoid = s1
    return medoid

def assign_to_cluster(new_series, cluster_medoids):
        min_distance = float('inf')
        best_cluster = None

        for x in range(len(cluster_medoids)):
            distance, _ = fastdtw([new_series], [cluster_medoids[x]], dist=euclidean)
            if distance < min_distance:
                min_distance = distance
                best_cluster = x

        return best_cluster

def wasserstein_barycenter(cluster):
    """Compute the Wasserstein barycenter of a cluster of 1D distributions."""
    n_distributions = len(cluster)
    n_bins = len(cluster[0])
    
    # Uniform weights and the cost matrix for 1D data
    weights = np.ones(n_distributions) / n_distributions
    M = ot.dist(np.arange(n_bins).reshape(-1, 1), np.arange(n_bins).reshape(-1, 1)).astype(np.float64)
    M /= M.max()
    
    # Compute barycenter
    barycenter = ot.bregman.barycenter(cluster, M, reg=1e-3, weights=weights)
    return barycenter

def wasserstein_distance(P, Q):
    """Compute the 1D Wasserstein distance between two distributions P and Q."""
    print(P, Q)
    n_bins = len(P)
    M = ot.dist(np.arange(n_bins).reshape(-1, 1), np.arange(n_bins).reshape(-1, 1)).astype(np.float64)
    M /= M.max()
    return ot.emd2(P, Q, M)

def wasserstein_kmeans(distributions, k, max_iter=10):
    """Perform Wasserstein's k-means clustering for 1D distributions."""
    n_samples = len(distributions)
    n_bins = len(distributions[0])
    
    # Randomly initialize cluster centroids
    centroids = distributions[np.random.choice(n_samples, k, replace=False)]
    
    for iteration in range(max_iter):
        # Assign distributions to the closest centroid
        with ProcessPoolExecutor() as executor:
            distances = list(executor.map(compute_distances_to_centroids, [(d, centroids) for d in distributions]))
        labels = np.argmin(distances, axis=1)
        # Update the centroids to the barycenter of the clusters
        new_centroids = [wasserstein_barycenter([distributions[i] for i in range(n_samples) if labels[i] == j]) for j in range(k)]
        
        if np.allclose(new_centroids, centroids):
            break
        centroids = new_centroids

    return labels, centroids

def compute_distances_to_centroids(args):
    d, centroids = args
    return [wasserstein_distance(d, centroid) for centroid in centroids]

def fdtw_clustering(series):

    n = len(series)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            #perfrom time warping
            #check series for nan values
            series[i] = series[i].ffill().bfill()
            series[j] = series[j].ffill().bfill()

            distance, _ = fastdtw([series[i]], [series[j]], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

def compute_kde(series):
    """
    Compute the KDE for a given series.
    """
    kde = gaussian_kde(series)
    distribution = kde(np.linspace(min(series), max(series), len(series)))
    
    # Normalize the distribution so it sums to 1
    distribution /= distribution.sum()
    return distribution



def main():

    msgs = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_message_10.csv'))
    orders = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_orderbook_10.csv'))
    #restrict msgs and orders to 2 files
    msgs = msgs[0:5]
    orders = orders[0:5]
    price_series = np.empty((5,23400))
    volume_series = np.empty((5,23400))
    vp_series = np.empty((5,23400))

    profiler = cProfile.Profile()

    """
    #Convert all orderbook and message files to all_series format
    for x in range(10):
        orderbook_file = orders[x]
        msg_file = msgs[x]
        df = get_data(0, 10000000, freq_per_second=1, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
        #replace all nan values with interpolated values
        df = df.interpolate()
        series[x] = df['price']
    """
    

    """
    df = get_data(0, 10000000, freq_per_second=1, directory = "", orderbook_filename = orderbook_file, message_filename = msg_file)
    #replace all nan values with interpolated values
    test_series = df['price'].ffill().bfill()"""
        
    
    args = [(0, 10000000, 1, "", order, msg) for order, msg in zip(orders, msgs)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_data, *zip(*args))) 

    x = 0
    for result in results:
         if x<10:
            price_series[x] = result['price'].ffill().bfill()
            #check series for nan values
          
            #fill volume series with bid ask ratio for all 10 levels
            volume_series[x] = (result['b_size_0'] / result['a_size_0']).ffill().bfill()
            #combine both volumen and price into one series such that a data
            #point is a tuple of (price, volume)
            #vp_series[x] = (price_series[x], volume_series[x])
            x +=1

    #print(vp_series[-1].shape)
    
    price_distance_matrix = fdtw_clustering(price_series)
    volume_distance_matrix = fdtw_clustering(volume_series)
    #vp_distance_matrix = fdtw_clustering(vp_series)
    

    condensed_D1 = squareform(price_distance_matrix)
    condensed_D2 = squareform(volume_distance_matrix)
    #condensed_D3 = squareform(vp_distance_matrix)
    # Perform hierarchical clustering
    Z = linkage(condensed_D1, 'ward')
    dendrogram(Z)
    plt.title('Hierarchical Clustering of Time Series')
    plt.xlabel('Time Series')
    plt.ylabel('Distance')
    

    Z1 = linkage(condensed_D2, 'ward')
    dendrogram(Z)
    plt.title('Hierarchical Clustering of Time Series')
    plt.xlabel('Time Series')
    plt.ylabel('Distance')
    

    
    """
    #convert series to probability distributions using kde
    distributions = np.array([compute_kde(s) for s in series])
    # normalize the distributions for mass

    # Perform Wasserstein's k-means clustering
    labels, centroids = wasserstein_kmeans(distributions, k=5)
    """

    # Assuming k clusters
    labels = fcluster(Z, t=3, criterion='maxclust')
    print(labels)
    # construct  of clusters from lables
    clusters = []
    for i in range(3):
        clusters.append([])

    for i in range(len(labels)):
            clusters[labels[i]-1].append(price_series[i])
    
    # Find medoid of each cluster
    medoids = np.empty((3,23400))
    for x in range(len(clusters)):
        medoids[x] = find_medoid(clusters[x])
    print(medoids)
    #Find what cluster the test_series belongs to
   

    #assign cluster to test_series
    #assigned_cluster = assign_to_cluster(test_series, medoids)

    #print(assigned_cluster)



    
    """
    gmm = GaussianMixture(n_components=5) 
    gmm.fit(price_series)

    # Get cluster assignments
    labels = gmm.predict(price_series)
    print("Cluster labels:", labels)
    probs = gmm.predict_proba(price_series)
    print(probs)
    """


if __name__ == '__main__':
    #take command line arguments for start, end and freq_per_second
  
    freeze_support()  
    main()
    