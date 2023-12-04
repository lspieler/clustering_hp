from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from concurrent.futures import ProcessPoolExecutor
from get_data import get_data, get_simple_data
from multiprocessing import pool
import glob
from ffnn import feed_foward_nn
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import concurrent.futures
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from multiprocessing import freeze_support
from fft import fft_distance
import cProfile
import ot
from scipy.stats import gaussian_kde
from scipy import signal
import getopt, sys
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from lstm import run_lstm, cluster_lstm
import numpy as np
import ot  # Optimal Transport library
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist



def find_medoid(cluster):
    min_distance = float('inf')
    medoid = None
    
    for i, s1 in enumerate(cluster):
        current_distance = sum(fastdtw([s1], [s2], dist=euclidean)[0] for j, s2 in enumerate(cluster) if i != j)
        
        if current_distance < min_distance:
            min_distance = current_distance
            medoid = s1

    return medoid

def assign_to_cluster(new_series, cluster_medoids, data_portion):
        min_distance = float('inf')
        best_cluster = None

        for x in range(len(cluster_medoids)):
            distance, _ = fastdtw([new_series[:data_portion]], [cluster_medoids[x][:data_portion]], dist=euclidean)
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


def compute_wasserstein_distances(series):
    series = np.array([np.sort(s) for s in series])
    n = series.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Compute the Wasserstein distance between series i and j
            dist = ot.wasserstein_1d(series[i], series[j])
            distances[i, j] = dist
            distances[j, i] = dist  # The distance matrix is symmetric

    return distances

def wasserstein_kmeans(series, n_clusters):
    distances = compute_wasserstein_distances(series)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(distances)
    return labels

def compute_distances_to_centroids(args):
    d, centroids = args
    return [wasserstein_distance(d, centroid) for centroid in centroids]

def wmd_assign_to_cluster(new_series, cluster_representatives):
    """
    Assign a new series to an existing cluster based on the smallest Wasserstein distance to the cluster representatives.
    """
    new_series_dist = np.sort(new_series)  # Convert the new series into a distribution
    distances = [ot.wasserstein_1d(new_series_dist, rep) for rep in cluster_representatives]

    return np.argmin(distances)

def wmd_get_centroids(cluster_series):
    """
    Compute a representative distribution for a cluster.
    This could be a simple mean of the distributions, or a more complex representation.
    """
    # This is a placeholder; the actual implementation depends on your choice of representation
    return np.mean(np.array(cluster_series), axis=0)



def fdtw_clustering(series):

    n = len(series)
    distance_matrix = np.zeros((n, n))
    fourier_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            #perfrom time warping
            distance, _ = fastdtw([series[i]], [series[j]], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            #fourier distance
           
    return distance_matrix

def fft_clustering(series, distance_metric = 'euclidean'):

    n = len(series)
    fourier_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):

            fourier_distance_matrix[i, j] = fft_distance(series[i], series[j], detrend = True, dc_component = True, phase = True, distance_metric = distance_metric)   
            fourier_distance_matrix[j, i] = fft_distance(series[j], series[i], detrend = True, dc_component = True, phase = True, distance_metric= distance_metric)
              
    return fourier_distance_matrix

def compute_kde(series):
    """
    Compute the KDE for a given series.
    """
    kde = gaussian_kde(series)
    distribution = kde(np.linspace(min(series), max(series), len(series)))
    
    # Normalize the distribution so it sums to 1
    distribution /= distribution.sum()
    return distribution




def cluster(freq_per_second, num_clusters, poriton, learner, layers= 100):
    num_files = 10

    msgs = sorted(glob.glob('/Users/lspieler/Semesters/Fall 23/Honors Project/test-clustering/data/AAPL/AAPL_*.csv'))[:num_files]

    # get data using get_simple data
    result = np.empty((num_files,23400))
    x = 0
    for x in range(len(msgs)):
        df = get_simple_data(0, 10000000, msgs[x], freq_per_second)
        df = df.iloc[0:23400]
        df["price"] = (df["ask_1"] + df["bid_1"]).bfill().ffill()/2
        result[x] = (df["price"]/df['price'].iloc[0] -1 ) * 100
        print(result[x])
    
    data_length = result.shape[1]

    test_price = result[-1]
    result = result[:-1]

    plt.plot(test_price)
    
    num_files = len(msgs)
    data_portion = int(data_length * poriton)
    #exclude last day from msgs and orders
 
    price_series = np.empty((num_files,data_length))
    volume_series = np.empty((num_files, data_length))
    vp_series = np.empty((num_files,data_length))

    """
    args = [(0, 10000000, freq_per_second, "", order, msg) for order, msg in zip(orders, msgs)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_data, *zip(*args))) 
    
    executor.shutdown(wait=True)
    """

    
    #print(vp_series[-1].shape)
    
    price_distance_matrix = fdtw_clustering(price_series)
    volume_distance_matrix = fdtw_clustering(volume_series)
    #vp_distance_matrix = fdtw_clustering(vp_series)

    fourier_price_distance_matrix = fft_clustering(price_series)
    condensed_D1 = squareform(fourier_price_distance_matrix)
    """
    Z = linkage(condensed_D1, 'ward')
    dendrogram(Z)
    plt.title('Hierarchical Clustering of Time Series')
    plt.xlabel('Time Series')
    plt.ylabel('Distance')
    #output dendrogram to png file
    plt.savefig('fourier_price_dendrogram.png')
    input("stop")
    """
    condensed_D1 = squareform(price_distance_matrix)
    condensed_D2 = squareform(volume_distance_matrix)
    #condensed_D3 = squareform(vp_distance_matrix)
    # Perform hierarchical clustering
    
    Z = linkage(condensed_D1, 'ward')
    """
    dendrogram(Z)
    plt.title('Hierarchical Clustering of Time Series')
    plt.xlabel('Time Series')
    plt.ylabel('Distance')
    #output dendrogram to png file
    plt.savefig('price_dendrogram.png')

    Z1 = linkage(condensed_D2, 'ward')
 


    dendrogram(Z1)
    plt.title('Hierarchical Clustering of Time Series')
    plt.xlabel('Time Series')
    plt.ylabel('Distance')
    #output dendrogram to png file
    plt.savefig('volume_dendrogram.png')
    """
   

    
    """
    #convert series to probability distributions using kde
    distributions = np.array([compute_kde(s) for s in series])
    # normalize the distributions for mass

    # Perform Wasserstein's k-means clustering
    labels, centroids = ƒ_kmeans(distributions, k=5)
    """

    # Assuming k clusters
    labels = fcluster(Z, t=num_clusters, criterion='maxclust')
    print(labels)

    # construct  of clusters from lables
    clusters = []
    for i in range(num_clusters):
        clusters.append([])

    for i in range(len(labels)):
            clusters[labels[i]-1].append(price_series[i])
    
    # Find medoid of each cluster
    medoids = np.empty((num_clusters,data_length))
    for x in range(len(clusters)):
        medoids[x] = find_medoid(clusters[x])
    
    # find cluster of test series
    test_cluster = assign_to_cluster(test_price, medoids)

    # halve all price series and save to new variable
    X = np.empty((len(msgs),data_portion))
    for x in range(len(price_series)):
        X[x] = price_series[x][0:data_portion]
    
    #get final price of each day as the y
    y = np.empty(len(msgs))
    for x in range(len(price_series)):
        y[x] = price_series[x][-1]

    if learner == "lstm":
        run_lstm(X, y, test_price, data_portion, layers, layers, 1, 100)
    elif learner == "ffnn":
        feed_foward_nn(X, y, test_price, data_portion, clusters, test_cluster, layers)
    elif learner == "cluster-lstm":
        cluster_lstm(clusters, test_cluster, test_price, data_portion, layers, 100, 100) 


   
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
    if len(sys.argv) != 6:
        print("Usage: python test-cluster.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    freq_per_second = str(sys.argv[1])
    num_clusters = int(sys.argv[2])
    poriton = float(sys.argv[3])
    learner = str(sys.argv[4])
    layers = int(sys.argv[5])
    
    
    freeze_support()  
    cluster(freq_per_second, num_clusters, poriton, learner, layers)
    