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
    fourier_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            #perfrom time warping
            distance, _ = fastdtw([series[i]], [series[j]], dist=euclidean)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
            #fourier distance
           
    return distance_matrix

def fft_clustering(series):

    n = len(series)
    fourier_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):

            fourier_distance_matrix[i, j] = fft_distance(series[i], series[j], detrend = True, dc_component = True, phase = True, distance_metric = 'euclidean')
            fourier_distance_matrix[j, i] = fft_distance(series[j], series[i], detrend = True, dc_component = True, phase = True, distance_metric = 'euclidean')
              
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

    msgs = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_message_10.csv'))[:16]
    orders = sorted(glob.glob('./data/monthly/AAPL_2023-04-*_orderbook_10.csv'))[:16]

    #get data from last file 
    test = get_data(0, 10000000, freq_per_second=freq_per_second, directory = "", orderbook_filename = orders[-1], message_filename = msgs[-1])
    #replace all nan values with interpolated values
    test = test.bfill().ffill()
    test_price =((test['price']/test['price'].iloc[0])-1).ffill().bfill()*100
    msgs = msgs[:-1]
    orders = orders[:-1]
    data_length = len(test_price)
    num_files = len(msgs)
    data_portion = int(data_length * poriton)
    #exclude last day from msgs and orders
 
    price_series = np.empty((num_files,data_length))
    volume_series = np.empty((num_files, data_length))
    vp_series = np.empty((num_files,data_length))

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
    
    args = [(0, 10000000, freq_per_second, "", order, msg) for order, msg in zip(orders, msgs)]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(get_data, *zip(*args))) 
    
    executor.shutdown(wait=True)

    

    # get data using get_simple data
    result = np.empty((num_files,23400))
    x = 0
    for x in range(len(msgs)):
        df = get_simple_data(0, 10000000, msgs[x], freq_per_second)
        result[x] = df

    

    x= 0
    for result in results:
        price_series[x] = ((result['price']/result['price'].iloc[0])-1).ffill().bfill()*100
        volume_series[x] = ((result['b_size_0'] / result['a_size_0'])-1).ffill().bfill().fillna(0)*100
        #combine both volumen and price into one series such that a data
        #point is a tuple of (price, volume)
        #vp_series[x] = (price_series[x], volume_series[x])
        x += 1
    
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
    labels, centroids = wasserstein_kmeans(distributions, k=5)
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
    