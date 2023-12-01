import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import multiprocessing as mp
from get_data import get_simple_data
from test_cluster import fdtw_clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from test_cluster import find_medoid
import sys
from ffnn import feed_foward_nn
from test_cluster import assign_to_cluster, fft_clustering
from lstm import run_lstm, cluster_lstm, episodic_lstm
import time
import time
from lstm_ns import run_lstm as run_lstm_ns
import logging


def init_worker():
    # This will be called by each pool process when it starts.
    logger = mp.get_logger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s/%(processName)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

def update_monitor(shared_updates, lock):

    print("Starting update monitor")
    with open('updates.txt', 'a') as file:
        while True:
            while shared_updates:  # While there are updates in the list
                update = shared_updates.pop(0)  # Remove and get the first update
                with lock:
                     file.write(f'{update}\n')
            time.sleep(5) 

def lstm_cluster(num_clusters, data_portion, num_files, layers = 50, batch = 100, epoch = 150, distancemetric = 'euclidean', model = 'lstm'):
    ## set up logger
    logger = mp.get_logger()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt='[%(process)d] %(message)s',
        datefmt='%y-%m-%d %H:%M:%S')

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    data_length = 23400
    data_portion = int(data_length * data_portion)
    
  
    msgs = sorted(glob.glob('/Users/lspieler/Semesters/Fall 23/Honors Project/test-clustering/data/AAPL/AAPL_*.csv'))[:16]
    # add msft 
    
    print(len(msgs))
    mp.set_start_method('spawn')

    args = [(f, msgs, num_files, num_clusters, data_portion, layers, batch, epoch, distancemetric, model) for f in range(len(msgs) - num_files)]
    mp.log_to_stderr(logging.DEBUG)

    # Use the initializer to set up logging in each pool process
    with mp.Pool(processes=1, initializer=init_worker) as pool:
        # The main process logging
        logger = mp.get_logger()
        logger.debug('Number of tasks: %2d', len(args))

        # Run the tasks
        results = pool.starmap(process_files, args)
    
    print(results)
     # If you need to process results# check if none
    all_normal_results = []
    all_cluster_results = []
    for result in results:
        if result is not None:
            if result[0] is not None and result[1] is not None:
                all_normal_results.append(result[0])
                all_cluster_results.append(result[1])
            else:
                continue

    # write output to file
    with open(f"lstm-cluster-{num_clusters}-{data_portion}-{num_files}-{layers}-{batch}-{epoch}.txt", "w") as f:
        f.write(f"normal results: {all_normal_results}")
        f.write(f"cluster results: {all_cluster_results}")


def process_files(f, msgs, num_files, num_clusters, data_portion, layers, batch, epoch, distancemetric = 'euclidean', model = 'lstm'):
        normal_results = []
        cluster_results = []
        nn_results = []
        nn_cluster = []
        print(f)
        i = num_files + f
        files = msgs[f: i]
        print(len(files))

        result = np.zeros((num_files,23400))
        plotting = np.zeros((num_files,23400))

        x = 0
        for x in range(len(files)):
            df = get_simple_data(0, 10000000, files[x], 's')
            df = df.iloc[0:23400]
            if df.shape[0] != 23400:
                continue
            df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2
            plotting[x] = df['price'] /100000
            result[x] = ((df["price"]/df['price'].iloc[0]) -1) * 10
 

        test_price = result[-1]
        result = result[:-1]


        if distancemetric == 'euclidean':
            distance_matrix = fdtw_clustering(result)
        elif distancemetric == 'fft_euclidean':
            print('euclidean_fft')
            distance_matrix = fft_clustering(result, distance_metric = 'euclidean')
        elif distancemetric == 'fft_phase':
            print('phase')
            distance_matrix = fft_clustering(result, distance_metric = 'phase')
        elif distancemetric == 'fft_correlation':  
            print('correlation')
            distance_matrix = fft_clustering(result, distance_metric = 'correlation')
        elif distancemetric == 'wmd':
            print('not implemented')
        else:
            distance_matrix = fdtw_clustering(result)

        price_series = result

        condensed_D1 = squareform(distance_matrix)

        Z = linkage(condensed_D1, 'ward')

        labels = fcluster(Z, t=num_clusters, criterion='maxclust')

        dendrogram(Z)
        plt.title('Hierarchical Clustering of Time Series')
        plt.xlabel('Time Series')
        plt.ylabel('Distance')
        #output dendrogram to png file
        plt.savefig('price_dendrogram.png')

        plt.clf()
        # join all series and plot them with cluster coloring
        plt.plot(np.concatenate(plotting).ravel())
        # shade cluster days depending on clusters
        for i in range(len(labels)):
            if labels[i] == 1:
                plt.axvspan(i*23400, (i+1)*23400, facecolor='red', alpha=0.5)
            elif labels[i] == 2:
                plt.axvspan(i*23400, (i+1)*23400, facecolor='blue', alpha=0.5)
            elif labels[i] == 3:
                plt.axvspan(i*23400, (i+1)*23400, facecolor='green', alpha=0.5)
            elif labels[i] == 4:
                plt.axvspan(i*23400, (i+1)*23400, facecolor='yellow', alpha=0.5)
        plt.title('Time Series')
        plt.xlabel('Time')
        plt.show()


        clusters = []
        for i in range(num_clusters):
            clusters.append([])

        for i in range(len(labels)):
                clusters[labels[i]-1].append(price_series[i])
        
        # if clusters only contain 1 data point, break
        for i in range(len(clusters)):
            print(len(clusters[i]))
            if len(clusters[i]) <= 1:
                return

        # Find medoid of each cluster
        medoids = np.empty((num_clusters, 23400))
        for x in range(len(clusters)):
            medoids[x] = find_medoid(clusters[x])

        # find cluster of test series
        test_cluster = assign_to_cluster(test_price, medoids, data_portion)

        # halve all price series and save to new variable
        X = price_series

        y = np.empty((len(result),data_portion))
        
        #episodic_lstm(X, y, test_price, data_portion, 64, 300, 10)

           # compute absolute difference between last value in x and y
        logger = mp.get_logger()  # Get the logger set up for multiprocessing
        logger.debug(f'Worker f starting')
        print(test_cluster)

        if model == 'lstm':
            normal_results.append(run_lstm(X, y, test_price, data_portion, 150, 50, 20, epoch =20))
        elif model == 'ffnn':
            normal_results.append(feed_foward_nn(X, y, test_price, data_portion, 64,100))
      
        #run_lstm_ns(X, y, test_price, data_portion, 150, 50, 1, epoch =30)

        print("moving on")

        cluster_series = np.empty((len(clusters[test_cluster]),X.shape[1]))

        for i in range(len(clusters[test_cluster])):
            cluster_series[i] = clusters[test_cluster][i]


        cluster_y = np.empty((len(clusters[test_cluster]),data_portion))

        print(cluster_series.shape)
        print("fitting cluster")

        if model == 'lstm':
            cluster_results.append(run_lstm(cluster_series, cluster_y, test_price, data_portion, 150, 50,20,epoch = 20))
        elif model == 'ffnn':
            nn_cluster.append(feed_foward_nn(X, y, test_price, data_portion, 64,100))

        return(normal_results, cluster_results)

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python test-cluster.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    num_clusters = int(sys.argv[1])
    data_portion = float(sys.argv[2])
    num_files = int(sys.argv[3])
    distancemetric = str(sys.argv[4])
    model = str(sys.argv[5])
    print(num_clusters, data_portion, num_files)
    lstm_cluster(num_clusters, data_portion, num_files, layers = 200, batch = 1, epoch = 10, distancemetric = distancemetric, model = model)

