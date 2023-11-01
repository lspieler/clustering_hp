import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool, Manager, Process
from get_data import get_simple_data
from test_cluster import fdtw_clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from test_cluster import find_medoid
import sys
from test_cluster import assign_to_cluster
from lstm import run_lstm, cluster_lstm
import time

def update_monitor(shared_updates, lock):

    print("Starting update monitor")
    with open('updates.txt', 'a') as file:
        while True:
            while shared_updates:  # While there are updates in the list
                update = shared_updates.pop(0)  # Remove and get the first update
                with lock:
                     file.write(f'{update}\n')
            time.sleep(5) 

def lstm_cluster(num_clusters, data_portion, num_files, layers = 50, batch = 100, epoch = 150):
    data_length = 23400
    data_portion = int(data_length * data_portion)
    

    msgs = sorted(glob.glob('/Users/lspieler/Semesters/Fall 23/Honors Project/test-clustering/data/AAPL/AAPL_*.csv'))
        

    args = [(f, msgs, num_files, num_clusters, data_portion, layers, batch, epoch) for f in range(len(msgs) - num_files)]
    with Pool(processes=6) as pool:
        results = pool.starmap(process_files, args)
  
     # If you need to process results
    all_normal_results = [res[0] for res in results]
    all_cluster_results = [res[1] for res in results]
    
    # write output to file
    with open(f"lstm-cluster-{num_clusters}-{data_portion}-{num_files}-{layers}-{batch}-{epoch}.txt", "w") as f:
        f.write(f"normal results: {all_normal_results}")
        f.write(f"cluster results: {all_cluster_results}")


def process_files(f, msgs, num_files, num_clusters, data_portion, layers, batch, epoch):
        normal_results = []
        cluster_results = []
        print(f)
        i = num_files + f
        files = msgs[f: i]
        print(len(files))

        result = np.zeros((num_files,23400))

        x = 0
        for x in range(len(files)):
            df = get_simple_data(0, 10000000, files[x], 100000)
            df = df.iloc[0:23400]
            df["price"] = (df["ask_1"] + df["bid_1"]).bfill().ffill()/2
            result[x] = (df["price"]/df['price'].iloc[0] -1 ) * 100



        test_price = result[-1]
        result = result[:-1]


        price_distance_matrix = fdtw_clustering(result)

        price_series = result

        condensed_D1 = squareform(price_distance_matrix)

        Z = linkage(condensed_D1, 'ward')

        labels = fcluster(Z, t=num_clusters, criterion='maxclust')

        clusters = []
        for i in range(num_clusters):
            clusters.append([])

        for i in range(len(labels)):
                clusters[labels[i]-1].append(price_series[i])
        
        # if clusters only contain 1 data point, break
        for i in range(len(clusters)):
            if len(clusters[i]) <= 1:
                break

        # Find medoid of each cluster
        medoids = np.empty((num_clusters, 23400))
        for x in range(len(clusters)):
            medoids[x] = find_medoid(clusters[x])

        # find cluster of test series
        test_cluster = assign_to_cluster(test_price, medoids, data_portion)

        # halve all price series and save to new variable
        X = np.empty((len(result),data_portion))
        for x in range(len(price_series)):
            X[x] = price_series[x][0:data_portion]


        print(price_series.shape)
        #get final price of each day as the y
        y = np.empty((len(result),data_portion))
        for x in range(len(price_series)):
            y[x] = price_series[x][data_portion:]

        #adjust test price y
 

        # compute absolute difference between last value in x and y
    

        normal_results.append(run_lstm(X, y, test_price, data_portion, 100, layers, 1, epoch =250))


        cluster_results.append(cluster_lstm(clusters, test_cluster, test_price, data_portion, layers, batch, epoch = 30))

        return(normal_results, cluster_results)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test-cluster.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    num_clusters = int(sys.argv[1])
    data_portion = float(sys.argv[2])
    num_files = int(sys.argv[3])
    print(num_clusters, data_portion, num_files)
    lstm_cluster(num_clusters, data_portion, num_files, layers = 30, batch = 1, epoch = 100)

