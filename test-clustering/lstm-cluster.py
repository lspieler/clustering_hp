import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from get_data import get_simple_data
from test_cluster import fdtw_clustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster
from test_cluster import find_medoid

from test_cluster import assign_to_cluster
from lstm import run_lstm, cluster_lstm


def lstm_cluster(num_clusters, data_portion, num_files, layers = 50, batch = 100, epoch = 150, ):
    data_length = 23400 * data_portion
 
    msgs = sorted(glob.glob('/mnt/research/d.byrd/lobster/sp500_lev1_sec1/AAPL/AAPL_*.csv'))
    print(len(msgs))
    input("Press enter to continue")

    result = np.zeros((num_files,23400))

    x = 0
    for x in range(len(msgs)):
        df = get_simple_data(0, 10000000, msgs[x], 100000)
        df = df.iloc[0:23400]
        df["price"] = (df["ask_1"] + df["bid_1"]).bfill().ffill()/2 * 1000
        result[x] = (df["price"]/df['price'].iloc[0] -1 )

    
    
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


    run_lstm(X, y, test_price, data_portion, layers, layers, batch, epoch)
    cluster_lstm(clusters, test_cluster, test_price, data_portion, layers, batch, epoch) 


if __name__ == "__main__":
    lstm_cluster(2, 0.5, 1)



