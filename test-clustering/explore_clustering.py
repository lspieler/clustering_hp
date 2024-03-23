
from get_data import get_data
import numpy as np
import pandas as pd
import glob
from get_data import get_simple_data
from distances import dtw_dist, euc_dist, correlation_dist, maximum_shifting_correlation_dist
from cluster_algs import hierarchical, dbscan, spectral_clustering, kmeans, fuzzy_cmeans, get_test_assignment
import matplotlib.pyplot as plt
from preprocessing import fft_transform
from simple_linear import simple_multi_linear
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score

# Calculate silhouette score for each clustering algorithm



def testing(distance = 'dtw', pre = None, num_cluster = 3, num_files = 20, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True):
    data_portion = data_portion
    num_files = num_files
    day_cutoff = day_cutoff

    spy_files = sorted(glob.glob("C:\\Users\\leo\Documents\\clustering_hp\\data\\SPY\\*.csv"))[-40:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\leo\Documents\\clustering_hp\\data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

    data_portion = data_portion

    result = np.zeros((len(files),23400))

    for x in range(len(files)):
        df = get_simple_data(0, 10000000, files[x], 's')
        spy = get_simple_data(0, 10000000, spy_files[x], 's')
        spy = spy.iloc[0:23400]
        df = df.iloc[0:23400]
        if df.shape[0] != 23400:
            continue
        df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2
        spy["price"] = (spy["ask_1"] + spy["bid_1"]).ffill().bfill()/2
        normalized_price = ((df["price"]/df['price'].iloc[0]) -1)
        normalized_spy = ((spy["price"]/spy['price'].iloc[0]) -1)
        result[x] = (normalized_price - normalized_spy) * 10


    # remove zero value days
    result = result[~np.all(result == 0.0, axis=1)]


    test_data = result[num_files:]
    result = result[:num_files]
    price_series = result
    non_cut_price_series = price_series[:,:day_cutoff]

    # take day cutoff and cluster based on last part of day
    if inverse:
        price_series = price_series[:,day_cutoff:]
    else:
        price_series = price_series[:,:day_cutoff]


    for i in range(price_series.shape[0]):
        price_series[i] = ((price_series[i]+1)/(price_series[i][0]+1)) - 1

    if pre == 'fft':
        for i in range(price_series.shape[0]):
            price_series[i] = fft_transform(price_series[i], detrend = True)

    # for each pirce series devide by the first value and subtract 1
    

    if distance == 'dtw':    
        distance_matrix = dtw_dist(price_series)
    elif distance == 'euc':
        distance_matrix = euc_dist(price_series)
    elif distance == 'correlation':
        distance_matrix = correlation_dist(price_series) 
    elif distance == 'max_shift_correlation':
        distance_matrix = maximum_shifting_correlation_dist(price_series)

    if normalize:
        distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

    print(distance_matrix)
   
    hierarchical_clusters = hierarchical(num_clusters = num_cluster, dist_matrix=distance_matrix)
    #dbscan_clusters = dbscan(dist_matrix=distance_matrix)
    spectral_clusters = spectral_clustering(num_clusters=num_cluster, dist_matrix=distance_matrix)
    kmeans_clusters = kmeans(num_clusters=num_cluster, price_series = price_series)
    fuzzy_clusters = fuzzy_cmeans(num_clusters=num_cluster, price_series= price_series)

    silhouette_hierarchical = silhouette_score(distance_matrix, hierarchical_clusters)
    silhouette_spectral = silhouette_score(distance_matrix, spectral_clusters)
    silhouette_kmeans = silhouette_score(distance_matrix, kmeans_clusters)
    silhouette_fuzzy = silhouette_score(distance_matrix, fuzzy_clusters)


    ch_hierarchical = calinski_harabasz_score(price_series, hierarchical_clusters)
    ch_spectral = calinski_harabasz_score(price_series, spectral_clusters)
    ch_kmeans = calinski_harabasz_score(price_series, kmeans_clusters)
    ch_fuzzy = calinski_harabasz_score(price_series, fuzzy_clusters)

    
    non_cut_test_data = test_data[:,:day_cutoff]  
    test_data = test_data[:,day_cutoff:]


    assignemnts = get_test_assignment(non_cut_test_data, hierarchical_clusters, non_cut_price_series, "euclidean")
    print(assignemnts)
    

    # get test assignments 

    """
    # plot series where x is intial value and y is final value and color by cluster for all algs
    price_series = result[:,day_cutoff:]
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.scatter(price_series[:,0], price_series[:,-1], c=hierarchical_clusters)
    plt.xlabel('Initial Value')
    plt.ylabel('Final Value')
    plt.title('Hierarchical Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.scatter(price_series[:,0], price_series[:,-1], c=dbscan_clusters)
    plt.xlabel('Initial Value')
    plt.ylabel('Final Value')
    plt.title('DBSCAN Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.scatter(price_series[:,0], price_series[:,-1], c=spectral_clusters)
    plt.xlabel('Initial Value')
    plt.ylabel('Final Value')
    plt.title('Spectral Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.scatter(price_series[:,0], price_series[:,-1], c=kmeans_clusters)
    plt.xlabel('Initial Value')
    plt.ylabel('Final Value')
    plt.title('K-means Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.scatter(price_series[:,0], price_series[:,-1], c=fuzzy_clusters)
    plt.xlabel('Initial Value')
    plt.ylabel('Final Value')
    plt.title('Fuzzy C-means Clustering')
    plt.colorbar()

    plt.tight_layout()
    # save fig
    #plt.savefig(f'./plots/initial_final_{distance}_{pre}_{num_cluster}_{num_files}_{data_portion}_{day_cutoff}_{normalize}.png')
    plt.close()
    # plot series by varaince for all algs
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.scatter(price_series.var(axis=1), price_series.mean(axis=1), c=hierarchical_clusters)
    plt.xlabel('Variance')
    plt.ylabel('Mean')
    plt.title('Hierarchical Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.scatter(price_series.var(axis=1), price_series.mean(axis=1), c=dbscan_clusters)
    plt.xlabel('Variance')
    plt.ylabel('Mean')
    plt.title('DBSCAN Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.scatter(price_series.var(axis=1), price_series.mean(axis=1), c=spectral_clusters)
    plt.xlabel('Variance')
    plt.ylabel('Mean')
    plt.title('Spectral Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.scatter(price_series.var(axis=1), price_series.mean(axis=1), c=kmeans_clusters)
    plt.xlabel('Variance')
    plt.ylabel('Mean')
    plt.title('K-means Clustering')
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.scatter(price_series.var(axis=1), price_series.mean(axis=1), c=fuzzy_clusters)
    plt.xlabel('Variance')
    plt.ylabel('Mean')
    plt.title('Fuzzy C-means Clustering')
    plt.colorbar()

    plt.tight_layout()
    # save fig
    #plt.savefig(f'./plots/mean_var_{distance}_{pre}_{num_cluster}_{num_files}_{data_portion}_{day_cutoff}_{normalize}.png')
    plt.close()

    # plot original series by cluster for all algs

    plt.figure(figsize=(12, 8))
    cluster_colors = ['r', 'g', 'b']    
    plt.subplot(2, 3, 1)
    for i in range(3):
        plt.plot(non_cut_price_series[hierarchical_clusters == i].T, color=cluster_colors[i])
    plt.title('Hierarchical Clustering')

    plt.subplot(2, 3, 2)
    for i in range(3):
        plt.plot(non_cut_price_series[dbscan_clusters == i].T, color=cluster_colors[i])
    plt.title('DBSCAN Clustering')

    plt.subplot(2, 3, 3)
    for i in range(3):
        plt.plot(non_cut_price_series[spectral_clusters == i].T, color=cluster_colors[i])
    plt.title('Spectral Clustering')

    plt.subplot(2, 3, 4)
    for i in range(3):
        plt.plot(non_cut_price_series[kmeans_clusters == i].T, color=cluster_colors[i])
    plt.title('K-means Clustering')

    plt.subplot(2, 3, 5)
    for i in range(3):
        plt.plot(non_cut_price_series[fuzzy_clusters == i].T, color=cluster_colors[i])
    plt.title('Fuzzy C-means Clustering')

    plt.tight_layout()
    # save fig
    #plt.savefig(f'./plots/series_{distance}_{pre}_{num_cluster}_{num_files}_{data_portion}_{day_cutoff}_{normalize}.png')
    plt.clf()
    plt.close()
    """

    return silhouette_hierarchical, silhouette_spectral, silhouette_kmeans, silhouette_fuzzy, ch_hierarchical, ch_spectral, ch_kmeans, ch_fuzzy


    

  




if __name__ == "__main__":
    scores = []
    for x in range (2,10):
        print(x)

        scores.append(testing(distance = 'dtw', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))

        scores.append(testing(distance = 'euc', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))
       
        #scores.append(testing(distance = 'correlation', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))

        scores.append(testing(distance = 'max_shift_correlation', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))
    
        scores.append(testing(distance = 'dtw', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse =True))
        scores.append(testing(distance = 'euc', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))
        #scores.append(testing(distance = 'correlation', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))
        scores.append(testing(distance = 'max_shift_correlation', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = False, inverse = True))
        scores.append(testing(distance = 'dtw', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        scores.append(testing(distance = 'euc', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        #scores.append(testing(distance = 'correlation', pre = None , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        scores.append(testing(distance = 'max_shift_correlation', pre = None , num_cluster = 3, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        scores.append(testing(distance = 'dtw', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        scores.append(testing(distance = 'euc', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        #scores.append(testing(distance = 'correlation', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))
        scores.append(testing(distance = 'max_shift_correlation', pre = 'fft' , num_cluster = x, num_files = 30, data_portion = 512, day_cutoff = 22000, normalize = True, inverse = True))


    scores = np.array(scores)

    # find the best silhouette score overall and print the indecies
    silhouette_scores = scores[:,0:4]
    ch_scores = scores[:,4:]

    kmeans_silhouette = silhouette_scores[:,2]
    kmeans_ch = ch_scores[:,2]

    hierarchical_silhouette = silhouette_scores[:,0]
    hierarchical_ch = ch_scores[:,0]

    spectral_silhouette = silhouette_scores[:,1]
    spectral_ch = ch_scores[:,1]

    fuzzy_silhouette = silhouette_scores[:,3]
    fuzzy_ch = ch_scores[:,3]

    print(f"Kmeans Silhouette: {kmeans_silhouette.max()} at {kmeans_silhouette.argmax()}")
    print(f"Kmeans CH: {kmeans_ch.max()} at {kmeans_ch.argmax()}")
    print(f"Hierarchical Silhouette: {hierarchical_silhouette.max()} at {hierarchical_silhouette.argmax()}")
    print(f"Hierarchical CH: {hierarchical_ch.max()} at {hierarchical_ch.argmax()}")
    print(f"Spectral Silhouette: {spectral_silhouette.max()} at {spectral_silhouette.argmax()}")
    print(f"Spectral CH: {spectral_ch.max()} at {spectral_ch.argmax()}")
    print(f"Fuzzy Silhouette: {fuzzy_silhouette.max()} at {fuzzy_silhouette.argmax()}")
    print(f"Fuzzy CH: {fuzzy_ch.max()} at {fuzzy_ch.argmax()}")

        

    