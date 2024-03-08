## try logistic regression on the data
import numpy as np
import matplotlib.pyplot as plt
import glob
from get_data import get_simple_data
import sys
import skfuzzy as fuzz
import torch
import gc
from torch.utils.data import Dataset, DataLoader 
import multiprocessing as mp
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import soft_dtw
from multiprocessing import Pool

def test_single_model(num_clusters, data_portion, num_files, file_offset):

    files = sorted(glob.glob("C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\AAPL\\*.csv"))[-210 - file_offset:-file_offset]
    data_portion = int(23400 * data_portion)
    result = np.zeros((len(files),23400))
    plotting = np.zeros((len(files),23400))
    continous = np.zeros((len(files)* 23400))

    for x in range(len(files)):
        df = get_simple_data(0, 10000000, files[x], 's')
        df = df.iloc[0:23400]
        if df.shape[0] != 23400:
            continue
        df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2
        continous[x*23400:(x+1)*23400] = df["price"]
        plotting[x] = df['price'] /100000
        result[x] = ((df["price"]/df['price'].iloc[0]) -1) * 10
        

    test_data = result[num_files:]
    print(test_data.shape)
    result = result[:num_files]
    print(result.shape)
    price_series = result

    X = price_series
    # reshape data for scaling
   
    dist = 20
    num_steps = (len(X[-1]) - data_portion-200) // dist 
    xdf = np.zeros((len(X), num_steps, data_portion))
    ydf = np.zeros((len(X), num_steps, 200))
    # get superfluous data points
    superfluous = (len(X[-1]) - data_portion-200) % dist
        # Create a rolling window of x based on data portion, moving by 200 points each step
    for h in range(len(X)):
        for i in range(0, len(X[-1]) - data_portion - 200 - superfluous, dist):  # Step by 200
            # Calculate the index for storing the data, considering the step size
            index = i // dist 
            xdf[h][index] = X[h][i:i+data_portion]
            ydf[h][index] = X[h][i+data_portion:i+data_portion+200]


    xdf = xdf.reshape(-1, xdf.shape[-1])
    ydf = ydf.reshape(-1, ydf.shape[-1])
    # normalize each period by first value of the period
    for i in range(len(xdf)):
        xdf[i] = (xdf[i] +1) /( xdf[i][-1] +1) -1

    scaler = MinMaxScaler()
    xdf = scaler.fit_transform(xdf)
    ydf = np.where(ydf[:,-1] > ydf[:,0], 1, 0)

    # create a logistic regression model
    model = LogisticRegression(max_iter=50000)
    model.fit(xdf, ydf)
    model_accuracy = model.score(xdf.reshape(-1, data_portion), ydf)

    test_X = test_data
    test_xdf = np.zeros((len(test_X), len(test_X[0]) - data_portion - 200, data_portion))
    test_ydf = np.zeros((len(test_X), len(test_X[0]) - data_portion - 200, 200))
    
    # create a rolling window of x based on data portion
    for h in range(len(test_X)):
        for i in range(len(test_X[0]) - data_portion - 200):
            
            test_xdf[h][i] = test_X[h][i:i+data_portion]
            test_ydf[h][i] = test_X[h][i+data_portion:i+data_portion+200]
    
    #scaling
    test_xdf = test_xdf.reshape(-1, test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])
    for i in range(len(test_xdf)):
        test_xdf[i] = (test_xdf[i] +1) /( test_xdf[i][-1] +1) -1

    # rescale data
    test_xdf = scaler.fit_transform(test_xdf)
    test_ydf = np.where(test_ydf[:,-1] > test_ydf[:,0], 1, 0)
    print(test_xdf.shape)


    predictions = model.predict(test_xdf.reshape(-1, data_portion))
    probs = model.predict_proba(test_xdf.reshape(-1, data_portion))

    model_accuracy = np.mean(predictions == test_ydf)
    combined = np.column_stack((probs, test_ydf, predictions))
    print(model_accuracy)
    return combined

def run_model(params):
    num_clusters, data_portion, num_files, file_offset = params
    # Assuming `file_offset` is now a parameter of `test_single_model` and `test_cluster_model`
    if num_clusters == 1:  # Indicator for single model
        result = test_single_model(num_clusters, data_portion, num_files, file_offset)
        filename = f"single_results_{data_portion}_{file_offset}.npy"
    else:  # For cluster model
        result = test_cluster_model(num_clusters, data_portion, num_files, file_offset)
        filename = f"cluster_results_{num_clusters}_{data_portion}_{file_offset}.npy"
    
    # Save the result
    np.save(filename, result)
    return filename, result


def test_cluster_model(num_clusters, data_portion, num_files, file_offset):
    scaler = MinMaxScaler()
    files = sorted(glob.glob("C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\AAPL\\*.csv"))[-210-file_offset :-file_offset]
    data_portion = int(23400 * data_portion)
    result = np.zeros((len(files),23400))
    plotting = np.zeros((len(files),23400))
    continous = np.zeros((len(files)* 23400))

    for x in range(len(files)):
        df = get_simple_data(0, 10000000, files[x], 's')
        df = df.iloc[0:23400]
        if df.shape[0] != 23400:
            continue
        df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2
        continous[x*23400:(x+1)*23400] = df["price"]
        plotting[x] = df['price'] /100000
        result[x] = ((df["price"]/df['price'].iloc[0]) -1) * 10
        

    test_data = result[num_files:]
    print(test_data.shape)
    result = result[:num_files]
    print(result.shape)
    price_series = result

    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        result.T,  
        c=num_clusters,  # Number of clusters
        m=2,  # Fuzziness parameter
        error=0.005,  # Stopping criterion
        maxiter=10000,  # Maximum number of iterations
        init=None  # Initialization method (None for random)
    )

    # get cluster assignemnts for test data points based on existing clusters via fuzzy partition
    cluster_assignments = np.argmax(u, axis=0)


    # print cluster validation metrics
    print(f'Cluster center: {cntr}')
    print(f'Fuzzy partition coefficient: {fpc}')
    print(f'Error after {p} iterations: {fpc}')

    clusters_acc = np.zeros(num_clusters)
    models = []
    for i in range (num_clusters):
        cluster_number = i
        # count number of points in this cluster
        count = np.count_nonzero(cluster_assignments == i)
        cluster_series = np.empty((count, len(price_series[0])))
        idx = 0
        for j in range(len(cluster_assignments)):
            if cluster_assignments[j] == i:
                cluster_series[idx] = price_series[j]
                idx += 1
        
        X = cluster_series
        print(X.shape)

        dist = 20
        num_steps = (len(X[-1]) - data_portion-200) // dist  # Integer division to get the number of steps possible
        xdf = np.zeros((len(X), num_steps, data_portion))
        ydf = np.zeros((len(X), num_steps, 200))
        superfluous = (len(X[-1]) - data_portion-200) % dist

        print('building data')
    # Create a rolling window of x based on data portion, moving by 200 points each step
        for h in range(len(X)):
            for i in range(0, len(X[-1]) - data_portion - 200 -superfluous, dist):  # Step by 200
                # Calculate the index for storing the data, considering the step size
                index = i // dist
                xdf[h][index] = X[h][i:i+data_portion]
                ydf[h][index] = X[h][i+data_portion:i+data_portion+200]

        # drop cluster dimension
        xdf = xdf.reshape(-1, xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])

        # normalize each period by first value of the period
        for i in range(len(xdf)):
            xdf[i] = (xdf[i] +1) /( xdf[i][-1] +1) -1
        
        # scale data
        xdf = scaler.fit_transform(xdf)
        ydf = np.where(ydf[:,-1] > ydf[:,0], 1, 0)

        print('fitting model')
        # create a logistic regression model
        model = LogisticRegression(max_iter=10000)
        model.fit(xdf, ydf)
        models.append(model)
        # predict the test data
    # delete numpy arrays to free up memory
    del xdf, ydf
    gc.collect()

    cluster_assignments = []
    memberships = []
    # compare day to all cluster centers and find the closest one
    for day in test_data:
        # calculate distance to each cluster center
        distances = np.linalg.norm(cntr - day, axis=1)
        memberships.append(distances)
        # find the closest cluster center
        closest_cluster = np.argmin(distances)
        # assign the point to the closest cluster
        cluster_assignments.append(closest_cluster)
    test_cluster_assignments = np.array(cluster_assignments)

    preds = []
    gues = []
    test_X = test_data
    test_xdf = np.zeros((len(test_X), len(test_X[0]) - data_portion - 200, data_portion))
    test_ydf = np.zeros((len(test_X), len(test_X[0]) - data_portion - 200, 200))
    test_cdf = np.zeros((len(test_X), len(test_X[0]) - data_portion - 200))
    # create a rolling window of x based on data portion
    for h in range(len(test_X)):
        for i in range(len(test_X[0]) - data_portion - 200):
            
            test_xdf[h][i] = test_X[h][i:i+data_portion]
            test_cdf[h][i] = test_cluster_assignments[h]
            # put next 200 data poitns in ydf
            test_ydf[h][i] = test_X[h][i+data_portion:i+data_portion+200]
    
    #scaling
    test_xdf = test_xdf.reshape(-1, test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])
    for i in range(len(test_xdf)):
        test_xdf[i] = (test_xdf[i] +1) /( test_xdf[i][-1] +1) -1

    # rescale data
    test_xdf = scaler.fit_transform(test_xdf)
    test_cdf = test_cdf.reshape(-1)
    test_ydf = np.where(test_ydf[:,-1] > test_ydf[:,0], 1, 0)

    for i in range(len(test_xdf)):
        cluster_number = int(test_cdf[i])

        model = models[cluster_number]
        guess = model.predict(test_xdf[i].reshape(1, -1))
        prbs = model.predict_proba(test_xdf[i].reshape(1, -1))
        preds.append(prbs)
        gues.append(guess)

    
    preds = np.array(preds).reshape(-1, 2)
    gues = np.array(gues)
    print(preds.shape, gues.shape, test_ydf.shape)
    clusters_acc = np.mean(gues == test_ydf)


    combined = np.column_stack((preds, test_ydf, gues))
    combined = np.column_stack((combined, gues))
    
    return combined


def run_single_model(args):
    num_clusters, data_portion, num_files, file_offset = args
    return test_single_model(num_clusters, data_portion, num_files, file_offset)

def run_cluster_model(args):
    num_clusters, data_portion, num_files, file_offset = args
    return test_cluster_model(num_clusters, data_portion, num_files, file_offset)

def main():
    num_files = 100  # Example value, set as needed
    num_clusters = 5  # Example value, set as needed

    # Prepare arguments for multiprocessing
    single_model_args = [(num_clusters, i / 10, num_files, file_offset) for i in range(1, 10) for file_offset in range(0, 301, 10)]
    cluster_model_args = [(h, i / 10, num_files, file_offset) for i in range(1, 10) for h in range(2, 20) for file_offset in range(0, 301, 10)]

    # Create a multiprocessing pool
    with Pool(processes=30) as pool:
        # Run single model tests
        single_results = pool.map(run_single_model, single_model_args)
        # Save the results with correct filenames
        for args, result in zip(single_model_args, single_results):
            _, data_portion, _, file_offset = args
            np.save(f"single_results_dp_{data_portion}_fo_{file_offset}", result)

        # Run cluster model tests
        cluster_results = pool.map(run_cluster_model, cluster_model_args)
        # Save the results with correct filenames
        for args, result in zip(cluster_model_args, cluster_results):
            h, data_portion, _, file_offset = args
            np.save(f"cluster_results_{h}_dp_{data_portion}_fo_{file_offset}", result)


if __name__ == "__main__":
    main()



