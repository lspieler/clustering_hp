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
from lstm import run_lstm, cluster_lstm
import multiprocessing as mp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import datetime
import skfuzzy as fuzz

import torch
from torch.utils.data import Dataset, DataLoader

def calculate_medoids(price_series, cluster_assignments, num_clusters):
    medoids = []
    for i in range(num_clusters):
        cluster_data = price_series[cluster_assignments == i]
        medoid = np.mean(cluster_data, axis=0)
        medoids.append(medoid)
    return medoids

def similarity(distance, bandwidth):
    """
    Calculate the similarity as an exponential decay function of the distance.
    'bandwidth' controls the rate of decay (fuzziness).
    """
    return np.exp(-distance**2 / (2. * bandwidth**2))

def calculate_fuzzy_memberships(new_data, medoids, bandwidth):
    memberships = []
    for data_point in new_data:
        distances = [np.linalg.norm(data_point - medoid) for medoid in medoids]
        similarities = [similarity(d, bandwidth) for d in distances]
        total_similarity = sum(similarities)
        normalized_memberships = [s / total_similarity for s in similarities]
        memberships.append(normalized_memberships)
    return np.array(memberships)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        current_sample = self.data[idx]
        current_target = self.targets[idx]
        return torch.tensor(current_sample, dtype=torch.float), torch.tensor(current_target, dtype=torch.float)

class LSTM(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size)

        self.linear = torch.nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        # Detach the hidden state to prevent backpropagating through the entire sequence
        self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions


def test_single_model(num_clusters, data_portion, num_files, layers = 50, batch = 100, epoch = 150, ):

    batch_size = batch # Define your batch size
    files = sorted(glob.glob('/Users/lspieler/Semesters/Fall 23/Honors Project/test-clustering/data/AAPL/AAPL_*.csv'))[:16]
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
     

    continous = continous / continous[0]

    test_data = result[num_files:]

    print(test_data.shape)
    result = result[:num_files]
    print(result.shape)
    price_series = result

 
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        result.T,  
        c=num_clusters, 
        m=2,  # Fuzziness parameter
        error=0.005,  # Stopping criterion
        maxiter=1000,  # Maximum number of iterations
        init=None  # Initialization method (None for random)
    )

    cluster_assignments = np.argmax(u, axis=0)
    print(cluster_assignments)
    
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

        xdf = np.zeros((len(X), len(X[0]) - data_portion -200, data_portion))
        ydf = np.zeros((len(X), len(X[0]) - data_portion -200, 200))

        # create  a rolling window of x based on data portion
        for h in range(len(X)):
            for i in range(len(X[0]) - data_portion - 200):
                xdf[h][i] = X[h][i:i+data_portion] 
                # put next 200 data poitns in ydf
                ydf[h][i] = X[h][i+data_portion:i+data_portion+200]
        
        #turn 3d into 2d arrays
        xdf = xdf.reshape(-1, xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)

        # Create the dataset
        dataset = TimeSeriesDataset(tensor_X, tensor_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model = LSTM(xdf.shape[-1], layers, ydf.shape[-1])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        for epoch in range(epoch):
            for inputs, targets in dataloader:
            
                # Zero out the gradients
                model.zero_grad()
                # Perform forward pass
                y_pred = model(inputs)
                # Compute loss
                loss = torch.sqrt(torch.mean((y_pred - targets)**2))
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
            # Print loss for every epoch
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

        # save model
        torch.save(model.state_dict(), f"models/lstm_cluster_{cluster_number}.pth")

        
                
    
    # halve all price series and save to new variable
    X = price_series

    xdf = np.zeros((len(X), len(X[0]) - data_portion -200, data_portion))
    ydf = np.zeros((len(X), len(X[0]) - data_portion -200, 200))

    # create  a rolling window of x based on data portion
    for h in range(len(X)):
        for i in range(len(X[0]) - data_portion - 200):
            xdf[h][i] = X[h][i:i+data_portion] 
            # put next 200 data poitns in ydf
            ydf[h][i] = X[h][i+data_portion:i+data_portion+200]

    # test data
    test_xdf = np.zeros((len(test_data), len(test_data[0]) - data_portion -200, data_portion))
    test_ydf = np.zeros((len(test_data), len(test_data[0]) - data_portion -200, 200))

    for h in range(len(test_data)):
        for i in range(len(test_data[0]) - data_portion - 200):
            test_xdf[h][i] = test_data[h][i:i+data_portion] 
            # put next 200 data poitns in ydf
            test_ydf[h][i] = test_data[h][i+data_portion:i+data_portion+200] 

    #turn 3d into 2d arrays
    xdf = xdf.reshape(-1, xdf.shape[-1])
    ydf = ydf.reshape(-1, ydf.shape[-1])

    
    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(xdf, dtype=torch.float)
    tensor_y = torch.tensor(ydf, dtype=torch.float)

    # Create the dataset
    dataset = TimeSeriesDataset(tensor_X, tensor_y)

    batch_size = batch # Define your batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = LSTM(xdf.shape[-1], layers, ydf.shape[-1])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(epoch):
        for inputs, targets in dataloader:
       
            # Zero out the gradients
            model.zero_grad()
            # Perform forward pass
            y_pred = model(inputs)
            # Compute loss
            loss = torch.sqrt(torch.mean((y_pred - targets)**2))
            # Perform backward pass
            loss.backward()
            # Perform optimization
            optimizer.step()
        # Print loss for every epoch
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

    # save model
    torch.save(model.state_dict(), f"models/lstm.pth")

    medoids = calculate_medoids(price_series, cluster_assignments, num_clusters)
    fuzzy_memberships = calculate_fuzzy_memberships(test_data, medoids, bandwidth=20)

    print("Testing model")
    # test on the test data
    test_xdf = test_xdf.reshape(-1, test_xdf.shape[-1]) 
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1]) 

    print(test_ydf.shape)
    
    test_tensor_X = torch.tensor(test_xdf, dtype=torch.float)
    test_tensor_y = torch.tensor(test_ydf, dtype=torch.float)

    test_dataset = TimeSeriesDataset(test_tensor_X, test_tensor_y)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
   
    normal_results = np.zeros((len(test_ydf), 200))

    model.load_state_dict(torch.load(f"models/lstm.pth"))
    with torch.no_grad():
        idx = 0
        for inputs, targets in test_dataloader:
            y_pred = model(inputs)
            test_loss = torch.sqrt(torch.mean((y_pred - targets)**2))
            normal_results[idx] = y_pred
            idx += 1

 
    cluster_models = []
    for i in range(num_clusters):
        model = LSTM(xdf.shape[-1], layers, ydf.shape[-1])
        model.load_state_dict(torch.load(f"models/lstm_cluster_{i}.pth"))
        cluster_models.append(model)
    

    cluster_results = np.zeros((len(test_ydf), 200))
    idx = 0

    d = 0
    for day in test_data:
        # get cluster membership
        cluster_membership = fuzzy_memberships[d]        # get cluster numbe
  
        # arrange day into xdf
        xdf = np.zeros((1, len(day) - data_portion -200, data_portion))
        ydf = np.zeros((1, len(day) - data_portion -200, 200))

        for i in range(len(day) - data_portion - 200):
            xdf[0][i] = day[i:i+data_portion] 
            # put next 200 data poitns in ydf
            ydf[0][i] = day[i+data_portion:i+data_portion+200]

        #turn 3d into 2d arrays
        xdf = xdf.reshape(-1, xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])

        print(ydf.shape)
        print(xdf.shape)

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)

        # Create the dataset
        dataset = TimeSeriesDataset(tensor_X, tensor_y)

        batch_size = 1 # Define your batch size
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

        # cluster predictions 
        with torch.no_grad():
                for inputs, targets in dataloader:
                 
                    out = []
                    for model in cluster_models:
                        model.eval()
                        y_pred = model(inputs)
                        test_loss = torch.sqrt(torch.mean((y_pred - targets)**2))
                        out.append(y_pred.detach().numpy())
                    # apply cluster membership to model outputs
                    #reshape out
                    out = np.array(out)
                    out = out.reshape(num_clusters, 200)

                    cluster_results[idx] = np.average(out, axis=0, weights=cluster_membership)
                    idx += 1
        d += 1


    print(cluster_results)
    print(normal_results)
    normal_loss = mean_squared_error(test_ydf, normal_results, squared = False)
    print("Normal loss: ", normal_loss)

    cluster_loss = mean_squared_error(test_ydf, cluster_results, squared = False)
    print("Cluster loss: ", cluster_loss)

    return normal_loss, cluster_loss



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python single_model_test.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    num_clusters = int(sys.argv[1])
    data_portion = float(sys.argv[2])
    num_files = int(sys.argv[3])
    test_single_model(num_clusters, data_portion, num_files, layers = 200, batch = 1000, epoch = 50)
