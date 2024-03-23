import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import multiprocessing as mp
from get_data import get_simple_data
import sys
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from cluster_algs import hierarchical, kmeans, fuzzy_cmeans, spectral_clustering, dbscan, get_test_assignment
from distances import dtw_dist, euc_dist, correlation_dist, msc_dist
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
   def __init__(self, input_size=1, hidden_layer_size=128, output_size=1):
       super().__init__()
       self.hidden_layer_size = hidden_layer_size
       self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True, dropout=0.2, num_layers=1)
       self.linear1 = torch.nn.Linear(hidden_layer_size, output_size)

   def forward(self, x):
       # x.shape should be [batch_size, sequence_length, num_features]
       lstm_out, _ = self.lstm(x)
       last_time_step_out = lstm_out[:, -1, :]
       output = self.linear1(last_time_step_out)
       return output


def test_single_model(num_clusters, data_portion, num_files, layers = 50, batch = 100, epoch = 150,day_cutoff=22000 ):

    spy_files = sorted(glob.glob("C:\\Users\\leo\Documents\\clustering_hp\\data\\SPY\\*.csv"))[-40:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\leo\Documents\\clustering_hp\\data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

    data_portion = data_portion
    batch_size = batch
    cutoff = 23400 - day_cutoff

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
    

    
    test_data = result[num_files:]
    result = result[:num_files]
    price_series = result

    # clustering
    non_cut_price_series = price_series[:,:day_cutoff]
    price_series = price_series[:,day_cutoff:]
    test_data = test_data[:,day_cutoff:]
    non_cut_test_data = test_data[:,:day_cutoff]

    for i in range(price_series.shape[0]):
         price_series[i] = ((price_series[i]+1)/(price_series[i][0]+1)) - 1


    distance_matrix = dtw_dist(price_series)

    cluster_assignments = hierarchical(num_clusters, distance_matrix)
    
    price_series = non_cut_price_series

    for i in range (num_clusters):
        cluster_number = i
        # count number of points in this cluster
        print(cluster_assignments -1)
        count = np.count_nonzero(cluster_assignments -1 == i)
        cluster_series = np.empty((count, len(price_series[0])))
        idx = 0
        for j in range(len(cluster_assignments)):
            if cluster_assignments[j] -1 == i:
                cluster_series[idx] = price_series[j]
                idx += 1
        print(cluster_series.shape)
        X = cluster_series
        print(data_portion - cutoff)
        xdf = np.zeros((len(X), len(X[0]) - data_portion -cutoff, data_portion))
        ydf = np.zeros((len(X), len(X[0]) - data_portion -cutoff, cutoff))

        # create  a rolling window of x based on data portion
        for h in range(len(X)):
            for i in range(len(X[0]) - data_portion - cutoff):
                xdf[h][i] = X[h][i:i+data_portion] 
                # put next 200 data poitns in ydf
                ydf[h][i] = X[h][i+data_portion:i+data_portion+cutoff]
        
        #turn 3d into 2d arrays
        xdf = xdf.reshape(-1, xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)

        # Create the dataset
        dataset = TimeSeriesDataset(tensor_X, tensor_y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        model = LSTM(1, layers, cutoff)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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

    xdf = np.zeros((len(X), len(X[0]) - data_portion -cutoff, data_portion))
    ydf = np.zeros((len(X), len(X[0]) - data_portion -cutoff, cutoff))

    # create  a rolling window of x based on data portion
    for h in range(len(X)):
        for i in range(len(X[0]) - data_portion - cutoff):
            xdf[h][i] = X[h][i:i+data_portion] 
            # put next 200 data poitns in ydf
            ydf[h][i] = X[h][i+data_portion:i+data_portion+cutoff]

    # test data
    test_xdf = np.zeros((len(test_data), len(test_data[0]) - data_portion -cutoff, data_portion))
    test_ydf = np.zeros((len(test_data), len(test_data[0]) - data_portion -cutoff, cutoff))

    for h in range(len(test_data)):
        for i in range(len(test_data[0]) - data_portion - cutoff):
            test_xdf[h][i] = test_data[h][i:i+data_portion] 
            # put next 200 data poitns in ydf
            test_ydf[h][i] = test_data[h][i+data_portion:i+data_portion+cutoff] 

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
    

    cluster_results = np.zeros((len(test_ydf),cutoff))
    idx = 0

    d = 0
    for day in test_data:
        # get cluster membership
        cluster_membership = cluster_assignments[d] -1    # get cluster numbe
  
        # arrange day into xdf
        xdf = np.zeros((1, len(day) - data_portion -cutoff, data_portion))
        ydf = np.zeros((1, len(day) - data_portion -cutoff, cutoff))

        for i in range(len(day) - data_portion - cutoff):
            xdf[0][i] = day[i:i+data_portion] 
            # put next 200 data poitns in ydf
            ydf[0][i] = day[i+data_portion:i+data_portion+cutoff]

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
                    out = out.reshape(num_clusters, cutoff)

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
    data_portion = int(sys.argv[2])
    num_files = int(sys.argv[3])
    test_single_model(num_clusters, data_portion, num_files, layers = 512, batch = 1000, epoch = 50)
