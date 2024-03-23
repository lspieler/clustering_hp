import numpy as np
import matplotlib.pyplot as plt
import glob
from get_data import get_simple_data
import sys
import skfuzzy as fuzz
import torch
from torch.utils.data import Dataset, DataLoader 
import multiprocessing as mp
from tqdm import tqdm
# import scaler
from distances import dtw_dist, euc_dist, correlation_dist, maximum_shifting_correlation_dist
from cluster_algs import kmeans, hierarchical, dbscan, spectral_clustering, fuzzy_cmeans, get_test_assignment
from sklearn.preprocessing import MinMaxScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.current_device())  # Returns the current device index
print(torch.cuda.device_count())  # Returns the number of GPUs available
print(torch.cuda.get_device_name(0)) 
torch.cuda.empty_cache()

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
   
class Clasifier(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=128, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, batch_first=True, dropout=0.2, num_layers=1)
        self.linear = torch.nn.Linear(hidden_layer_size, hidden_layer_size*2)
        self.linear1 = torch.nn.Linear(hidden_layer_size*2, output_size)
        self.classify = torch.nn.Linear(output_size, 2)

    def forward(self, x):
        # x.shape should be [batch_size, sequence_length, num_features]
        lstm_out, _ = self.lstm(x)
        last_time_step_out = lstm_out[:, -1, :]
        hidden = self.linear(last_time_step_out)
        output = self.linear1(hidden)
        output = self.classify(output)
        output = torch.nn.functional.softmax(output, dim=1)
        return output


def test_single_model(num_clusters, data_portion, num_files, layers = 32, batch = 10, epoch = 20, day_cutoff = 22000):

    spy_files = sorted(glob.glob("C:\\Users\\leo\Documents\\clustering_hp\\data\\SPY\\*.csv"))[-40:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\leo\Documents\\clustering_hp\\data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

    data_portion = data_portion
    batch_size = batch

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
    inter_test = test_data

    # clustering
    non_cut_price_series = price_series[:,:day_cutoff]
    price_series = price_series[:,day_cutoff:]
    test_data = test_data[:,day_cutoff:]
    non_cut_test_data = test_data[:,:day_cutoff]

    for i in range(price_series.shape[0]):
         price_series[i] = ((price_series[i]+1)/(price_series[i][0]+1)) - 1

    distance_matrix = dtw_dist(price_series)
    cluster_assignments = hierarchical(num_clusters, distance_matrix)
    test_cluster_assignments = get_test_assignment(non_cut_test_data, cluster_assignments, non_cut_price_series, distance_metric='fastdtw')
    
    X = result
    dist = 20
    num_steps = (len(X[-1]) - data_portion -20) // dist  # Integer division to get the number of steps possible
    print(num_steps)
    # Adjust the shapes of xdf and ydf to account for the step size of 200
    xdf = np.zeros((len(X), num_steps, data_portion, 1))
    ydf = np.zeros((len(X), num_steps, 20))
    residual = (len(X[-1]) - data_portion - 20) % dist

    i = 0
    for h in range(len(X)):
        for i in range(0, len(X[-1]) - data_portion - 20 -residual, dist):  # Step by 200
            # Calculate the index for storing the data, considering the step size
            index = i // dist
            xdf[h][index, :, 0] = X[h][i:i+data_portion]
            # Put numerical cluster assignment in xdf
            #xdf[h][index, :, 1] = cluster_assignments[h]
            # Put next 200 data points in ydf
            ydf[h][index] = X[h][i+data_portion:i+data_portion+20]


    xdf = xdf.reshape(-1, xdf.shape[-2], xdf.shape[-1])
    ydf = ydf.reshape(-1, ydf.shape[-1])
    
    i = 0
    for i in range(len(xdf)):
        xdf[i] =((xdf[i] + 1) / (xdf[i][-1] + 1))-1

    scaler = MinMaxScaler()
    print(xdf.shape, ydf.shape)
    # scale only price data, not cluster assignments
    xdf[:, :, 0] = scaler.fit_transform(xdf[:, :, 0].reshape(-1, 1)).reshape(-1, xdf.shape[-2])

    print(xdf.shape, ydf.shape)
    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(xdf, dtype=torch.float)
    tensor_y = torch.tensor(ydf, dtype=torch.float)

   
    dataset = TimeSeriesDataset(tensor_X, tensor_y)
    batch_size = batch 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = LSTM(input_size=1, hidden_layer_size=layers, output_size=20)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
    losses = np.zeros(epoch)
    model.to(device) 
    # Training loop
    epochs = epoch
    for epoch in range(epochs):
        for batch_X, batch_y in tqdm(dataloader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
       
            model.zero_grad()
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        losses[epoch] = loss.item()
  
    X = inter_test

    num_steps = (len(X[-1]) - data_portion -20) // dist  # Integer division to get the number of steps possible
    # Adjust the shapes of xdf and ydf to account for the step size of 200
    test_xdf = np.zeros((len(X), num_steps, data_portion, 1))
    test_ydf = np.zeros((len(X), num_steps, 20))
    residual = (len(X[-1]) - data_portion - 20) % dist

    i = 0
    for h in range(len(X)):
        for i in range(0, len(X[-1]) - data_portion - 20 -residual, dist):  # Step by 200
            # Calculate the index for storing the data, considering the step size
            index = i // dist
            test_xdf[h][index, :, 0] = X[h][i:i+data_portion]
            # Put numerical cluster assignment in xdf
            #xdf[h][index, :, 1] = cluster_assignments[h]
            # Put next 200 data points in ydf
            test_ydf[h][index] = X[h][i+data_portion:i+data_portion+20]


    test_xdf =  test_xdf.reshape(-1, test_xdf.shape[-2], test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])
        
    i = 0
    for i in range(len(test_xdf)):
        test_xdf[i] =((test_xdf[i] + 1) / (test_xdf[i][-1] + 1))-1

    scaler = MinMaxScaler()
    print(test_xdf.shape, test_ydf.shape)
    # scale only price data, not cluster assignments
    test_xdf[:, :, 0] = scaler.fit_transform(test_xdf[:, :, 0].reshape(-1, 1)).reshape(-1, test_xdf.shape[-2])

    print(test_xdf, test_ydf)

    test_xdf = test_xdf.reshape(-1, test_xdf.shape[-2], test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])

    tensor_test_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_test_y = torch.tensor(test_ydf, dtype=torch.float)

    test_dataset = TimeSeriesDataset(tensor_test_X, tensor_test_y)
    model.cpu()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(tensor_test_X)
    print(predictions, tensor_test_y)
    #copy tensor to cpu and numpy
    loss = criterion(predictions, tensor_test_y)
    print(f"Test Loss: {loss}")

    return losses


def test_cluster_model(num_clusters, data_portion, num_files, layers = 32, batch = 10, epoch = 20, day_cutoff = 22000):

    spy_files = sorted(glob.glob("C:\\Users\\leo\Documents\\clustering_hp\\data\\SPY\\*.csv"))[-40:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\leo\Documents\\clustering_hp\\data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

    data_portion = data_portion
    batch_size = batch

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
    price_series = result[:num_files]
    

    # clustering
    non_cut_price_series = price_series[:,:day_cutoff]
    price_series = price_series[:,day_cutoff:]
    test_data = test_data[:,day_cutoff:]
    non_cut_test_data = test_data[:,:day_cutoff]

    for i in range(price_series.shape[0]):
         price_series[i] = ((price_series[i]+1)/(price_series[i][0]+1)) - 1

    distance_matrix = dtw_dist(price_series)
    cluster_assignments = hierarchical(num_clusters, distance_matrix)
    test_cluster_assignments = get_test_assignment(non_cut_test_data, cluster_assignments, non_cut_price_series, distance_metric='fastdtw')
    
    price_series = result[:num_files]
    test_data = result[num_files:]

    
    cutoff = 23400 - day_cutoff
    dist = 20

    models = []
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

    
        num_steps = (len(X[-1]) - data_portion -20) // dist  # Integer division to get the number of steps possible
        print(num_steps)
        # Adjust the shapes of xdf and ydf to account for the step size of 200
        xdf = np.zeros((len(X), num_steps, data_portion, 1))
        ydf = np.zeros((len(X), num_steps, 20))
        residual = (len(X[-1]) - data_portion - 20) % dist

        i = 0
        for h in range(len(X)):
            for i in range(0, len(X[-1]) - data_portion - 20 -residual, dist):  # Step by 200
                # Calculate the index for storing the data, considering the step size
                index = i // dist
                xdf[h][index, :, 0] = X[h][i:i+data_portion]
                # Put numerical cluster assignment in xdf
                #xdf[h][index, :, 1] = cluster_assignments[h]
                # Put next 200 data points in ydf
                ydf[h][index] = X[h][i+data_portion:i+data_portion+20]


        xdf = xdf.reshape(-1, xdf.shape[-2], xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])
        
        i = 0
        for i in range(len(xdf)):
            xdf[i] =((xdf[i] + 1) / (xdf[i][-1] + 1))-1

        scaler = MinMaxScaler()
        print(xdf.shape, ydf.shape)
        # scale only price data, not cluster assignments
        xdf[:, :, 0] = scaler.fit_transform(xdf[:, :, 0].reshape(-1, 1)).reshape(-1, xdf.shape[-2])

        print(xdf.shape, ydf.shape)

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)

    
        dataset = TimeSeriesDataset(tensor_X, tensor_y)
        batch_size = batch 
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        model = LSTM(input_size=1, hidden_layer_size=layers, output_size=20)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)
        losses = np.zeros(epoch)
        model.to(device) 
        # Training loop
        epochs = epoch
        for epoch in range(epochs):
            for batch_X, batch_y in tqdm(dataloader):
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
        
                model.zero_grad()
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
            losses[epoch] = loss.item()

        models.append(model)

   
    X = test_data
    num_steps = (len(X[-1]) - data_portion -20) // dist  # Integer division to get the number of steps possible
    # Adjust the shapes of xdf and ydf to account for the step size of 200
    test_xdf = np.zeros((len(X), num_steps, data_portion, 1))
    test_ydf = np.zeros((len(X), num_steps, 20))
    residual = (len(X[-1]) - data_portion - 20) % dist

    i = 0
    for h in range(len(X)):
        for i in range(0, len(X[-1]) - data_portion - 20 -residual, dist):  # Step by 200
            # Calculate the index for storing the data, considering the step size
            index = i // dist
            test_xdf[h][index, :, 0] = X[h][i:i+data_portion]
            # Put numerical cluster assignment in xdf
            #xdf[h][index, :, 1] = cluster_assignments[h]
            # Put next 200 data points in ydf
            test_ydf[h][index] = X[h][i+data_portion:i+data_portion+20]


    test_xdf =  test_xdf.reshape(-1, test_xdf.shape[-2], test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])
        
    i = 0
    for i in range(len(test_xdf)):
        test_xdf[i] =((test_xdf[i] + 1) / (test_xdf[i][-1] + 1))-1

    scaler = MinMaxScaler()
    print(test_xdf.shape, test_ydf.shape)
    # scale only price data, not cluster assignments
    test_xdf[:, :, 0] = scaler.fit_transform(test_xdf[:, :, 0].reshape(-1, 1)).reshape(-1, test_xdf.shape[-2])

    print(test_xdf, test_ydf)

    tensor_test_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_test_y = torch.tensor(test_ydf, dtype=torch.float)

    test_dataset = TimeSeriesDataset(tensor_test_X, tensor_test_y)

    test_dataset = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)
    
    # test data on assigned cluster models
    overall = []
    with torch.no_grad():
        for x,y in tqdm(test_dataset):
            x = x.to(device)
            y = y.to(device)
            preds = []
            for model in models:
                model.eval()
                predictions = model(x)
                preds.append(predictions)
            overall.append(preds)

    #copy tensor to cpu and numpy
    print(predictions)

    print(test_cluster_assignments[0])
    # plot actaul and all 3 models predictions
    plt.plot(test_ydf[0].reshape(-1, 1), label='Actual')
    plt.plot(overall[0][0].cpu().numpy().reshape(-1, 1), label='Model 1')
    plt.plot(overall[0][1].cpu().numpy().reshape(-1, 1), label='Model 2')
    plt.plot(overall[0][2].cpu().numpy().reshape(-1, 1), label='Model 3')
    plt.legend()
    plt.show()
    print(test_cluster_assignments)
    for pred in overall:
        cluster_preds = []
        for i in range(len(pred)):
            cluster_preds.append(pred[i].cpu().numpy())
    print(cluster_preds)
    # calculate loss 
    loss = np.mean((cluster_preds - test_ydf)**2)
    print(f"Test Loss: {loss}")

    return overall

    



if __name__ == "__main__":
   if len(sys.argv) != 4:
       print("Usage: python single_model_test.py freq_per_second num_clusters poriton learner layers")
       sys.exit(1)
   num_clusters = int(sys.argv[1])
   data_portion = int(sys.argv[2])
   num_files = int(sys.argv[3])

   # call the function
   results = test_single_model(num_clusters, data_portion, num_files)
   results = test_cluster_model(num_clusters, data_portion, num_files)

   write_file = open("single_model_results.txt", "w")
   write_file.write(str(results) + "\n")
   write_file.close()

       
