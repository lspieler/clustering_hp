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
       self.linear = torch.nn.Linear(hidden_layer_size, hidden_layer_size*2)
       self.linear1 = torch.nn.Linear(hidden_layer_size*2, output_size)

   def forward(self, x):
       # x.shape should be [batch_size, sequence_length, num_features]
       lstm_out, _ = self.lstm(x)
       last_time_step_out = lstm_out[:, -1, :]
       hidden = self.linear(last_time_step_out)
       output = self.linear1(hidden)
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


def test_single_model(num_clusters, data_portion, num_files, layers = 512, batch = 10, epoch = 20):

    batch_size = batch # Define your batch size
    files = sorted(glob.glob("C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\AAPL\\*.csv"))[:100]
    data_portion = int(23400 * data_portion)
    result = np.zeros((len(files),23400))

    for x in range(len(files)):
        df = get_simple_data(0, 10000000, files[x], 's')
        df = df.iloc[0:23400]
        if df.shape[0] != 23400:
            continue
        df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2

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

   # print cluster validation metrics
    print(f'Cluster center: {cntr}')
    print(f'Fuzzy partition coefficient: {fpc}')
    print(f'Error after {p} iterations: {fpc}')


    cluster_assignments = np.argmax(u, axis=0)
    print(cluster_assignments)

    # prepare data with cluster assingments
    X = price_series
    print(X)

  # get test data clustering assignments
    dist = 20
    num_steps = (len(X[-1]) - data_portion -200) // dist  # Integer division to get the number of steps possible
    # Adjust the shapes of xdf and ydf to account for the step size of 200
    xdf = np.zeros((len(X), num_steps, data_portion, 2))
    ydf = np.zeros((len(X), num_steps, 200))
    residual = (len(X[-1]) - data_portion - 200) % dist

# Create a rolling window of x based on data portion, moving by 200 points each step
    for h in range(len(X)):
        for i in range(0, len(X[-1]) - data_portion - 200 -residual, dist):  # Step by 200
            # Calculate the index for storing the data, considering the step size
            index = i // dist
            xdf[h][index, :, 0] = X[h][i:i+data_portion]
            # Put numerical cluster assignment in xdf
            xdf[h][index, :, 1] = cluster_assignments[h]
            # Put next 200 data points in ydf
            ydf[h][index] = X[h][i+data_portion:i+data_portion+200]


    xdf = xdf.reshape(-1, xdf.shape[-2], xdf.shape[-1])
    ydf = ydf.reshape(-1, ydf.shape[-1])

    for i in range(len(xdf)):
        xdf[i] =((xdf[i] + 1) / (xdf[i][-1] + 1))-1

    scaler = MinMaxScaler()
    # scale only price data, not cluster assignments
    xdf[:, :, 0] = scaler.fit_transform(xdf[:, :, 0].reshape(-1, 1)).reshape(-1, xdf.shape[-2])

    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(xdf, dtype=torch.float)
    tensor_y = torch.tensor(ydf, dtype=torch.float)

   
    dataset = TimeSeriesDataset(tensor_X, tensor_y)
    batch_size = batch 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = LSTM(input_size=2, hidden_layer_size=layers, output_size=200)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
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

    # Predict using the model
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(batch_X)


    
    #copy tensor to cpu and numpy
    predictions = predictions.cpu()
    batch_y = batch_y.cpu()
    plt.clf()
    plt.plot(predictions[0].numpy())
    plt.plot(batch_y[0].numpy())
    plt.show()
    
    ydf = np.where(ydf[:,-1] > ydf[:,0], 1, 0)
    #construct classifier dataset wehre y is whether the stock rises or falls in the next 200 points
    dataset = TimeSeriesDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = Clasifier(input_size=2, hidden_layer_size=layers, output_size=200)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    losses = np.zeros(epoch)
    model.to(device)
    # Training loop
    epochs = epoch
    for epoch in range(epochs):
        for batch_X, batch_y in tqdm(dataloader):
            print(batch_X.shape, batch_y.shape)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y.long())
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
        losses[epoch] = loss.item()

    # Predict using the model
    accuracies = []
    model.eval()  # Set the model to evaluation mode
    # get in sample accuracy
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            print(f"Test Loss: {loss.item()}")
            # compute accuracy
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == batch_y).sum().item()
            accuracy = correct / batch_size
            accuracies.append(accuracy)

    print(f"Mean accuracy: {np.mean(accuracies)}")
  
    """

    test_xdf = np.zeros((len(test_data), len(test_data[0]) - data_portion -200, data_portion, 2))
    test_ydf = np.zeros((len(test_data), len(test_data[0]) - data_portion -200, 200))

    for h in range(len(test_data)):
        for i in range(0, len(test_data[0]) - data_portion - 200, dist):  # Step by 200
            index = i // dist
            test_xdf[h][i, :, 0] = test_data[h][i:i+data_portion]
            test_xdf[h][i, :, 1] = cluster_assignments[h]
            test_ydf[h][i] = test_data[h][i+data_portion:i+data_portion+200]

    test_xdf = test_xdf.reshape(-1, test_xdf.shape[-2], test_xdf.shape[-1])
    test_ydf = test_ydf.reshape(-1, test_ydf.shape[-1])

    tensor_test_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_test_y = torch.tensor(test_ydf, dtype=torch.float)

    test_dataset = TimeSeriesDataset(tensor_test_X, tensor_test_y)
    model.cpu()
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        predictions = model(tensor_test_X)
    #copy tensor to cpu and numpy
    predictions = predictions
    batch_y = tensor_test_y
    loss = np.mean((predictions - batch_y)**2)
    print(f"Test Loss: {loss}")
    """

    return losses
    



if __name__ == "__main__":
   if len(sys.argv) != 4:
       print("Usage: python single_model_test.py freq_per_second num_clusters poriton learner layers")
       sys.exit(1)
   num_clusters = int(sys.argv[1])
   data_portion = float(sys.argv[2])
   num_files = int(sys.argv[3])

   # call the function
   results = test_single_model(num_clusters, data_portion, num_files)


   write_file = open("single_model_results.txt", "w")
   write_file.write(str(results) + "\n")
   write_file.close()

       
