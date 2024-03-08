import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import multiprocessing as mp
from get_data import get_simple_data
import sys
from lstm import run_lstm, cluster_lstm
import multiprocessing as mp
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import skfuzzy as fuzz
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
from lstm_dprep import prep_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = torch.nn.LSTM(input_size, hidden_layer_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(hidden_layer_size, output_size, bias=True)
        
    def forward(self, input_seq):
        x, _ = self.lstm(input_seq)
        x = x[:, -1, :]
        x = self.linear(x)
  
        return x


def test_single_model(num_clusters, num_files, layers = 2, batch = 10, epochs = 20, forecast = 20, lookback = 200):

    batch_size = batch # Define your batch size
    files = sorted(glob.glob(r'C:\Users\Leo\OneDrive\Dokumente\Files 2024\clustering_hp\Data\AAPL\*.csv'))[:120]
    data_portion = lookback
    price = np.zeros((len(files),23400))
    obi = np.zeros((len(files),23400))


    print(data_portion)
    for x in range(len(files)):
        df = get_simple_data(0, 10000000, files[x], 's')
        df = df.iloc[0:23400]
        if df.shape[0] != 23400:
            continue
        df["price"] = (df["ask_1"] + df["bid_1"]).ffill().bfill()/2
        df["obi"] = df[" bid_size_1"] / df["ask_size_1"].ffill().bfill()
        price[x] = ((df["price"]/df['price'].iloc[0]) -1) * 10
        obi[x] = np.log10(df["obi"].values +1)

    test_data = price[num_files:]
    train_data = price[:num_files]

    obi = obi.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    obi = scaler.fit_transform(obi)
    obi = obi.reshape(-1, 23400)
 
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        train_data.T,  
        c=num_clusters, 
        m=2,  # Fuzziness parameter
        error=0.005,  # Stopping criterion
        maxiter=1000,  # Maximum number of iterations
        init=None  # Initialization method (None for random)
    )

    test_data = obi[num_files:]
    train_data = obi[:num_files]
    
    xdf,ydf = prep_data(train_data, data_portion, forecast, dist = 1, normalize = False)

    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(xdf, dtype=torch.float)
    tensor_y = torch.tensor(ydf, dtype=torch.float)
    # Create the dataset
    dataset = TimeSeriesDataset(tensor_X, tensor_y)

    test_xdf, test_ydf = prep_data(test_data, data_portion, forecast, dist = 1, normalize=False)

    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_y = torch.tensor(test_ydf, dtype=torch.float)

    # Create the dataset
    test_dataset = TimeSeriesDataset(tensor_X, tensor_y)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)

    print(xdf.shape, test_xdf.shape)


    batch_size = batch 
    losses = []
    test_losses = []
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = LSTM(input_size =1, hidden_layer_size=layers, output_size=forecast).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        inter_losses = []
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, targets)
            loss.backward()
            optimizer.step()
            inter_losses.append(loss.item())
        # Print loss for every epoch
        print(f'Epoch: {epoch+1}, Loss: {np.mean(inter_losses)}')
        losses.append(np.mean(inter_losses))
        # get test lost for model
        model.eval()
        with torch.no_grad():
            idx = 0
            inter_losses = []
            for inputs, targets in test_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                model.eval()
                y_pred = model(inputs)
                test_loss = criterion(y_pred, targets)
                inter_losses.append(test_loss.item())
                idx += 1
            test_losses.append(np.mean(inter_losses))
            print(f"Test Loss: {np.mean(inter_losses)}")
        torch.save(model.state_dict(), f"models/lstm.pth")
        model.train()

    # plot loss
    plt.plot(losses, label = "Training Loss")
    plt.plot(test_losses, label = "Test Loss")
    plt.legend()
    plt.show()
    plt.clf()

    # plot predictions over test dtaa
    predictions = []
    model.eval()
    for X, y in test_dataloader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y_pred = model(X)
            predictions.append(y_pred)
    predictions = torch.cat(predictions, axis=0)
    predictions = predictions.cpu().detach().numpy()
    
    print(predictions.shape, test_ydf.shape)
    # plot predictions
    plt.plot(predictions, label = "Predicted",alpha = 0.5)
    plt.plot(test_ydf, label = "True", alpha = 0.5)
    plt.legend()
    plt.show()

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python single_model_test.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    num_clusters = int(sys.argv[1])
    data_portion = int(sys.argv[2])
    num_files = int(sys.argv[3])
    forward = int(sys.argv[4])
    
 
    test_single_model(num_clusters, num_files, layers =32, batch = 10, epochs = 20, forecast = forward, lookback = data_portion)
    print("done")
