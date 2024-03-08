
import numpy as np
import matplotlib.pyplot as plt
import glob
from get_data import get_simple_data
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import math
from sklearn.metrics import mean_squared_error
from lstm_dprep import prep_data
import skfuzzy as fuzz

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class PrivTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len=5000):
        super(PrivTransformer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len

        self.embedding = nn.LSTM(1, d_model, batch_first=True)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        """
        src: Tensor, shape [batch_size, seq_len, d_model]
        tgt: Tensor, shape [batch_size, seq_len, d_model]
        """

        # pass sequence of hiddenstates as embeddings]
        src, _ = self.embedding(src)
        tgt, _ = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

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

def test_single_model(num_clusters, data_portion, num_files, layers = 256, batch = 16, epoch = 40):
    spy_files = sorted(glob.glob("C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\SPY\\*.csv"))[-20:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

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



    test_data = result[num_files:]
    result = result[:num_files]
    price_series = result

    xdf, ydf = prep_data(price_series, data_portion, 512, dist = 512, normalize = True)

    ydf =  ydf.reshape(ydf.shape[0], ydf.shape[1], 1)
    # reshape data to have 1 features
    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(xdf, dtype=torch.float)
    tensor_y = torch.tensor(ydf, dtype=torch.float)
    
    # define dataset
    dataset = TimeSeriesDataset(tensor_X, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False)

    transofmer = PrivTransformer(d_model=512, nhead=8, num_decoder_layers=3, num_encoder_layers=3, dim_feedforward=2048)
    criterion = torch.nn.MSELoss()
    optimizer= torch.optim.Adam(transofmer.parameters(), lr=0.00001)
    epochs = epoch
    transofmer.to(device)
    for epoch in range(epoch):
        for clusters, targets in tqdm(dataloader):
            transofmer.zero_grad()
            clusters = clusters.float().to(device)
            targets = targets.float().to(device)
            outputs = transofmer(clusters, targets)
            loss = criterion(outputs, targets)
       
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Loss: {loss.item()}')
    print("done")

    # save model
    torch.save(transofmer.state_dict(), 'transformer_model.pth')
    plt.clf()
    # plot an example sequence
    plt.figure(figsize=(6, 6))
    plt.plot(targets[0].cpu().detach().numpy(), label='target')
    plt.plot(outputs[0].cpu().detach().numpy(), label='output')
    plt.show()
    plt.legend()

    # test data
    test_xdf, test_ydf = prep_data(test_data, data_portion, 512, dist = 512, normalize = True)
    test_ydf =  test_ydf.reshape(test_ydf.shape[0], test_ydf.shape[1], 1)
    # Convert your data to PyTorch tensors
    tensor_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_y = torch.tensor(test_ydf, dtype=torch.float)
    # Create the dataset
    test_dataset = TimeSeriesDataset(tensor_X, tensor_y)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    
    test_losses = []
    transofmer.eval()
    with torch.no_grad():
        idx = 0
        inter_losses = []
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            transofmer.eval()
            y_pred = transofmer(inputs, targets)
   
            test_loss = criterion(y_pred, targets)
            inter_losses.append(test_loss.item())
            idx += 1
        test_losses.append(np.mean(inter_losses))

    print(f"Test Loss: {np.mean(inter_losses)}")
    # plot test example
    plt.clf()
    plt.figure(figsize=(6, 6))
    plt.plot(targets[-1].cpu().detach().numpy(), label='target')
    plt.plot(y_pred[-1].cpu().detach().numpy(), label='output')
    plt.legend()
    plt.show()
    return np.mean(inter_losses)


# cluster model
def cluster_transformer(num_clusters, data_portion, num_files, layers = 256, batch = 16, epoch = 20, forecast = 512, look_back = 512, batch_size = 32):
    spy_files = sorted(glob.glob("C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\SPY\\*.csv"))[-20:]
    date_range = [file.split("\\")[-1].split(".")[0][4:-22] for file in spy_files]
    files = [f"C:\\Users\\Leo\\OneDrive\\Dokumente\\Files 2024\\clustering_hp\\Data\\AAPL\\AAPL_{date}_34200000_57600000_1_1.csv" for date in date_range]

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
    
    test_data = result[num_files:]
    result = result[:num_files]
    price_series = result

    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        price_series.T,  
        c=num_clusters, 
        m=2,  # Fuzziness parameter
        error=0.005,  # Stopping criterion
        maxiter=1000,  # Maximum number of iterations
        init=None  # Initialization method (None for random)
    )

    cluster_assignments = np.argmax(u, axis=0)

    for i in range(num_clusters):
        print(f"Cluster {i} has {np.sum(cluster_assignments == i)} samples")
        cluster_number = i
        # count number of points in this cluster
        count = np.count_nonzero(cluster_assignments == i)
        cluster_series = np.empty((count, len(price_series[0])))
        idx = 0
        for j in range(len(cluster_assignments)):
            if cluster_assignments[j] == i:
                cluster_series[idx] = price_series[j]
                idx += 1
        
        xdf, ydf = prep_data(cluster_series, data_portion, forecast, dist = 512)
        print(xdf.shape, ydf.shape)
        print(xdf)
        ydf =  ydf.reshape(ydf.shape[0], ydf.shape[1], 1)
        print(ydf)

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)
        # Create the dataset
        dataset = TimeSeriesDataset(tensor_X, tensor_y)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        model = PrivTransformer(d_model=512, nhead=8, num_decoder_layers=3, num_encoder_layers=3, dim_feedforward=2048)
        criterion = torch.nn.MSELoss()
        optimizer= torch.optim.Adam(model.parameters(), lr=0.00001)
        epochs = epoch
        model.to(device)
        for epoch in range(epoch):
            for clusters, targets in tqdm(dataloader):
                model.zero_grad()
                clusters = clusters.float().to(device)
                targets = targets.float().to(device)
                outputs = model(clusters, targets)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Loss: {loss.item()}')

        # save model
        torch.save(model.state_dict(), f'transformer_model_{i}.pth')


    # Testing cluster models:
    print("Testing model")
    medoids = calculate_medoids(price_series, cluster_assignments, num_clusters)
    fuzzy_memberships = calculate_fuzzy_memberships(test_data, medoids, bandwidth=20)

    test_xdf, test_ydf = prep_data(test_data, data_portion, forecast, dist = 512)
    test_ydf =  test_ydf.reshape(test_ydf.shape[0], test_ydf.shape[1], 1)

    tensor_X = torch.tensor(test_xdf, dtype=torch.float)
    tensor_y = torch.tensor(test_ydf, dtype=torch.float)
    
    print(test_xdf.shape, test_ydf.shape) 

    test_dataset = TimeSeriesDataset(tensor_X, tensor_y)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False)
    lossess = []
    model.load_state_dict(torch.load(f"transformer_model_{0}.pth"))
    with torch.no_grad():
        idx = 0
        for inputs, targets in tqdm(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            model.eval()
            y_pred = model(inputs, targets)
            test_loss = torch.sqrt(torch.mean((y_pred - targets)**2))
            lossess.append(test_loss.item())
            idx += 1
    
    cluster_models = []
    for i in range(num_clusters):
        model.load_state_dict(torch.load(f"transformer_model_{i}.pth"))
        cluster_models.append(model)
    
    cluster_results = np.zeros((len(test_ydf), 512))
    idx = 0
    d = 0
    for day in test_data:
        # get cluster membership
        cluster_membership = fuzzy_memberships[d]        # get cluster numbe
  
        # arrange day into xdf
        xdf, ydf = prep_data([day], data_portion, forecast, dist = 10)
        ydf =  ydf.reshape(ydf.shape[0], ydf.shape[1], 1)

        # Convert your data to PyTorch tensors
        tensor_X = torch.tensor(xdf, dtype=torch.float)
        tensor_y = torch.tensor(ydf, dtype=torch.float)


        # Create the dataset
        dataset = TimeSeriesDataset(tensor_X, tensor_y)

        batch_size = 1 # Define your batch size
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
        lossess_cluster = []
        with torch.no_grad():
               for inputs, targets in tqdm(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    out = []
                    for model in cluster_models:
                        model.eval()
                        model.to(device)
                        y_pred = model(inputs, targets)
                        test_loss = torch.sqrt(torch.mean((y_pred - targets)**2))
                        out.append(y_pred.cpu().detach().numpy())
                        lossess_cluster.append(test_loss.item())
                    # apply cluster membership to model outputs
                    #reshape out
                    out = np.array(out)
                    out = out.reshape(out.shape[0], out.shape[2])

                    cluster_results[idx] = np.average(out, axis=0, weights=cluster_membership)
                    idx += 1
        d += 1

    normal_loss = np.mean(lossess)
    print("Normal loss: ", normal_loss)

    cluster_loss = np.mean(lossess_cluster)
    print("Cluster loss: ", cluster_loss)


    return normal_loss, cluster_loss


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python single_model_test.py freq_per_second num_clusters poriton learner layers")
        sys.exit(1)
    
    num_clusters = int(sys.argv[1])
    data_portion = int(sys.argv[2])
    num_files = int(sys.argv[3])
    print(test_single_model(num_clusters, data_portion, num_files))
    print(cluster_transformer(num_clusters, data_portion, num_files))
    
    print("done")