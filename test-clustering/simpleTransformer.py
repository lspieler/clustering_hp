import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertModel, TimeSeriesTransformerConfig, TimeSeriesTransformerModel
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Sample data (replace with your stock data)
prices = np.random.rand(100, 48)  # 100 days, 48 time steps per day (e.g., 30-minute intervals)

# Normalize data
scaler = MinMaxScaler()
prices = scaler.fit_transform(prices)

# Split into first half and second half of the day
X = prices[:, :24]
y = prices[:, 24:]

X = np.expand_dims(X, axis=-1)
y = np.expand_dims(y, axis=-1)
# Create DataLoader
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define model
config = BertConfig(
    vocab_size=1,  # we're not doing NLP
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=256,
)
transformer = BertModel(config)

class TimeSeriesPredictor(nn.Module):
    def __init__(self):
        super(TimeSeriesPredictor, self).__init__()
        self.transformer = transformer
        self.decoder = nn.Linear(64, 24)  # predict 24 time steps

    def forward(self, x):
        output = self.transformer(inputs_embeds=x)
        return self.decoder(output.last_hidden_state)

model = TimeSeriesPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 1000
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")



test_X = torch.tensor(X[-10:], dtype=torch.float32)
test_y = torch.tensor(y[-10:], dtype=torch.float32)

# Predict using the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(test_X)

# Convert tensors to numpy for plotting
test_y = test_y.numpy()
predictions = predictions.numpy()

# Plot actual vs predicted values
for i in range(len(test_X)):
    plt.figure(figsize=(12, 6))
    plt.plot(test_y[i], label='Actual')
    plt.plot(predictions[i], label='Predicted', linestyle='dashed')
    plt.legend()
    plt.title(f"Sample {i+1}")
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.show()