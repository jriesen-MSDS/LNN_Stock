import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    # Simple Moving Averages
    df['SMA_10'] = df['Close'].rolling(window=3).mean()
    df['SMA_50'] = df['Close'].rolling(window=6).mean()

    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    positive_flow = (money_flow.where(typical_price > typical_price.shift(), 0)).rolling(window=6).sum()
    negative_flow = (money_flow.where(typical_price < typical_price.shift(), 0)).rolling(window=6).sum()
    df['MFI'] = 100 - (100 / (1 + (positive_flow / negative_flow)))
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['Bollinger_Upper'] = df['Close'].rolling(20).mean() + 2 * df['Close'].rolling(20).std()
    df['Bollinger_Lower'] = df['Close'].rolling(20).mean() - 2 * df['Close'].rolling(20).std()

    # Fill NaN values with 0
    df.fillna(0, inplace=True)
    return df


# Fetch and preprocess stock data
def get_clean_financial_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = calculate_technical_indicators(data)
    data['Returns'] = data['Close'].pct_change().fillna(0)
    return data
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class LNNStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LNNStep, self).__init__()
        self.hidden_size = hidden_size
        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.g = nn.Linear(input_size + hidden_size, hidden_size)
        self.h = nn.Linear(input_size + hidden_size, hidden_size)
        self.activation = torch.sigmoid

    def forward(self, x, h):
        # Ensure tensors are on the same device
        combined = torch.cat([x, h], dim=-1).to(x.device)
        f_out = self.f(combined)
        g_out = self.g(combined)
        h_out = self.h(combined)
        new_h = self.activation(-f_out) * g_out + (1 - self.activation(-f_out)) * h_out
        return new_h


class LNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length):
        super(LNN, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lnn_step = LNNStep(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, return_hidden=False):
        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)  # Move h to the same device as x
        hidden_states = []

        for t in range(self.seq_length):
            h = self.lnn_step(x[:, t, :], h)
            if return_hidden:
                hidden_states.append(h.clone().detach().cpu().numpy())  # Store hidden states for analysis

        out = self.fc(h)
        if return_hidden:
            return out, hidden_states
        return out


class StackedLNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, num_layers=2):
        super(StackedLNN, self).__init__()
        self.lnn_layers = nn.ModuleList([LNNStep(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.zeros(x.size(0), hidden_size).to(x.device)
        for layer in self.lnn_layers:
            for t in range(x.size(1)):
                h = layer(x[:, t, :], h)
        return self.fc(h)


# Load and preprocess the data
data = get_clean_financial_data('DIA', '2020-01-01', '2024-11-20')

# Create input sequences and labels
def create_sequences(data, seq_length):
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = 1 if data[i + seq_length, 3] > data[i + seq_length - 1, 3] else 0
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# Parameters
seq_length = 20
features = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10',
                 'Bollinger_Lower', 'Bollinger_Upper']]
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)
sequences, labels = create_sequences(scaled_features, seq_length)

# Sequential split
split_index = int(len(sequences) * 0.8)
X_train, X_test = sequences[:split_index], sequences[split_index:]
y_train, y_test = labels[:split_index], labels[split_index:]


# PyTorch Dataset
class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the LNN model
input_size = 13
hidden_size = 200
output_size = 2
model = LNN(input_size, hidden_size, output_size, seq_length)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
# Define the device
# Define the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Ensure model is on the correct device
model = model.to(device)

# Training loop
epochs = 100 # Reduced for testing
# Training loop
for epoch in range(epochs):
    total_loss = 0
    model.train()
    for x, y in train_loader:
        # Move data to the same device as the model
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)  # Move test data to the correct device
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Predict the next day
last_sequence = data.iloc[-seq_length:]
last_features = last_sequence[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10', 'Bollinger_Lower', 'Bollinger_Upper']]
scaled_last_features = scaler.transform(last_features)
input_tensor = torch.tensor(scaled_last_features, dtype=torch.float32).unsqueeze(0).to(device)  # Move to device

# Predict the next day
model.eval()
with torch.no_grad():
    # Pass return_hidden=True to get hidden states
    output, hidden_states = model(input_tensor, return_hidden=True)
    prediction = torch.argmax(output, dim=1).item()

if prediction == 1:
    print("The model predicts that the next day is a BUY day.")
else:
    print("The model predicts that the next day is a SELL day.")

# Convert hidden states to a NumPy array for plotting
hidden_states = np.array(hidden_states)  # Shape: (seq_length, batch_size, hidden_size)
hidden_states = hidden_states[:, :]  # Take the batch dimension (for batch size = 1)
# Debugging device consistency
print("Model device:", next(model.parameters()).device)  # Should match input_tensor.device
print("Input tensor device:", input_tensor.device)


# Forward pass
input_tensor.requires_grad = True
# Move input tensor to the selected device
input_tensor = input_tensor.to(device)
# Debug: Check device of the input tensor
print("Input tensor device:", input_tensor.device)

output = model(input_tensor)
predicted_class = torch.argmax(output, dim=1).item()

# Compute gradient of the predicted class score w.r.t. input
output[0, predicted_class].backward()  # Backpropagate for the predicted class
feature_importances = input_tensor.grad.abs().mean(dim=1).squeeze().cpu().numpy()  # Aggregate gradient magnitudes

# Select the correct number of rows for scaled_last_features
last_sequence = data.iloc[-seq_length:]  # Select the last 20 rows
last_features = last_sequence[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10', 'Bollinger_Lower', 'Bollinger_Upper']]
scaled_last_features = scaler.transform(last_features)  # Shape: [20, 13]

# Convert to tensor for model input
#input_tensor = torch.tensor(scaled_last_features.T, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 10, 10)
print("input_tensor.shape:", input_tensor.shape)  # Ensure shape is (1, seq_length, input_size)

# Ensure input tensor for SHAP has the correct shape
# Generate SHAP values
explainer = shap.DeepExplainer(model, input_tensor)
shap_values = explainer.shap_values(input_tensor)

# Debugging: Print shapes
print("Shape of shap_values:", np.array(shap_values).shape)  # Should be (1, 30, 13, 2)
print("Shape of scaled_last_features:", scaled_last_features.shape)  # Should be (30, 13)
print("Model device:", next(model.parameters()).device)  # Should match input_tensor.device
print("Input tensor device:", input_tensor.device)  # Ensure this matches model.device

# Select SHAP values for a single class (e.g., "SELL")
shap_values_single_class = shap_values[0, :, :, 0]  # Use class 0 (SELL) SHAP values

# Ensure shapes match
print("input_tensor.shape:", input_tensor.shape)  # [1, 20, 13]
print("scaled_last_features.shape:", scaled_last_features.shape)  # [20, 13]
print("shap_values_single_class.shape:", shap_values_single_class.shape)  # [20, 13]

# SHAP visualization
shap.summary_plot(
    shap_values_single_class,
    features=scaled_last_features,
    feature_names=['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10', 'Bollinger_Lower', 'Bollinger_Upper']
)
# Check devices of key tensors
print("Model device:", next(model.parameters()).device)
print("Input tensor device:", input_tensor.device)
# Convert hidden states to a NumPy array for plotting
hidden_states = np.array(hidden_states)  # Shape: (seq_length, batch_size, hidden_size)
hidden_states = hidden_states[:, 0, :]  # Take the batch dimension (for batch size = 1)
# Convert input features to NumPy array
input_features = scaled_last_features  # Shape: (seq_length, input_size)
# Compute correlation for each hidden unit with each feature
correlations = []
for i in range(hidden_states.shape[1]):  # Iterate over hidden units
    hidden_unit_correlations = []
    for j in range(input_features.shape[1]):  # Iterate over input features
        corr = np.corrcoef(hidden_states[:, i], input_features[:, j])[0, 1]
        hidden_unit_correlations.append(corr)
    correlations.append(hidden_unit_correlations)
# Convert to a DataFrame for better readability
correlation_df = pd.DataFrame(
    correlations,
    columns=['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10', 'Bollinger_Lower', 'Bollinger_Upper'],
    index=[f'Hidden Unit {i + 1}' for i in range(hidden_states.shape[1])]
)

# Display the correlation table
print(correlation_df)

# Plot feature importances
plt.figure(figsize=(12, 6))
plt.bar(
    ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_10', 'SMA_50', 'RSI', 'MFI', 'EMA_10', 'Bollinger_Lower', 'Bollinger_Upper'],
    feature_importances
)
plt.title("Feature Importance Based on Gradients")
plt.ylabel("Importance")
plt.xlabel("Feature")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(48, 24))
sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Hidden Units and Input Features")
plt.show()
# Create the plot
fig, ax1 = plt.subplots(figsize=(24, 12))

# Plot the hidden states on the left axis
for i in range(5):  # Plot the first 5 hidden units
    ax1.plot(hidden_states[:, i], label=f"Hidden Unit {i + 1}")
ax1.set_xlabel("Timestep")
ax1.set_ylabel("Hidden State Value")
ax1.legend(loc="upper left")
ax1.set_title("Hidden States and Close Price")

# Create a second y-axis for the Close Price
ax2 = ax1.twinx()
ax2.plot(last_sequence['Close'].values, label="Close Price", linestyle="--", color="black", alpha=0.7)
ax2.set_ylabel("Close Price")
ax2.legend(loc="upper right")

# Show the plot
plt.show()