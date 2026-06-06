"""
EV Charging Demand Forecasting - LSTM Model (Stretch Goal)
------------------------------------------------------------
This script is designed for Google Colab (with free T4 GPU acceleration).
To run:
1. Upload the generated 'data/processed_demand_data.csv' file to Google Colab.
2. In Colab, change runtime type to GPU (Runtime -> Change runtime type -> T4 GPU).
3. Install dependencies: pip install torch scikit-learn pandas numpy matplotlib
4. Run this script: python lstm_colab.py
"""

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch Imports
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Load Data
def load_and_preprocess_data(csv_path):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort data chronologically per zone
    df = df.sort_values(by=["city", "zone", "timestamp"]).reset_index(drop=True)
    
    # Identify target and features
    target_col = 'ev_demand_kw'
    
    # Select important numerical columns for LSTM
    feature_cols = [
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_peak_hour', 'is_holiday',
        'sin_hour', 'cos_hour', 'sin_month', 'cos_month',
        'temperature', 'humidity', 'wind_speed', 'precipitation',
        'charging_stations',
        'demand_lag_1h', 'demand_lag_2h', 'demand_lag_3h'
    ]
    
    # Keep city/zone column for filtering
    meta_cols = ['timestamp', 'city', 'zone']
    
    return df, feature_cols, target_col, meta_cols

# 2. Build Sliding Sequence Windows
def create_sequences(df, feature_cols, target_col, seq_length=24):
    """Creates sequence datasets per zone to prevent data mixing."""
    print("Creating sequence windows of length", seq_length, "...")
    X_seqs = []
    y_seqs = []
    
    # Group by city/zone to generate sequences independently
    grouped = df.groupby(['city', 'zone'])
    
    # Fit scalers globally on the features and target
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # Scale columns
    scaled_features = scaler_x.fit_transform(df[feature_cols].values)
    scaled_targets = scaler_y.fit_transform(df[[target_col]].values)
    
    # Re-insert scaled data into df copies for windowing
    df_scaled = df.copy()
    for idx, col in enumerate(feature_cols):
        df_scaled[col] = scaled_features[:, idx]
    df_scaled[target_col] = scaled_targets.flatten()
    
    # Generate windows
    for name, group in df_scaled.groupby(['city', 'zone']):
        group_x = group[feature_cols].values
        group_y = group[target_col].values
        
        for i in range(len(group) - seq_length):
            X_seqs.append(group_x[i : i + seq_length])
            y_seqs.append(group_y[i + seq_length])
            
    return np.array(X_seqs), np.array(y_seqs), scaler_x, scaler_y

# 3. Define LSTM Network Architecture
class EVDemandLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(EVDemandLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0.0)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initializing hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # Forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

# 4. MAPE Calculation
def calculate_mape(y_true, y_pred):
    mask = y_true >= 1.0
    if not np.any(mask):
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    csv_path = "processed_demand_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found in current directory. Please upload it first.")
        return
        
    df, feature_cols, target_col, meta_cols = load_and_preprocess_data(csv_path)
    
    # 80/20 train/test split globally (time-series aware)
    unique_timestamps = sorted(df['timestamp'].unique())
    split_idx = int(len(unique_timestamps) * 0.8)
    split_time = unique_timestamps[split_idx]
    
    train_df = df[df['timestamp'] < split_time].copy()
    test_df = df[df['timestamp'] >= split_time].copy()
    
    seq_length = 24  # 24-hour sequence history
    
    # Create sequence datasets
    X_train, y_train, scaler_x, scaler_y = create_sequences(train_df, feature_cols, target_col, seq_length)
    # Use scaler fitted on training set for test set
    scaled_test_features = scaler_x.transform(test_df[feature_cols].values)
    scaled_test_targets = scaler_y.transform(test_df[[target_col]].values)
    
    test_df_scaled = test_df.copy()
    for idx, col in enumerate(feature_cols):
        test_df_scaled[col] = scaled_test_features[:, idx]
    test_df_scaled[target_col] = scaled_test_targets.flatten()
    
    X_test, y_test = [], []
    for name, group in test_df_scaled.groupby(['city', 'zone']):
        group_x = group[feature_cols].values
        group_y = group[target_col].values
        for i in range(len(group) - seq_length):
            X_test.append(group_x[i : i + seq_length])
            y_test.append(group_y[i + seq_length])
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    print(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Create DataLoaders
    batch_size = 128
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Model Hyperparameters
    input_dim = len(feature_cols)
    hidden_dim = 64
    num_layers = 2
    output_dim = 1
    learning_rate = 0.001
    num_epochs = 15
    
    # Instantiate Model, Loss, Optimizer
    model = EVDemandLSTM(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n--- Training LSTM ---")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}")
        
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # 5. Evaluate on Test Set
    model.eval()
    with torch.no_grad():
        X_test_gpu = X_test_t.to(device)
        predictions_scaled = model(X_test_gpu).cpu().numpy()
        
    # Inverse transform predictions and targets
    y_pred = scaler_y.inverse_transform(predictions_scaled).flatten()
    y_true = scaler_y.inverse_transform(y_test).flatten()
    
    # Clip negative values
    y_pred = np.clip(y_pred, 0, None)
    
    # Compute Metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n--- LSTM Model Evaluation Results ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kW")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")
    
    # Plot first 168 hours (1 week) of a sample zone prediction vs actual
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:168], label='Actual Demand', color='tab:blue', alpha=0.8)
    plt.plot(y_pred[:168], label='LSTM Forecast', color='tab:orange', linestyle='--', alpha=0.9)
    plt.title('EV Charging Demand Forecast (1-Week Out-of-Sample Sample)')
    plt.xlabel('Hours')
    plt.ylabel('Demand (kW)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lstm_sample_prediction.png', dpi=300)
    print("Saved sample prediction plot as 'lstm_sample_prediction.png'")
    
if __name__ == "__main__":
    main()
