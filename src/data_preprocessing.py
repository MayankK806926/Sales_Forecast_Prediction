import pandas as pd

def load_and_preprocess_data(file_path):
    """Load and preprocess the sales data."""
    data = pd.read_csv(file_path)
    data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')
    return data

def create_lagged_features(data, lag=1):
    """Create lagged features for time series forecasting."""
    lagged_data = data.copy()
    for i in range(1, lag+1):
        lagged_data[f'lag_{i}'] = lagged_data['Sales'].shift(i)
    return lagged_data