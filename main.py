import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib
import os
from src.data_preprocessing import load_and_preprocess_data, create_lagged_features
from src.model_training import train_xgboost_model
from src.visualization import plot_sales_trend, plot_predictions

def main():
    # Configuration
    data_path = 'data/train.csv'
    lag = 5
    test_size = 0.2
    model_params = {
        'objective': 'reg:squarederror',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    }
    model_save_path = 'models/xgboost_model.pkl'
    
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data(data_path)
    
    # Visualize sales trend
    plot_sales_trend(data)
    
    # Create lagged features
    sales_with_lags = create_lagged_features(data[['Order Date', 'Sales']], lag)
    sales_with_lags = sales_with_lags.dropna()
    
    # Prepare data for training
    X = sales_with_lags.drop(columns=['Order Date', 'Sales'])
    y = sales_with_lags['Sales']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    
    # Train model
    print("Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, model_params)
    
    # Save model
    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Make predictions
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"RMSE: {rmse:.2f}")
    
    # Visualize predictions
    plot_predictions(y_test, predictions)
    
    print("Sales forecasting completed successfully!")

if __name__ == "__main__":
    main()