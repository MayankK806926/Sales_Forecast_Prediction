# Sales Forecast Prediction

This project implements a sales forecasting model using XGBoost to predict future sales based on historical data.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)

## Project Overview

Sales forecasting is a critical component of business planning that helps organizations predict future sales and make informed decisions about inventory management, marketing strategies, and resource allocation. This project demonstrates how to build a sales forecast prediction model using Python and XGBoost.

Key features:
- Time series data preprocessing
- Feature engineering with lagged features
- XGBoost regression model
- Model evaluation with RMSE
- Visualization of results

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-forecast-prediction.git
   cd sales-forecast-prediction
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset and place it in the data/ directory.

## Usage
Run the main script:
  ```bash
  python main.py
  ```
Alternatively, you can explore the Jupyter notebook:
  ```bash
  jupyter notebook notebooks/sales_forecasting.ipynb
  ```
## Project Structure
  ```
sales-forecast-prediction/
│
├── data/               # Contains the dataset
├── models/             # Saved models
├── notebooks/          # Jupyter notebooks
├── src/                # Source code
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── main.py             # Main executable script
  ```
## Results
The model achieves an RMSE of 734.63 on the test set, indicating good predictive performance for sales forecasting.
Sample visualizations:
- Sales trend over time
- Actual vs predicted sales comparison
