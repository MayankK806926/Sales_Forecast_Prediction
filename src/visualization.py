import matplotlib.pyplot as plt

def plot_sales_trend(data):
    """Plot the sales trend over time."""
    sales_by_date = data.groupby('Order Date')['Sales'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(sales_by_date['Order Date'], sales_by_date['Sales'], 
             label='Sales', color='red')
    plt.title('Sales Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, predictions):
    """Plot actual vs predicted sales."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Sales', color='red')
    plt.plot(y_test.index, predictions, label='Predicted Sales', color='green')
    plt.title('Sales Forecasting using XGBoost')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()