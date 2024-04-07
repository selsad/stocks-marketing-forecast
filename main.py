# Importing necessary l
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# Load your dataset (assuming you have a CSV file named 'stock_data.csv' with 'Date' and 'Close' columns)
data = pd.read_csv('Google_test_data.csv')

# Preprocessing the data (assuming 'Date' is in datetime format and 'Close' is the closing price)
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


# Splitting data into training and testing sets (80% training, 20% testing)
train_size = int(len(data) * 0.8)
train_data, test_data = data.iloc[:train_size], data.iloc[train_size:]

# Fit the SARIMAX model
model = SARIMAX(train_data['Close'], order=(5,1,0))  # Adjust the order as needed
sarimax_model = model.fit()

# Make predictions
predictions = sarimax_model.predict(start=len(train_data), end=len(train_data)+len(test_data)-1, typ='levels')

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data['Close'], label='Train')
plt.plot(test_data.index, test_data['Close'], label='Test')
plt.plot(test_data.index, predictions, label='Predictions', color='red')
plt.title('Stock Market Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
