import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.preprocessing import MinMaxScaler

yf.pdr_override()

# Streamlit UI for input
st.title('Stock Market Prediction System')
y_symbol = st.text_input("Enter the Stock Ticker:", 'AAPL')

# Date range for the stock data
startdate = '2010-01-01'
enddate = '2023-1-10'

# Fetching the data
data = pdr.get_data_yahoo(y_symbol, startdate, enddate)

# Displaying the data
st.subheader('Data from 2010-2023')
st.write(data.describe())

# Various plots
st.subheader('Closing Price VS Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close, 'b', label='Closing price')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price VS Time with 100 moving average')
ma100 = data.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close, 'b', label='Closing price')
plt.plot(ma100, 'r', label='ma100')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price VS Time with 100 & 200 moving average')
ma200 = data.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close, 'b', label='Closing price')
plt.plot(ma100, 'r', label='ma100')
plt.plot(ma200, 'g', label='ma200')
plt.legend()
st.pyplot(fig)

# Splitting the data into training and testing
data_training = pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing = pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

# Concatenate past_100_days and data_testing
past_100_days = data_training.tail(100)
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)

# Continue with preprocessing and model predictions as before
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Load your model
model = load_model('keras_model.h5')

# Preparing the test data
input_data = scaler.fit_transform(final_data)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Making predictions
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final graph
st.subheader('Predictions VS Original Price')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.legend()
st.pyplot(fig2)
