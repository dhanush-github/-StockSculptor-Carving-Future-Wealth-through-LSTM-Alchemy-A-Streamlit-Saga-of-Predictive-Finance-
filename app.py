import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('Stock Predictions Model.keras')

# stock price prediction
st.header('Stock Price Predictor')

#fetch data from 2012 to 2023
stock =st.text_input('Enter Stock Symnbol', 'GOOG')
start = '2012-01-01'
end = '2023-12-31'

#Crawling data from yahoo finance website
data = yf.download(stock, start ,end)

st.subheader('Stock Data')
st.write(data)

#train - test Split
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

#Normalizing data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#fetching the last 100 data points from tran data --> as we need it for testing purpose. so concatenating both.
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

#plot for closing price Vs 50 days Moving average
st.subheader('Closing Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig1)

#plotting for 50 days Moving average Vs 100 days Moving average
st.subheader('Closing Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig2)

#plotting for 100 days Moving average Vs 200 days Moving average
st.subheader('Closing Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.legend()
plt.show()
st.pyplot(fig3)



x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

#converting into numpy arrays
x,y = np.array(x), np.array(y)

#predicting test data using trained model
predict = model.predict(x)

#fetching the scaling factor with the aid of built in function
scale = 1/scaler.scale_

#scaling back to normal range for comparision
predict = predict * scale
y = y * scale
#Final plot for Closing Price Vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict, 'r', label='Original Price')
plt.plot(y, 'g', label = 'Predicted Price')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)






