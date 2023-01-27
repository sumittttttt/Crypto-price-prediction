import pandas as pd
import numpy as np
import tensorflow as tf
import random as rn
import yfinance as yf
import streamlit as st
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from millify import millify
import pandas_datareader as web
from cryptocmd import CmcScraper

st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

csv = pd.read_csv('convertcsv.csv')
symbol = csv['symbol'].tolist()

# creating sidebar
ticker_input = st.selectbox('Enter or Choose Crypto Coin', symbol,index=symbol.index('ETH'))

start = dt.datetime.today() - dt.timedelta(5*365)
end = dt.datetime.today()

a = start.strftime('%d-%m-%Y')
b = end.strftime('%d-%m-%Y')

# initialise scraper with time interval for e.g a year from today
scraper = CmcScraper(ticker_input, a, b)
# Pandas dataFrame for the same data
df = scraper.get_dataframe()

st.write('It will take some seconds to fit the model....')
eth_df = df.sort_values(['Date'],ascending=True, axis=0)


#creating dataframe
eth_lstm = pd.DataFrame(index=range(0,len(eth_df)),columns=['Date', 'Close'])
for i in range(0,len(eth_df)):
    eth_lstm['Date'][i] = eth_df['Date'][i]
    eth_lstm['Close'][i] = eth_df['Close'][i]

#setting index
eth_lstm.index = eth_lstm.Date
eth_lstm.drop('Date', axis=1, inplace=True)
eth_lstm = eth_lstm.sort_index(ascending=True)


#creating train and test sets
dataset = eth_lstm.values
train = dataset[0:990,:]
valid = dataset[990:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print('Fitting Model')
#predicting 246 values, using past 60 from the train data
inputs = eth_lstm[len(eth_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(mean_squared_error(closing_price,valid))
acc = r2_score(closing_price,valid)*100

# for plotting
train = eth_df[:990]
valid = eth_df[990:]
valid['Predictions'] = closing_price

st.write('#### Actual VS Predicted Prices')

fig_preds = go.Figure()
fig_preds.add_trace(
    go.Scatter(
        x=train['Date'],
        y=train['Close'],
        name='Training data Closing price'
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Close'],
        name='Validation data Closing price'
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Predictions'],
        name='Predicted Closing price'
    )
)

fig_preds.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Predictions on Validation Data', template='gridon'
)

st.plotly_chart(fig_preds, use_container_width=True)

# metrics
mae = mean_absolute_error(closing_price, valid['Close'])
rmse = np.sqrt(mean_squared_error(closing_price, valid['Close']))
accuracy = r2_score(closing_price, valid['Close']) * 100

# with st.container():
# st.write('#### Metrics')
# col_11, col_22, col_33 = st.columns(3)
# col_11.metric('Absolute error between predicted and actual value', round(mae,2))
# col_22.metric('Root mean squared error between predicted and actual value', round(rmse,2))

# forecasting
real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
st.write('#### Next-Day Forecasting')

with st.container():
    col_111, col_222, col_333 = st.columns(3)
    col_111.metric(f'Closing Price Prediction of the next trading day for {symbol} is',
                   f' $ {str(round(float(prediction), 2))}')
    col_222.metric('Accuracy of the model is', f'{str(round(float(accuracy), 2))} %')


