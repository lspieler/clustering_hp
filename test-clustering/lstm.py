import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


def run_lstm(X, y, test_price, data_portion):
    print(X.shape)
    print(y.shape)
    X_train = X[:-1]
    y_train = y[:-1]
    scaler = MinMaxScaler()
    model = Sequential()
    scaled_data = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X[-1].reshape(1, -1))
    scaled_test = scaled_test.reshape(1, 14040, 1)
    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
    
    
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=150, return_sequences=True))
    model.add(LSTM(units=150))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 3. Training the LSTM
    model.fit(scaled_data, y_train, epochs=150, batch_size=50)

    # 4. Making Predictionscç
    # To predict the price for the next day using the last 250 days:
    """
    test = test_price.iloc[0:data_portion].values
    test_flattened = test.flatten().reshape(-1, 1)
    scaled_test = scaler.transform(test_flattened)
    scaled_test = scaled_test.reshape(1, 11700, 1)
    """
    predicted_price = model.predict(scaled_data) # De-normalize
    print(f'Acutal Price for Next Day: {y[:-1]}')
    print(f"Predicted Price for Next Day: {predicted_price}")

    # test on test data

    predicted_price = model.predict(scaled_test) # De-normalize
    print(f'Acutal Price for Next Day: {y[-1]}')
    print(f"Predicted Price for Next Day: {predicted_price}")
 
