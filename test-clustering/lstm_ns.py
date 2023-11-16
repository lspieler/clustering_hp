import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def run_lstm(X, y, test_price, data_portion, layer1 = 50, layer2=30, batch = 100, epoch = 150):
    plt.plot(X[0])
    logger = mp.get_logger()  # Get the logger set up for multiprocessing
    logger.debug(f'Initing model')
    #strategy = tf.distribute.MirroredStrategy()

    scaler = MinMaxScaler()
    model = Sequential()

    # find max and min of all series in X
    
    # maybe try without transfrom?
    
    #plt.plot(scaled_data[0])

    #plt.show()
    #split into x and y data
    y = X[:,data_portion:]
    X = X[:,:data_portion]
   
    # issues: work wiht scaler
    scaled_test = test_price.reshape(1, -1)

    test_x = scaled_test[0][:data_portion]
    test_y = scaled_test[0][data_portion:]
    test_y = test_y.reshape(1, test_y.shape[0])
    test_x = test_x.reshape(1, X.shape[1], 1)
    
    model.add(LSTM(units=layer1, return_sequences=False, input_shape=(X.shape[1], 1)))

    model.add(Dense(units=y.shape[1]))

    model.compile(optimizer='adam', loss='mean_squared_error')

    logger.debug(f'Fitting model')
    model.fit(X, y, epochs=epoch, batch_size=batch, verbose=1)
    logger.debug(f'Finished fitting model')

    #test
    predicted_price = model.predict(test_x)

    # combine test_x and predicted price
    test_x = test_x.reshape(test_x.shape[1])
    predicted_price = np.concatenate((test_x, predicted_price[0]), axis=0).reshape(1, -1)
    
    #invert scaling
    

    print("inference time")
    print(f"Predicted Price for Next Day: {predicted_price}")
    print(test_price.shape)
    print(predicted_price.shape)
    plt.plot(test_price)
    plt.plot(predicted_price[0])
    plt.show()

    return(predicted_price[0][-1], test_price[data_portion], test_price[-1])

