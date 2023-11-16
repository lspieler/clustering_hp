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
   
    logger = mp.get_logger()  # Get the logger set up for multiprocessing
    logger.debug(f'Initing model')
    #strategy = tf.distribute.MirroredStrategy()

    scaler = MinMaxScaler()
    model = Sequential()

    # find max and min of all series in X

    scaled_data = scaler.fit_transform(X)
    
    # maybe try without transfrom?
    
    #plt.plot(scaled_data[0])

    #plt.show()
    #split into x and y data
    y = scaled_data[:,data_portion:]

    X = scaled_data[:,:data_portion]
   
    # issues: work wiht scaler
    scaled_test = scaler.transform(test_price.reshape(1, -1))

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
    predicted_price = scaler.inverse_transform(predicted_price)

    print("inference time")
    print(f"Predicted Price for Next Day: {predicted_price}")
    print(test_price.shape)
    print(predicted_price.shape)
    plt.plot(test_price)
    plt.plot(predicted_price[0])
    plt.show()

    return(predicted_price[0][-1], test_price[data_portion], test_price[-1])



    """
    Create a new lstm for each cluster and train on each cluster
    """
def cluster_lstm(clusters, test_cluster, test_price, data_portion, layer1 = 50, batch = 100, epoch = 150):
    print(test_price)
    models = []
    for x in range(len(clusters)):


        cluster_series = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_series[y] = clusters[x][y][0:data_portion]

        
        cluster_y = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_y[y] = clusters[x][y][data_portion:]

        print(cluster_series)
        print(cluster_y)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cluster_series)
        scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
        print(scaled_data.shape)
        #scaled_test = scaler.transform(test_price[:data_portion].values.reshape(1, -1))
        #scaled_test = test_price.reshape(1, 11700, 1)
    

        model = Sequential()
        model.add(LSTM(units=layer1,return_sequences=False, input_shape=(scaled_data.shape[1], 1)))
        model.add(Dense(units=cluster_y.shape[1]))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(scaled_data, cluster_y, epochs=epoch, batch_size=batch)

        # if model rmse is too high train 10 more epochs

    ttest = scaler.transform(test_price[0:data_portion].reshape(1, -1))
    # test performance of each model on test series
    test_predictions = []
    for x in range(len(models)):
        test_predictions.append(models[x].predict(ttest.reshape(1, ttest.shape[1], 1)))

    print(test_cluster+1)
    for x in range(len(test_predictions)):
        print(x+1, test_predictions[x], test_price[-1], mean_squared_error([test_price[-1]], test_predictions[x]))
    
    print(test_predictions[test_cluster][0][-1], test_price[-1], test_price[0])
    
    return(test_predictions[test_cluster][0][-1], test_price[-1], test_price[0])