import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def run_lstm(X, y, test_price, data_portion, layer1 = 50, layer2=50, batch = 100, epoch = 150):
    
    #strategy = tf.distribute.MirroredStrategy()

    X_train = X[:-1]
    y_train = y[:-1]
    scaler = MinMaxScaler()
    model = Sequential()
    scaled_data = scaler.fit_transform(X_train)
    scaled_test = scaler.transform(X[-1].reshape(1, -1))
    scaled_test = scaled_test.reshape(1, X.shape[1], 1)
    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
    
    
    model.add(LSTM(units=layer1, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    # 3. Training the LSTM


    model.fit(scaled_data, y_train, epochs=epoch, batch_size=batch)

        

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

    return(predicted_price[0][0], y[-1])
    #train lstm network on each cluster

    """
    Create a new lstm for each cluster and train on each cluster
    """
def cluster_lstm(clusters, test_cluster, test_price, data_portion, layer1 = 50, batch = 100, epoch = 150):

    models = []
    for x in range(len(clusters)):
        cluster_series = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_series[y] = clusters[x][y][0:data_portion]
        cluster_final_price = np.empty(len(cluster_series))
        for y in range(len(cluster_series)):
            cluster_final_price[y] = cluster_series[y][-1]

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cluster_series)
        scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
        print(scaled_data.shape)
        #scaled_test = scaler.transform(test_price[:data_portion].values.reshape(1, -1))
        #scaled_test = test_price.reshape(1, 11700, 1)
    

        model = Sequential()
        model.add(LSTM(units=layer1, return_sequences=False, input_shape=(scaled_data.shape[1], 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(scaled_data, cluster_final_price, epochs=epoch, batch_size=batch)
      
        models.append(model)

    # test performance of each model on test series
    test_predictions = []
    for x in range(len(models)):
        test_predictions.append(models[x].predict(test_price[0:data_portion].reshape(1, -1)))

    print(test_cluster+1)
    for x in range(len(test_predictions)):
        print(x+1, test_predictions[x], test_price[-1], mean_squared_error([test_price[-1]], test_predictions[x]))
    
    return(test_cluster+1, test_predictions[test_cluster][0][0], test_price[-1])