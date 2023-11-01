import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import tensorflow as tf
from sklearn.metrics import mean_squared_error


def run_lstm(X, y, test_price, data_portion, layer1 = 50, layer2=30, batch = 100, epoch = 150):
    
    #strategy = tf.distribute.MirroredStrategy()


 
    test_x = test_price[0:data_portion]
    test_y = test_price[data_portion:]

    plt.plot(X[0])
    plt.plot(y[0])    
    scaler = MinMaxScaler()
    model = Sequential()
    scaled_data = scaler.fit_transform(X)
    test = scaler.transform(test_price[:data_portion].reshape(1, -1))
    test = test.reshape(1, X.shape[1], 1)
    scaled_test = scaler.transform(X[-1].reshape(1, -1))
    scaled_test = scaled_test.reshape(1, X.shape[1], 1)
    scaled_data = scaled_data.reshape(scaled_data.shape[0], scaled_data.shape[1], 1)
    
    model.add(LSTM(units=layer1, return_sequences=False, input_shape=(X.shape[1], 1)))

    model.add(Dense(units=y.shape[1]))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(scaled_data, y, epochs=epoch, batch_size=batch)


    # if model rmse is too high train 10 more epochs
    count = 0
    while count <= 10:
        if mean_squared_error(y, model.predict(scaled_data)) > 0.2:
            model.fit(scaled_data, y, epochs=10, batch_size=batch)
            count += 1
            if count == 10:
                return(0, 0)

    # 4. Making Predictionscç
    # To predict the price for the next day using the last 250 days:
    """
    test = test_price.iloc[0:data_portion].values
    test_flattened = test.flatten().reshape(-1, 1)
    scaled_test = scaler.transform(test_flattened)
    scaled_test = scaled_test.reshape(1, 11700, 1)
    """
    # test on test data

    predicted_price = model.predict(test) # De-normalize
    print(f'Acutal Price for Next Day: {test_y}')
    print(f"Predicted Price for Next Day: {predicted_price}")

    return(predicted_price[0][0], test_y[0])

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
        model.add(LSTM(units=layer1,return_sequences=False, input_shape=(scaled_data.shape[1], 1)))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(scaled_data, cluster_final_price, epochs=epoch, batch_size=batch)

        # if model rmse is too high train 10 more epochs
        count = 0
        while count < 10:
            if mean_squared_error(cluster_final_price, model.predict(scaled_data)) > 0.01:
                model.fit(scaled_data, cluster_final_price, epochs=10, batch_size=batch)
                count += 1
                if count == 10:
                    return(0, 0)
        models.append(model)

    ttest = scaler.transform(test_price[0:data_portion].reshape(1, -1))
    # test performance of each model on test series
    test_predictions = []
    for x in range(len(models)):
        test_predictions.append(models[x].predict(ttest.reshape(1, ttest.shape[1], 1)))

    print(test_cluster+1)
    for x in range(len(test_predictions)):
        print(x+1, test_predictions[x], test_price[-1], mean_squared_error([test_price[-1]], test_predictions[x]))
    
    return(test_predictions[test_cluster][0][0], test_price[-1])