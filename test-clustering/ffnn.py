import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error



def feed_foward_nn(X, y, test_price, data_portion, clusters, test_cluster, layers):
    # Model Definition and Training
    #create array for test mse and train mse
    test_mse_arr = np.empty(100)
    mse_arr = np.empty(100)
    for x in range(100):
        model = MLPRegressor(hidden_layer_sizes=(layers, layers), max_iter=x+1, verbose=False).fit(X, y)

        # Predictions and Evaluation``
        predictions = model.predict(X)
        #pause = 
        # input("Press enter to continue")

        mse = mean_squared_error(y, predictions)
        predict_test = model.predict(test_price.iloc[0:data_portion].values.reshape(1, -1))
        test_mse = mean_squared_error([test_price.iloc[-1]], predict_test)
        #save train and test mse
        test_mse_arr[x] = test_mse
        mse_arr[x] = mse

    #plot test and train mse for each iteration
   
   
    """
    plt.plot(test_mse_arr)
    plt.plot(mse_arr)
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.show()
    """
 
    #create new models for each cluster and train on each cluster
    models = []
    for x in range(len(clusters)):
        cluster_series = np.empty((len(clusters[x]),data_portion))
        for y in range(len(clusters[x])):
            cluster_series[y] = clusters[x][y][0:data_portion]
        # get final price for each series in cluster
        cluster_final_price = np.empty(len(cluster_series))
        for y in range(len(cluster_series)):
            cluster_final_price[y] = cluster_series[y][-1]


        models.append(MLPRegressor(hidden_layer_sizes=(layers,layers), max_iter=100, verbose=False).fit(cluster_series, cluster_final_price))
    

    
    print(layers)
    # test performance of each model on test series
    test_predictions = []   
    for x in range(len(models)):
        test_predictions.append(models[x].predict(test_price.iloc[0:data_portion].values.reshape(1, -1)))


    #print cluster number, prediction, actual, and rmse
    print(test_cluster+1)
    for x in range(len(test_predictions)):
        print(x+1, test_predictions[x], test_price.iloc[-1], mean_squared_error([test_price.iloc[-1]], test_predictions[x]))

