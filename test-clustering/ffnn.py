import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error



def feed_foward_nn(X, y, test_price, data_portion, layers, iters):
    # Model Definition and Training
    #create array for test mse and train mse

    y = X[:,data_portion:]
    X = X[:,:data_portion]
   
    model = MLPRegressor(hidden_layer_sizes=(layers, layers), max_iter=iters, verbose=False).fit(X, y)

    # Predictions and Evaluation``
    predictions = model.predict(X)
    #pause = 
    # input("Press enter to continue")

    mse = mean_squared_error(y, predictions)
    predict_test = model.predict(test_price[0:data_portion].reshape(1, -1))
    print(test_price[data_portion:].shape)
    print(predict_test[0].shape)
    test_mse = mean_squared_error(test_price[data_portion:], predict_test[0])
    #save train and test mse
    test_mse_arr = test_mse
    mse_arr= mse
    print(predict_test[-1], test_price[data_portion], test_price[-1])
    return (predict_test[-1], test_price[data_portion], test_price[-1])