        
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def prep_data(cluster_series, data_portion, forecast, dist = 10, normalize = True):

        X = cluster_series

        dist = 10
        num_steps = (len(X[-1]) - data_portion-forecast) // dist 
        xdf = np.zeros((len(X), num_steps, data_portion))
        ydf = np.zeros((len(X), num_steps, forecast))
        # get superfluous data points
        superfluous = (len(X[-1]) - data_portion-forecast) % dist
            # Create a rolling window of x based on data portion, moving by 200 points each step
        for h in range(len(X)):
            for i in range(0, len(X[-1]) - data_portion - forecast - superfluous, dist):  # Step by 200
                # Calculate the index for storing the data, considering the step size
                index = i // dist 
                xdf[h][index] = X[h][i:i+data_portion]
                ydf[h][index] = X[h][i+data_portion:i+data_portion+forecast]


    
        xdf = xdf.reshape(-1, xdf.shape[-1])
        ydf = ydf.reshape(-1, ydf.shape[-1])
        # normalize each period by first value of the period
        if normalize:
            for i in range(len(xdf)):
                xdf[i] = (xdf[i] +1) /( xdf[i][-1] +1) -1

            for i in range(len(ydf)):
                ydf[i] = (ydf[i] +1) /(ydf[i][0] +1) -1


        ydf = ydf.reshape(-1, forecast)
        xdf = xdf.reshape(-1, data_portion)
        xdf = xdf.reshape((xdf.shape[0], xdf.shape[1], 1))

        return xdf, ydf
