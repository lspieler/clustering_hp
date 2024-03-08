from tslearn.metrics import soft_dtw
import numpy as np

def dtw_dist(price_series):
        # create distance matrix using soft dynamic time warping
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = soft_dtw(price_series[i], price_series[j], gamma=0.1)

    return dist_matrix

def euc_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = np.linalg.norm(price_series[i] - price_series[j])

    return dist_matrix

def correlation_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = np.linalg.norm(np.correlate(price_series[i], price_series[j]))

    return dist_matrix

def maximum_shifting_correlation_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = np.linalg.norm(np.correlate(price_series[i], price_series[j], mode='full'))

    return dist_matrix

