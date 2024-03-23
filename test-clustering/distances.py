from tslearn.metrics import soft_dtw
import numpy as np
from preprocessing import fft_transform
from fastdtw import fastdtw

def compute_distance(i, j, price_series):
    return i, j, soft_dtw(price_series[i], price_series[j], gamma=1.0)

def dtw_dist(price_series):
    n = len(price_series)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = fastdtw(price_series[i], price_series[j])[0]
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix

def euc_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = np.linalg.norm(price_series[i] - price_series[j])
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix

def corr(i, j, price_series):
    return i, j, np.linalg.norm(np.correlate(price_series[i], price_series[j], mode='full'))

def correlation_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = np.linalg.norm(np.correlate(price_series[i], price_series[j], mode='full'))
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix

def msc_dist(i, j, price_series):
    signal1 = np.array(price_series[i])
    signal2 = np.array(price_series[j])
    
    # Ensure signals are 1-dimensional
    if signal1.ndim != 1 or signal2.ndim != 1:
        raise ValueError("Input signals must be 1-dimensional arrays.")
    
    # Compute cross-correlation
    corr = np.correlate(signal1, signal2, mode='full')
    
    # Find shift that maximizes correlation
    shift = np.argmax(corr) - (len(signal1) - 1)
    max_corr = corr.max()
    
    return i, j, abs(shift), max_corr

def maximum_shifting_correlation_dist(price_series):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            i, j, shift, max_corr = msc_dist(i, j, price_series)
            dist_matrix[i][j] = abs(shift)
            dist_matrix[j][i] = abs(shift)

    return dist_matrix

def fft_distances(price_series, distance_metric='euclidean', detrend=True, dc_component=True, phase=True):
    dist_matrix = np.zeros((len(price_series), len(price_series)))
    for i in range(len(price_series)):
        for j in range(len(price_series)):
            dist_matrix[i][j] = fft_transform(price_series[i], price_series[j], distance_metric, detrend, dc_component, phase)
            dist_matrix[j][i] = dist_matrix[i][j]

    return dist_matrix
