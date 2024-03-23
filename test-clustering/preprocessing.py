import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def fft_variant(ts1, ts2, detrend = True, dc_component = True, phase = True, distance_metric = 'euclidean'):
    # Detrend data and remove mean
    if detrend:
        ts1 = signal.detrend(ts1 - np.mean(ts1))
        ts2 = signal.detrend(ts2 - np.mean(ts2))

    # Apply FFT to both time series
    freq1 = np.fft.fft(ts1)
    freq2 = np.fft.fft(ts2)

    if not dc_component:
        # Remove the DC component
        freq1[0] = 0
        freq2[0] = 0
    
    if distance_metric == 'euclidean':
     # Calculate the Euclidean distance between the frequency representations
        distance = np.linalg.norm(freq1 - freq2)
    elif distance_metric == 'phase':
        # Compare phase spectrum as well
        phase1 = np.angle(freq1)
        phase2 = np.angle(freq2)
        distance = np.linalg.norm(phase1 - phase2)
    elif distance_metric == 'correlation':
        #compare in correlation domain
        corr1 = np.correlate(freq1, freq2)
        corr2 = np.correlate(freq2, freq1)
        distance = np.linalg.norm(corr1 - corr2)
    elif distance_metric == 'magnitude':
        mag1 = np.abs(freq1)
        mag2 = np.abs(freq2)
        distance = np.linalg.norm(mag1 - mag2)
    
    return abs(distance)


def pca():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=2)
    pca.fit(X)
    print(pca.explained_variance_ratio_)  
    print(pca.singular_values_)\
    

def smoothing(series, factor = 0.9):
    smoothed = np.zeros(len(series))
    for i in range(1, len(series)):
        smoothed[i] = factor * smoothed[i-1] + (1 - factor) * series[i]
    return smoothed


def fft_transform(series, detrend = True, dc_component = True):
    # Detrend data and remove mean
    if detrend:
        series = signal.detrend(series - np.mean(series))

    # Apply FFT to both time series
    freq = np.fft.fft(series)

    if not dc_component:
        # Remove the DC component
        freq[0] = 0
    
    return freq