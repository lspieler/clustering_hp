import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def fft_distance(ts1, ts2, detrend = True, dc_component = True, phase = True, distance_metric = 'euclidean'):

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