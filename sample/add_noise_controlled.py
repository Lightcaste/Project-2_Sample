import math

import numpy as np
import scipy
import scipy.fftpack as fftpk
import scipy.io.wavfile
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
import array as arr 
from scipy.fftpack import fft

sample_rate, data = wavfile.read('male.wav')
FFT = abs(scipy.fft.fft(data))
times = np.arange(len(data))/float(sample_rate)
freqs = fftpk.fftfreq(len(FFT), (1.0/sample_rate))





# Gauss noise adding
rho = 10              # rho - spectral noise density unit/SQRT(Hz)
sr = sample_rate        # sr  - sample rate
mu = 0                  # mu  - mean value, optional
sigma =  rho * np.sqrt(sr/2)
noise = np.random.normal(mu, sigma, len(data))
data_adding_noise = data + noise
FFT_adding_noise = abs(scipy.fft.fft(data_adding_noise))
freqs_adding_noise = fftpk.fftfreq(len(FFT_adding_noise), (1.0/sample_rate))
plt.plot(times,data)
plt.show()


def SNR (signal, sigma):
    data_float = signal.astype('float64')
    length = len(data_float)
    temp = np.sum(data_float**2)
    Power_signal = temp / length
    P_noise = sigma**2
    SNR = Power_signal / P_noise
    SNR_dB = 10 * np.log10(SNR)
    return  SNR_dB


def MSE (signal, sigma):
    noise = np.random.normal(0, sigma, len(signal))
    signal_noise = signal + noise
    data_float = signal.astype('float64')
    data_noise_float = signal_noise.astype('float64')
    length =len(data_float)
    temp = np.sum((data_float-data_noise_float)**2)
    MSE = temp/ length
    MSE_dB = 10 * np.log10(MSE)
    return (MSE_dB)


# plot MSE and SNR in variable sigma
h = []
k = []

for x in range (0, sr//2, sr//100):
    h.append(SNR (data, x))
    k.append(MSE (data, x))
plt.ylabel('MSE (dB)')
plt.xlabel('SNR (dB)')
plt.plot(h,k)
plt.show()
