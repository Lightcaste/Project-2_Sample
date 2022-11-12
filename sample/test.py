import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fftpk
import scipy


sample_rate, data = wavfile.read('male.wav')
FFT = abs(scipy.fft.fft(data))
times = np.arange(len(data))/float(sample_rate)
freqs = fftpk.fftfreq(len(FFT), (1.0/sample_rate))


rho = 100                # rho - spectral noise density unit/SQRT(Hz)
sr = sample_rate        # sr  - sample rate
n = 1000                # n   - no of points
mu = 0                  # mu  - mean value, optional
sigma =  rho * np.sqrt(sr/2)
noise = np.random.normal(mu, sigma, len(data))
data_adding_noise = data + noise
FFT_adding_noise = abs(scipy.fft.fft(data_adding_noise))
freqs_adding_noise = fftpk.fftfreq(len(FFT_adding_noise), (1.0/sample_rate))

################## CALLING SIGNALS FROM OTHER SCRIPT############################

train = data
train  = train.astype('float64')
noisy = data_adding_noise
noisy = noisy.astype('float64')

def LMSFilter(xn,dn,M,mu,err):
    L = xn.shape[0]
    w = np.zeros(M)

    for k in range(L)[M:L]:
        x = xn[k-M:k][::-1]
        en = dn[k] - x.T.dot(w)
        if(en>err):
            break
        w = w + 2*mu*en*x
    
    yn = np.zeros(L)
    for k in range(L)[M:L]:
        x = xn[k-M:k][::-1]
        yn[k] = w.T.dot(x)

    return yn,w,en


yn,w,en = LMSFilter(train,noisy,50,1e-6,100)

plt.figure(figsize=(25, 10))
plt.plot(times, data_adding_noise, label='Data with Noise')
plt.plot(times, data, color ='red',label='Data without Noise')
plt.plot(times, yn, color ='g', label = 'Data after Filter')
#plt.xlim([0,times[-1]])
#plt.xlim([0,8])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc="upper right")
plt.title("Time Domain")
plt.show()
