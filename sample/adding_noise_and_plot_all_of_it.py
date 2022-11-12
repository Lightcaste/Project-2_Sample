import math

import numpy as np
import scipy
import scipy.fftpack as fftpk
import scipy.io.wavfile
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt

sample_rate, data = wavfile.read('male.wav')
FFT = abs(scipy.fft.fft(data))
times = np.arange(len(data))/float(sample_rate)
freqs = fftpk.fftfreq(len(FFT), (1.0/sample_rate))

# Gauss noise adding
rho = 100              # rho - spectral noise density unit/SQRT(Hz)
sr = sample_rate        # sr  - sample rate
#n = 1000                # n   - no of points
mu = 5                  # mu  - mean value, optional
sigma =  rho * np.sqrt(sr/2)
noise = np.random.normal(mu, sigma, len(data))
data_adding_noise = data + noise

FFT_adding_noise = abs(scipy.fft.fft(data_adding_noise))
freqs_adding_noise = fftpk.fftfreq(len(FFT_adding_noise), (1.0/sample_rate))


#Print data
print("The sample rate is: " + str(sample_rate)+  " sample per second")
print("There are " + str(len(data)) + " samples.")
print("Then the audio should last: "+ str(len(data) / sample_rate))
print(data)
print(data_adding_noise)


# Export file
wavfile.write('male_adding_noise.wav',sample_rate,data_adding_noise.astype(np.int16))


# plot time domain
plt.figure(figsize=(25, 10))
plt.subplot(2, 1, 1)
plt.plot(times, data_adding_noise, label='Data with Noise')
plt.plot(times, data, color ='red',label='Data without Noise')
#plt.xlim([0,times[-1]])
#plt.xlim([0,8])

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend(loc="upper right")
plt.title("Time Domain")


# plot frequency domain
plt.subplot(2, 1, 2)


plt.plot(freqs_adding_noise[range(len(FFT_adding_noise)//2)], FFT_adding_noise[range(len(FFT_adding_noise)//2)], label='Data with Noise')                                                          
plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)],color='red', label='Data without Noise' ) 

plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title("Frequency Domain")
plt.legend(loc="upper right")

#plt.subplots_adjust(left=0.07, right=0.8, top=0.9, bottom=0.2)
plt.tight_layout(rect=[0, 0, 1, 1])

plt.show()


