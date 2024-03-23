import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, np.pi*2, 100)
y = np.cos(x)
spec = np.abs(np.fft.fftshift(np.fft.ifft(y)))
sample = np.fft.fftshift(np.fft.fftfreq(x.size, d=(x[1]-x[0])/2/np.pi))

center_i = x.size//2
max_i1 = np.argmax(spec[center_i:])
max_i2 = np.argmax(spec[:center_i])

print(sample[center_i:][max_i1])
print(sample[:center_i][max_i2])

plt.plot(sample, spec)
plt.show()