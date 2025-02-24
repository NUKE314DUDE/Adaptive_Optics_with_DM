import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Misc_Tools import gaussian

np.random.seed(42)
x = np.linspace(-5, 5, 100)
y = gaussian(x) * np.sqrt(2*np.pi)
noise = np.random.normal(size = x.size)
y_noise = y + 0.1 * noise

coff = curve_fit(gaussian, x, y_noise)[0]

print(coff)
plt.figure(figsize = (10 ,6.18))
plt.ion()
plt.plot( x, y, label = 'original', color = 'green')
plt.plot(x, y_noise, label = 'with noise', color = 'red')
plt.plot( x, gaussian(x, *coff), label = 'fitted curve', color = 'blue')
plt.legend()
plt.ioff()
plt.show()