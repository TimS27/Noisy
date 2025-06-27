from scipy import constants
import numpy as np
import matplotlib.pyplot as plt
import binreduce

wavelength = 1064e-9
E_photon = (constants.h * constants.c) / wavelength
lod = 3 * 5.05e-18 * 0.25 / E_photon
lod2 = (3.2 * (1.61e-6 / 0.67e5) * 0.25) / E_photon
print(lod)


time = np.load("E:\\Measurements/46/2025-06-23/500microWsignal-complete-power.npy")[0]
voltage = np.load("E:\\Measurements/46/2025-06-23/500microWsignal-complete-power.npy")[1]

plt.figure()
plt.title('Raw data')
plt.plot(time, voltage)
plt.show()

sample_rate = 500000  # samples per second
bin_duration = 0.1  # seconds
bin_size = int(sample_rate * bin_duration)

# Reshape into bins
num_bins = len(voltage) // bin_size
binned = voltage[:num_bins * bin_size].reshape((num_bins, bin_size))
bin_averages = np.mean(binned, axis=1)

# Time axis for binned data
t_bins = np.arange(num_bins) * bin_duration

# Plot
plt.figure()
plt.plot(t_bins, bin_averages)
plt.xlabel("Time (s)")
plt.ylabel("Averaged signal (V)")
plt.title("10 s Measurement (Bin Average: 0.1 s)")
plt.grid()
plt.show()
