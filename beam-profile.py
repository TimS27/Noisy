import numpy as np
import matplotlib.pyplot as plt

# Load 2D intensity data (e.g., tab- or comma-separated)
beam_data = np.loadtxt("E:/Measurements/46/2025-05-06/beam-profile-at-detector-disctance-test.csv", delimiter=';')  # adjust delimiter as needed

plt.imshow(beam_data, cmap='hot', origin='lower')  # other cmaps: gnuplot2, magma, hot, ...
plt.colorbar(label='Intensity')
plt.title("Beam Profile")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")
plt.show()