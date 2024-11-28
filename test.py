import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# Parameters
f_ro = 50e3  # Relaxation oscillation frequency (50 kHz)
damping_factor = 5e3  # Damping factor
relaxation_amplitude = 0.002  # Amplitude scale
fs = 1e6  # Sampling rate (1 MHz)
T = 0.01  # Total time (10 ms)
N = int(fs * T)  # Number of samples
time = np.arange(N) / fs  # Time array

# Derived parameters
omega_ro = 2 * np.pi * f_ro  # Angular frequency of relaxation oscillation
gamma = damping_factor  # Damping coefficient

# Generate white noise
white_noise = np.random.randn(N)

# Define oscillator transfer function coefficients
b = [omega_ro**2]  # Numerator coefficients
a = [1, 2 * gamma / fs, (omega_ro / fs)**2]  # Denominator coefficients for discrete time

# Filter white noise to simulate relaxation oscillations
relaxation_oscillations = lfilter(b, a, white_noise) * relaxation_amplitude

# Remove any offset (optional, for visualization purposes)
relaxation_oscillations -= np.mean(relaxation_oscillations)

# Plot the time-domain relaxation oscillations
plt.figure(figsize=(10, 4))
plt.plot(time, relaxation_oscillations, label="Relaxation Oscillations")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("White-Noise-Excited Relaxation Oscillations")
plt.legend()
plt.grid()
plt.show()