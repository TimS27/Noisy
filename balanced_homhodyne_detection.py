import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.constants import h, c, e

# Constants
wavelength = 1064e-9  # Laser wavelength (m)
frequency = c / wavelength  # Optical frequency (Hz)
power_laser = 7e-3  # Laser power (W)
quantum_efficiency = 0.5  # Detector quantum efficiency
#rin_file = "RIN-data/RIN-osci-noise-eater-off.csv"  # File containing RIN data in dB/Hz

data = pd.read_csv("rin.csv")
frequency = np.array(data['frequency'].tolist())
dBHz = np.array(data['dBHz'].tolist())

# Simulation parameters
sampling_rate = 1e8  # Sampling rate for time-domain simulation (Hz)
duration = 1e-3  # Simulation duration (s)
frequencies = np.fft.rfftfreq(int(sampling_rate * duration), 1 / sampling_rate)

# Load RIN data (assumes two columns: frequency (Hz), RIN (dB/Hz))
#rin_data = np.loadtxt(rin_file)
#rin_freq = rin_data[:, 0]  # Frequency points (Hz)
#rin_psd_db = rin_data[:, 1]  # RIN in dB/Hz

# Convert RIN to linear scale
rin_psd_linear = 10**(dBHz / 10)  # Linear RIN (W/W)/Hz

# Interpolate RIN data to simulation frequencies
rin_interpolated = np.interp(frequencies, frequency, rin_psd_linear)

# Generate noise in time domain
np.random.seed(42)  # For reproducibility
num_samples = int(sampling_rate * duration)
rin_noise = np.sqrt(rin_interpolated) * np.random.normal(0, 1, len(frequencies))
rin_time = np.fft.irfft(rin_noise, num_samples)

# Add shot noise
photon_energy = h * frequency
mean_photons_per_sec = power_laser / photon_energy
shot_noise_std = np.sqrt(mean_photons_per_sec) * e * quantum_efficiency
shot_noise = np.random.normal(0, shot_noise_std, num_samples)

# Simulate balanced detector
photocurrent1 = rin_time + shot_noise  # Detector 1 signal
photocurrent2 = -rin_time + shot_noise  # Detector 2 signal
differential_signal = photocurrent1 - photocurrent2

# Calculate PSD of the differential signal
differential_psd = np.abs(np.fft.rfft(differential_signal))**2 / sampling_rate

# Plot results
plt.figure(figsize=(10, 6))
plt.loglog(frequencies, rin_interpolated, label="Interpolated RIN (linear)")
plt.loglog(frequencies, differential_psd[:len(frequencies)], label="Differential Signal PSD")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (W/Hz)")
plt.title("Balanced Homodyne Detection Noise Simulation")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.show()
