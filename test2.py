import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import constants

# Laser parameters
lambda_laser = 1064e-9  # Wavelength of Nd:YAG laser (m)
laser_power = 0.5  # Laser power in watts (500 mW)
photon_energy = constants.h * constants.c / lambda_laser  # Energy of a single photon (J)
photon_rate = laser_power / photon_energy  # Photons per second [1/s]

# Simulation parameters
fs = 1e8  # Sampling frequency in Hz
duration = 1e-3  # Duration of the signal in seconds
n_samples = int(fs * duration)  # Total number of samples
time = np.linspace(0, duration, n_samples, endpoint=False)

# Noise parameters
rin_low_freq = -130  # Baseline RIN at low frequencies in dBc/Hz
rin_high_freq = -150  # Baseline RIN at high frequencies in dBc/Hz
relaxation_freq = 650e3  # Relaxation oscillation frequency in Hz
relaxation_quality = 5  # Q-factor of the relaxation oscillation
relaxation_peak = -105  # Peak RIN level in dBc/Hz

# Generate white noise
white_noise = np.random.normal(0, 1, n_samples)

# Generate flicker noise (1/f noise)
frequencies = np.fft.rfftfreq(n_samples, 1 / fs)
flicker_noise_amplitude = 1 / (frequencies + 1e-6)  # Avoid division by zero
flicker_noise = np.fft.irfft(flicker_noise_amplitude * np.random.normal(0, 1, len(frequencies)))

# Combine white and flicker noise for baseline RIN
base_noise = white_noise + flicker_noise
base_noise /= np.std(base_noise)  # Normalize to unit variance

# Scale baseline noise to achieve the desired RIN spectrum
base_noise_scaling = np.linspace(10**(rin_low_freq / 20), 10**(rin_high_freq / 20), n_samples) #*100
base_noise *= base_noise_scaling

# Calculate shot noise
shot_noise_amplitude = np.sqrt(photon_rate)  # Shot noise amplitude in counts/sqrt(Hz)
shot_noise = np.random.normal(0, 1, n_samples)
shot_noise /= np.std(shot_noise)  # Normalize to unit variance
shot_noise *= np.sqrt(shot_noise_amplitude / fs)  # Scale shot noise amplitude
shot_noise *= 0.00005

# Generate relaxation oscillation noise
omega_0 = 2 * np.pi * relaxation_freq  # Angular frequency
gamma = omega_0 / (2 * relaxation_quality)  # Damping coefficient

# Relaxation oscillator transfer function response
response = 1 / np.sqrt((omega_0**2 - (2 * np.pi * frequencies)**2)**2 + (2 * gamma * 2 * np.pi * frequencies)**2)
response /= np.max(response)  # Normalize to 1 at peak frequency

# Normalize relaxation noise power to achieve -105 dBc/Hz
relaxation_scaling = 10**((relaxation_peak - rin_low_freq) / 20) #*0.0015
relaxation_noise = relaxation_scaling * np.fft.irfft(response * np.fft.rfft(white_noise))

# Combine all noise components
total_noise = base_noise + shot_noise + relaxation_noise

# Simulate laser intensity signal (baseline intensity = 1)
laser_signal = 1 + total_noise

# Calculate the RIN spectrum using Welch's method
frequencies, psd = welch(laser_signal, fs=fs, nperseg=1024)
rin = 10 * np.log10(psd)# / np.mean(laser_signal)**2)  # RIN in dBc/Hz

# Plot the results
plt.figure(figsize=(10, 8))

# Time-domain signal
plt.subplot(3, 1, 1)
plt.plot(time[:2000], laser_signal[:2000], label="Laser Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time-Domain Laser Signal")
plt.grid()

# Individual noise components
plt.subplot(3, 1, 2)
plt.plot(time[:2000], base_noise[:2000], label="Base Noise (White + Flicker)")
plt.plot(time[:2000], shot_noise[:2000], label="Shot Noise")
plt.plot(time[:2000], relaxation_noise[:2000], label="Relaxation Noise")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Noise Components")
plt.legend()
plt.grid()

# RIN spectrum
plt.subplot(3, 1, 3)
plt.semilogx(frequencies, rin, label="RIN Spectrum")
plt.axhline(rin_low_freq, color="red", linestyle="--", label="Baseline RIN Low")
plt.axhline(rin_high_freq, color="blue", linestyle="--", label="Baseline RIN High")
plt.axvline(relaxation_freq, color="green", linestyle="--", label="Relaxation Peak")
plt.xlabel("Frequency (Hz)")
plt.ylabel("RIN (dBc/Hz)")
plt.title("Relative Intensity Noise (RIN)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

