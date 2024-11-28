import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.signal import welch

###################### Simulating Time Domain Data ########################

# Simulation parameters
fs = 100e6  # Sampling frequency (100 MHz)
delta_t = 1 / fs    # Time resolution
T = 1e-3   # Total duration of the signal (1 ms)
delta_f = 1 / T   # Frequency resolution
N = int(fs * T)  # Total number of samples
time = np.linspace(0, T, N, endpoint=False)  # Time vector

# Laser parameters
laser_wavelength = 1064e-9  # [nm]
P_avg = 230e-3  # Mean optical power (230 mW)
photon_energy = const.h * const.c / laser_wavelength  # [J]
avg_photon_number = P_avg / photon_energy  # [1/s]

# Relaxation oscillation parameters
f_ro = 650e3  # Relaxation oscillation frequency (650 kHz)
damping_factor = 0.05  # Damping factor for oscillations
relaxation_amplitude = 1e-5  # Amplitude of relaxation oscillations
num_bursts = 200  # Number of random bursts

# Noise contributions
white_noise_amplitude = 1e-5  # Amplitude of white noise
flicker_noise_amplitude = 1e-5  # Amplitude of 1/f noise
shot_noise_amplitude = np.sqrt(2 * const.e * P_avg / (photon_energy * delta_f))

# Generate noise components
np.random.seed(42)  # For reproducibility
white_noise = white_noise_amplitude * np.random.randn(N)  # Gaussian white noise
flicker_noise = flicker_noise_amplitude * np.cumsum(np.random.normal(0, 1, N)) / fs  # 1/f noise
shot_noise = shot_noise_amplitude * np.random.normal(0, 1, N)  # Shot noise

# Generate relaxation oscillations with low-frequency modulation
omega_ro = 2 * np.pi * f_ro  # Angular frequency
relaxation_signal = np.zeros_like(time)

# Random bursts
burst_times = np.random.choice(time, num_bursts, replace=False)  # Random start times
burst_durations = 0.0005 * np.random.rand(num_bursts)  # Random burst durations (up to 0.5 ms)

for burst_time, burst_duration in zip(burst_times, burst_durations):
    burst_start = int(burst_time * fs)
    burst_end = min(burst_start + int(burst_duration * fs), N)
    burst_length = burst_end - burst_start

    if burst_length > 0:
        envelope = np.exp(-damping_factor * (np.arange(burst_length) / fs))  # Damped envelope
        oscillation = relaxation_amplitude * np.sin(omega_ro * np.arange(burst_length) / fs)
        relaxation_signal[burst_start:burst_end] += oscillation * envelope

# Add low-frequency noise envelope
frequencies = np.fft.rfftfreq(N, d=delta_t)
flicker_psd = 1 / (frequencies + 1e-6)  # Avoid division by zero
low_freq_noise = np.fft.irfft(flicker_psd * np.random.normal(0, 1, len(frequencies)))
low_freq_noise = low_freq_noise[:N]  # Match length to time array
low_freq_noise /= np.max(np.abs(low_freq_noise))  # Normalize
modulated_relaxation_signal = relaxation_signal * (1 + 0.2 * low_freq_noise)  # Add low-frequency modulation

# Combine all components
fluctuations = (
    P_avg * (white_noise + flicker_noise)
    + modulated_relaxation_signal
    + shot_noise
)
intensity_signal = P_avg + fluctuations

###################### Calculate RIN Spectrum ########################

# Use Welch's method for power spectral density
frequencies, psd = welch(intensity_signal, fs=fs, nperseg=1024)
rin = 10 * np.log10(psd / (P_avg**2))  # RIN in dBc/Hz

###################### Plot Results ########################

plt.figure(figsize=(12, 8))

# Time-domain plot
plt.subplot(2, 1, 1)
plt.plot(time[:5000], intensity_signal[:5000], label="Intensity Signal")
plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.title("Simulated Laser Power (Time Domain)")
plt.grid()

# RIN spectrum
plt.subplot(2, 1, 2)
plt.semilogx(frequencies, rin, label="RIN Spectrum")
plt.axvline(f_ro, color="red", linestyle="--", label="Relaxation Oscillation Frequency")
plt.xlabel("Frequency (Hz)")
plt.ylabel("RIN (dBc/Hz)")
plt.title("RIN Spectrum with Relaxation Oscillations and Low-Frequency Noise")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
