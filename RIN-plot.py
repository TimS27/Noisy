import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from scipy import constants

# laser
laser_wavelength = 1064e-9

# Oscilloscope settings
sample_interval = 1e-8  # Time between samples in seconds
record_length = 100000  # Number of samples
vertical_offset = 0.2124  # DC offset in volts
probe_attenuation = 1  # Probe attenuation factor

# ESA settings
esa_RBW = 1e4   # [10 kHz]

# Derived parameters
fs = 1 / sample_interval  # Sampling frequency in Hz
time = np.arange(record_length) * sample_interval  # Time array in seconds

# Reading CSV file
data1 = pd.read_csv("RIN-data/RIN-osci-noise-eater-off.csv")
data2 = pd.read_csv("RIN-data/RIN-osci-noise-eater-on.csv")
data_esa1 = pd.read_csv("RIN-data/RIN-ESA-noise-eater-off-20MHz-1s.csv")
data_esa2 = pd.read_csv("RIN-data/RIN-ESA-noise-eater-on-20MHz-1s.csv")
data = pd.read_csv("14102024-DANL-with-photodiode-on-blocked-1MHz-RBW.csv") # Photodiode dark noise

# Converting column data to list then array
#time = np.array(data['time'].tolist())
voltage1 = np.array(data1['voltage'].tolist())
voltage2 = np.array(data2['voltage'].tolist())
freqencies_esa1 = np.array(data_esa1['Hz'].tolist())
dBm_esa1 = np.array(data_esa1['dBm'].tolist())
freqencies_esa2 = np.array(data_esa2['Hz'].tolist())
dBm_esa2 = np.array(data_esa2['dBm'].tolist())

# Remove DC offset
signal1 = voltage1 - vertical_offset
signal2 = voltage2 - vertical_offset

# Calculate the power of the signal (mean square value)
signal_power1 = np.mean(signal1**2)
signal_power2 = np.mean(signal2**2)

# Define the laser's average optical power in watts (adjust as needed)
P_avg = 0.5  # 500 mW laser

# Welch's method to calculate power spectral density (PSD)
frequencies1, psd1 = welch(signal1, fs=fs, nperseg=record_length // 10)
frequencies2, psd2 = welch(signal2, fs=fs, nperseg=record_length // 10)

# Convert PSD to RIN (relative intensity noise) in dB/Hz
#rin_psd = psd / signal_power  # Normalize PSD by signal power
#rin_dBc_per_Hz = 10 * np.log10(rin_psd)
rin_dB_per_Hz1 = 10 * np.log10(psd1)
rin_dB_per_Hz2 = 10 * np.log10(psd2)

P_noise_lin_esa1 = 10 ** ((dBm_esa1 - 30) / 10)
P_noise_lin_esa2 = 10 ** ((dBm_esa2 - 30) / 10)
rin_dB_per_HZ_esa1 = 10 * np.log10(P_noise_lin_esa1 / (P_avg ** 2 * esa_RBW))
rin_dB_per_HZ_esa2 = 10 * np.log10(P_noise_lin_esa2 / (P_avg ** 2 * esa_RBW))

# Calculate electronic_power_spectral_density_dBHz from NEP
R = 0.5
r = 50
nep = 1.2e-11   # [W/sqrt(Hz)] optical power
gain = 1e4      # [V/A]
neea = nep * R * gain    # [V/sqrt(Hz)]
# square neea and use P=R*I^2
electronic_power_spectral_density = (neea ** 2) / r # [W/Hz]
# calculate electronic power spectral density per delta_f = RBW
#electronic_power_spectral_density_per_delta_f = electronic_power_spectral_density * (delta_f)
# calculate electronic power spectral density in dB/Hz
electronic_power_spectral_density_dBHz = 10 * np.log10(electronic_power_spectral_density / P_avg)

# Calculate shot noise level
shot_noise_linear = 2 * constants.h * (constants.c / laser_wavelength) / P_avg
shot_noise_dBHz = 10 * np.log10(shot_noise_linear)

# Find relaxation oscillation peak frequency
psd_max1 = frequencies1[np.argmax(psd1)]
psd_max2 = frequencies2[np.argmax(psd2)]

# Save data from FFT analysis to plot e.g. with simulation
np.savetxt('rin.csv', [p for p in zip(frequencies1[1:], rin_dB_per_Hz1[1:])], delimiter=',', fmt='%s')

# Plot the RIN curve
plt.figure(figsize=(10, 6))
plt.plot(frequencies1[1:], rin_dB_per_Hz1[1:], label="RIN noise eater off (Osci)")
plt.plot(frequencies2[1:], rin_dB_per_Hz2[1:], label="RIN noise eater on (Osci)")
plt.plot(freqencies_esa1, rin_dB_per_HZ_esa1, label="RIN noise eater off (ESA)")
plt.plot(freqencies_esa2, rin_dB_per_HZ_esa2, label="RIN noise eater on (ESA)")
plt.axhline(shot_noise_dBHz, color='r', linestyle='--', label="Shot Noise Limit")
plt.axhline(electronic_power_spectral_density_dBHz, color='purple', linestyle='--', label="NEP")
plt.axvline(psd_max1, color='black', linestyle='--', label=r'$f_{relaxation-oscillations}$')
plt.text(1e4, rin_dB_per_Hz1[1] + 1, "noise eater off", color='tab:blue')
plt.text(1e4, rin_dB_per_Hz2[1] - 3, "noise eater on", color='tab:orange')
plt.text(2e5, -110, r'$f_{relaxation-oscillations}$')
plt.text(2e4, shot_noise_dBHz + 1, 'shot noise', color='red')
plt.text(2e4, electronic_power_spectral_density_dBHz + 1, 'NEP', color='purple')
plt.xscale("log")
plt.xlim(1e4, 1e7)
plt.xlabel("Frequency (Hz)")
plt.ylabel("RIN (dB/Hz)")
plt.title("Mephisto - Measured RIN")
plt.grid(which="both", linestyle="--", linewidth=0.5)
plt.legend()
plt.show()