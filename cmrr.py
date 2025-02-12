import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from scipy import constants

def load_data(file_path):
    data = pd.read_csv(file_path, header=0, delimiter=',')  # Read CSV assuming first row is header
    data = data.astype(float)  # Convert all data to float
    return data

def compute_psd(time, voltage, fs=None, nperseg=100000):
    """ Compute Power Spectral Density (PSD) using Welch’s method """
    if fs is None:
        fs = 1 / np.mean(np.diff(time))  # Estimate sampling frequency
    
    f, Pxx = welch(voltage, fs=fs, nperseg=nperseg)
    Pxx_dB = 10 * np.log10(Pxx)  # Convert to dB/Hz
    return f, Pxx_dB

def compute_fft_psd(time, voltage, fs):
    """ Compute PSD using FFT with high frequency resolution """
    #fs = 1 / np.mean(np.diff(time))  # Sampling frequency
    N = len(voltage)  # Number of points
    freq = np.fft.rfftfreq(N, d=1/fs)  # Positive frequencies
    fft_vals = np.fft.rfft(voltage)  # FFT computation
    psd = (np.abs(fft_vals) ** 2) / (fs * N)  # Normalize to get PSD
    psd_dB = 10 * np.log10(psd)  # Convert to dB/Hz
    return freq, psd_dB

def compute_cmrr(psd_unbalanced, psd_balanced):
    """ Compute Common Mode Rejection Ratio (CMRR) in dB """
    cmrr = psd_unbalanced - psd_balanced
    return cmrr


################# Load data #################
# Load data
""" file_unbalanced = "balanced-detector-data/Koheron/koheron-det1-chopped-800Hz-rin-highres-100kSs-5MS.csv"
file_balanced = "balanced-detector-data/Koheron/koheron-balanced-chopped-800Hz-rin-highres-100kSs-5MS.csv" """

file_unbalanced = "balanced-detector-data/Koheron/koheron-det1-chopped-800Hz-rin-highres-100kSs-5MS.csv"
file_balanced = "balanced-detector-data/Koheron/koheron-balanced-chopped-800Hz-rin-highres-100kSs-5MS.csv"

#file_balanced_unchopped = "balanced-detector-data/Koheron/koheron-balanced-rin-highres-2MSs-20MS.csv"
file_dark_noise = "balanced-detector-data/Koheron/koheron-dark-noise-100kSs-1MS.csv"
file_osci_noise = "osci-data/osci-dark-noise-100kSs-1MS.csv"

time_unbal, voltage_unbal = load_data(file_unbalanced)["time"].values, load_data(file_unbalanced)["voltage"].values
time_bal, voltage_bal = load_data(file_balanced)["time"].values, load_data(file_balanced)["voltage"].values
#time_bal_unchopped, voltage_bal_unchopped = load_data(file_balanced_unchopped)["time"].values, load_data(file_balanced_unchopped)["voltage"].values
time_dark_noise, voltage_dark_noise = load_data(file_dark_noise)["time"].values, load_data(file_dark_noise)["voltage"].values
time_osci_noise, voltage_osci_noise = load_data(file_osci_noise)["time"].values, load_data(file_osci_noise)["voltage"].values


# Compute PSDs
freq_unbal, psd_unbal = compute_psd(time_unbal, voltage_unbal, fs=1e5)
freq_bal, psd_bal = compute_psd(time_bal, voltage_bal, fs=1e5)
#freq_bal_unchopped, psd_bal_unchopped = compute_psd(time_bal_unchopped, voltage_bal_unchopped, fs=2e6)
freq_dark, psd_dark = compute_psd(time_dark_noise, voltage_dark_noise, fs=1e5)
freq_osci, psd_osci = compute_psd(time_osci_noise, voltage_osci_noise, fs=1e5)


""" # Procedure for CMRR at different frequencies
file_combi_balanced = "balanced-detector-data/Koheron/combi_balanced.csv"
time_data_balanced = []
freq_data_balanced = []
for i in ['time','100Hz','200Hz','300Hz','400Hz']:
    time_data_balanced.append(load_data(file_combi_balanced)[i].values)

freq, dB = compute_psd(time_data_balanced[0], time_data_balanced[1], fs=1e5)
freq_data_balanced.append(freq)
for j in np.linspace(1, len(time_data_balanced)-1, len(time_data_balanced)-1, dtype=int):
    freq, dB = compute_psd(time_data_balanced[0], time_data_balanced[j], fs=1e5)
    freq_data_balanced.append(dB)

file_combi_det1 = "balanced-detector-data/Koheron/combi_det1.csv"
time_data_det1 = []
freq_data_det1 = []
for i in ['time','100Hz','200Hz','300Hz','400Hz']:
    time_data_det1.append(load_data(file_combi_det1)[i].values)

freq, dB = compute_psd(time_data_det1[0], time_data_det1[1], fs=1e5)
freq_data_det1.append(freq)
for j in np.linspace(1, len(time_data_det1)-1, len(time_data_det1)-1, dtype=int):
    freq, dB = compute_psd(time_data_det1[0], time_data_det1[j], fs=1e5)
    freq_data_det1.append(dB)

# Compute CMRR
#print(compute_cmrr(psd_unbal[np.argmax(psd_unbal)], psd_bal[np.argmax(psd_unbal)]))
#cmrr = compute_cmrr(psd_unbal, psd_bal)
freq_cmrr = [100, 200, 300, 400]
result_cmrr = np.max(freq_data_balanced[1:], axis=1) - np.max(freq_data_det1[1:], axis=1)
print(result_cmrr)
#a[np.argmax(a)]-b[np.argmax(b)]
#print(a[np.argmax(a)]-b[np.argmax(b)])
#for i in np.linspace(0,4,5):
#    result_cmrr.append(compute_cmrr(i[np.argmax(i)], (b[i])[np.argmax(b[i])])) """


""" shot_noise_photons = np.sqrt(photons)
shot_noise_current = constants.elementary_charge*shot_noise_photons
shot_noise_voltage = shot_noise_current*Rf
shot_noise_dBHz = 20 * np.log10(shot_noise_voltage) """

# Calculate shot noise level
r = 0.65  # Responsivity in A/W
laser_wavelength = 1064e-9
Rf =39e3  # Transimpedance gain in ohms
P_avg = 100e-6
nu = constants.c / laser_wavelength
photons = P_avg/(constants.h*nu)
current = P_avg * r
shot_noise_current = np.sqrt(2 * constants.elementary_charge * current)
shot_noise_voltage = shot_noise_current*Rf
# Compute shot noise in dB/Hz
shot_noise_dBHz = 20 * np.log10(shot_noise_voltage)

# Compute shot noise current (RMS)
#shot_noise = np.sqrt(2 * constants.elementary_charge * current)
# Convert shot noise current to voltage using transimpedance gain
#shot_noise_voltage = shot_noise * Rf


############## Plotting #################
# Plot PSDs
plt.figure(figsize=(10, 6))
plt.semilogx(freq_unbal, psd_unbal, label="Unbalanced (Single Detector) 800 Hz")
plt.semilogx(freq_bal, psd_bal, label="Balanced (Difference) 800 Hz")
#plt.semilogx(freq_bal_unchopped, psd_bal_unchopped, label="Balanced (Difference) unchopped")
plt.semilogx(freq_dark, psd_dark, label="Detector Dark Noise")
plt.semilogx(freq_osci, psd_osci, label="System Noise Floor")
plt.xlim(10,5e4)
#plt.axhline(shot_noise_dBHz, color='r', linestyle='--', label="Shot Noise Limit")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB/Hz)")
plt.title("PSD of Unbalanced vs. Balanced Detection")
plt.legend()
plt.grid(True)
plt.show()

""" # Plot CMRRs
plt.figure(figsize=(10, 6))
plt.semilogx(freq_cmrr, result_cmrr, label="CMRR")
plt.xlim(10,1000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("CMRR (dB)")
plt.ylim(-38,-32)
plt.xlim(-1e9,1e3)
plt.title("CMRR at different chopping frequencies")
plt.legend()
plt.grid(True)
plt.show() """