import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import constants
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.signal import medfilt

fs = 500e3
window = 2

""" file_balanced = "E:/Measurements/46/2025-03-05/balanced-highres3-2,422mW-total.npy"
file_autobalanced = "E:/Measurements/46/2025-03-05/autobalanced-highres.npy"
file_osci = "E:/Measurements/46/2025-03-05/osci-dark-noise-highres.npy" """
""" file_balanced = "E:/Measurements/46/2025-03-06/balanced-lueftung-aus-500kSs-10s.npy"
file_autobalanced = "E:/Measurements/46/2025-03-06/autobalanced-lueftung-aus-500kSs-10s.npy"
file_osci = "E:/Measurements/46/2025-03-05/osci-dark-noise-highres.npy"
file_signal = "E:/Measurements/46/2025-03-06/balanced-signal-lueftung-aus-500kSs-10s.npy" """

""" file_balanced = "E:/Measurements/46/2025-03-05/balanced-highres3-2,422mW-total.npy"
file_autobalanced = "E:/Measurements/46/2025-03-05/autobalanced-highres.npy"
file_osci = "E:/Measurements/46/2025-03-05/nirvana-dark-noise-highres.npy"
file_signal = "E:/Measurements/46/2025-03-06/balanced-signal-lueftung-aus-500kSs-10s.npy" """

""" file_balanced = "E:/Measurements/46/2025-03-13/bal-2256microW.npy"
file_autobalanced = "E:/Measurements/46/2025-03-13/autobal-2256microW.npy"
file_osci = "E:/Measurements/46/2025-03-13/osci-dark.npy"
file_signal = "E:/Measurements/46/2025-03-13/nirvana-dark.npy" """

""" file_balanced = "E:/Measurements/46/2025-03-13/bal-368microW.npy"
file_autobalanced = "E:/Measurements/46/2025-03-13/autobal-368microW.npy"
file_dark = "E:/Measurements/46/2025-03-14/nirvana-dark-noise.npy"
file_osci = "E:/Measurements/46/2025-03-05/nirvana-dark-noise-highres.npy"
file_signal = "E:/Measurements/46/2025-03-06/balanced-signal-lueftung-aus-500kSs-10s.npy" """

file_balanced_400 = "E:/Measurements/46/2025-03-14/bal-400microW.npy"
file_autobalanced_400 = "E:/Measurements/46/2025-03-14/autobal-400microW.npy"
file_balanced_800 = "E:/Measurements/46/2025-03-14/bal-800microW.npy"
file_autobalanced_800 = "E:/Measurements/46/2025-03-14/autobal-800microW.npy"
file_dark = "E:/Measurements/46/2025-03-14/nirvana-dark-noise.npy"
file_osci = "E:/Measurements/46/2025-03-13/osci-dark.npy"
file_signal = "E:/Measurements/46/2025-03-14/signalhalf-800microW.npy"

time_data = []
time_data.append(np.load(file_balanced_400)[:,0])
for i in [file_balanced_400, file_autobalanced_400, file_balanced_800,file_autobalanced_800, file_dark, file_osci, file_signal]:
        time_data.append(np.load(i)[:,1])

# Compute PSD using Welch's method
# Balanced
#test = time_data[1] - (2.422e-3 * 100e3 * 0.7)
frequencies, psd_balanced_400 = welch(time_data[1], fs=fs, nperseg=fs//window)
#psd_balanced_fluctuations = psd_balanced - (2.422e-3 * 100e3 * 0.7)
#psd_balanced_norm = psd_balanced_fluctuations / (2.422e-3 * 100e3 * 0.7)#(np.mean(time_data[4])*6)
psd_balanced_400_sqrt = np.sqrt(psd_balanced_400)  # Convert to Hz^(-0.5)
#psd_balanced_sqrt_cheated = psd_balanced_sqrt * 1.35

# Autobalanced
frequencies, psd_autobalanced_400 = welch(time_data[2], fs=fs, nperseg=fs//window)
#psd_autobalanced_shifted = np.where(frequencies <= 5000, psd_autobalanced * 5, psd_autobalanced)
#psd_autobalanced_shifted_sqrt = np.sqrt(psd_autobalanced_shifted)  # Convert to Hz^(-0.5)
#psd_autobalanced_norm = psd_autobalanced / 
psd_autobalanced_400_sqrt = np.sqrt(psd_autobalanced_400)
#psd_autobalanced_sqrt_cheated = psd_autobalanced_sqrt * 0.97

""" ############# Quick interpolation for gain fitting ###############
xp = [0,2.5e5]
fp = [3.3, 1]
factors = np.interp(frequencies, xp, fp)
print(factors) """


average_signal_800_test = np.mean(time_data[7])

# Nirvana Dark Noise
frequencies, psd_dark = welch(time_data[5], fs=fs, nperseg=fs//window)
psd_dark_sqrt = np.sqrt(psd_dark) / (average_signal_800_test*2)

################# Fitting and normalizing gain ################
# **Step 1: Remove Peaks Using a Median Filter**
filtered_psd = medfilt(psd_dark_sqrt, kernel_size=3333)  # Smooth high peaks

# **Step 2: Perform Polynomial Fit**
degree = 5  # Choose polynomial degree (4-6 works well)
coeffs = np.polyfit(frequencies, filtered_psd, degree)  # Fit polynomial
psd_fitted = np.polyval(coeffs, frequencies)  # Evaluate polynomial

# **Step 3: Normalize Fit to 1 at the Middle Frequency**
f_mid = 125000  # Set the middle frequency to 125 kHz
mid_index = np.argmin(np.abs(frequencies - f_mid))  # Find index closest to f_mid
psd_fitted /= psd_fitted[mid_index]  # Normalize to make value at f_mid = 1

""" # **Step 3: Normalize Fit to 1 at the Highest Frequency**
psd_fitted /= psd_fitted[-1]  # Normalize so that max frequency value is 1 """



time_data_test = (time_data[3] - average_signal_800_test)
#time = time_data[3] / average_signal_800_test
frequencies, psd_balanced_800 = welch(time_data_test, fs=fs, nperseg=fs//window)
psd_balanced_800 = psd_balanced_800 / psd_fitted
psd_balanced_800_sqrt = np.sqrt(psd_balanced_800) / (average_signal_800_test*2)

frequencies, psd_autobalanced_800 = welch(time_data[4], fs=fs, nperseg=fs//window)
#psd_autobalanced_shifted = np.where(frequencies <= 5000, psd_autobalanced * 5, psd_autobalanced)
#psd_autobalanced_shifted_sqrt = np.sqrt(psd_autobalanced_shifted)  # Convert to Hz^(-0.5)
#psd_autobalanced_norm = psd_autobalanced / 
psd_autobalanced_800_sqrt = np.sqrt(psd_autobalanced_800) / (average_signal_800_test*2)

# Osci
frequencies, psd_osci = welch(time_data[6], fs=fs, nperseg=fs//window)
psd_osci_sqrt = np.sqrt(psd_osci)

# Signal
frequencies, psd_signal = welch(time_data[7], fs=fs, nperseg=fs//window)
psd_signal_sqrt = np.sqrt(psd_signal) / (average_signal_800_test)



#psd_balanced_800_sqrt_normalized = (np.sqrt(psd_balanced_800) / (average_signal_800_test*2)) / psd_dark

# Compute shot noise level (assuming Poisson statistics)
average_signal_400 = np.mean(time_data[7])
average_signal_800 = np.mean(time_data[7])
wavelength = 1550e-9
r = 0.7  # Photodetector responsivity (A/W)
p = 0.400e-3 # Total optical power (W)
g = 100e3 # V/A
#shot_noise_psd_800 = np.sqrt(2 * constants.h * constants.c / (wavelength * p))# / average_signal_800_test
#shot_noise_psd_800 = (g * np.sqrt(2 * constants.elementary_charge * r * p))/ average_signal_800_test#/ (2.422e-3 * 100e3 * 0.7)# * np.ones_like(frequencies)
shot_noise_psd_400 =  g * np.sqrt(2 * constants.elementary_charge * 2 * (average_signal_400 / g))# * np.ones_like(frequencies)
#shot_noise_psd_800 =  (g * np.sqrt(2 * constants.elementary_charge * 2 * (average_signal_800 / g))) #/ average_signal_800_test# * np.ones_like(frequencies)
#shot_noise_psd_800 =  np.sqrt(2 * constants.elementary_charge / (average_signal_800_test/g))# * np.ones_like(frequencies)
shot_noise_psd_800 =  np.sqrt(2 * constants.elementary_charge / (average_signal_800*2/(1*g)))# * np.ones_like(frequencies)

#psd_balanced_800_sqrt = psd_balanced_800_sqrt / (g *r)
#print(frequencies)

# Plot the results
fig, ax1 = plt.subplots(figsize=(8,6))
#plt.semilogy(gain_f_norm)
#plt.semilogy(frequencies, psd_balanced_400_sqrt, label='Balanced 400 mW')
#plt.semilogy(frequencies, psd_autobalanced_400_sqrt, label='Autobalanced 400 mW')
#plt.semilogy(frequencies, psd_signal_sqrt, label='Signal 800 microW')
ax1.semilogy(frequencies, psd_balanced_800_sqrt, label='Balanced 800 microW')
#plt.semilogy(frequencies, psd_autobalanced_800_sqrt, label='Autobalanced 800 microW')
#plt.semilogy(frequencies, psd_osci, label='Osci')
#plt.loglog(frequencies, shot_noise_psd, '--', label='Shot Noise Level')
ax1.axhline(shot_noise_psd_800, color='r', linestyle='--', label="Shot Noise 800 microW")
#plt.axhline(shot_noise_psd_400, color='lightblue', linestyle='--', label="Shot Noise 400 mW")
ax1.plot(frequencies, psd_dark_sqrt, label='Detector Dark Noise')

ax2 = ax1.twinx()
ax2.plot(psd_fitted)

plt.xlim(0,1.25e5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Hz$^{-0.5}$)')
plt.legend()
plt.title('Power Spectral Density vs. Shot Noise')
ax1.grid(True, which='both', linestyle='--', alpha=0.6)
plt.show()