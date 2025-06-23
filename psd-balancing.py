import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy import constants
from scipy.optimize import curve_fit
from scipy.signal import medfilt
from isfread_py3 import isfread
#from scipy.signal import savgol_filter
#from scipy.integrate import cumulative_trapezoid

font_size_label = 18
font_size_title = 20

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

# Functions
def compute_fosn(rin, rin_shot_noise):
    factor = np.abs(rin / rin_shot_noise) #factor = 10 ** (np.abs(rin - rin_shot_noise) / 10)

    return np.abs(factor)

def compute_cmrrs(unbalanced, balanced):
    cmrrs = 20 * np.log10(unbalanced / balanced)        # Factor 20 for (Amplitude) RIN values [1/sqrt(Hz)], factor 10 for (Power) RIN^2 values [V^2/Hz]
    return cmrrs

# Sample Parameters
fs = 250e3
window = 16
samples = 1e6

# Data Files
""" file_balanced = "E:/Measurements/46/2025-03-05/balanced-highres3-2,422mW-total.npy"
file_autobalanced = "E:/Measurements/46/2025-03-05/autobalanced-highres.npy"
file_osci = "E:/Measurements/46/2025-03-05/osci-dark-noise-highres.npy" """
""" file_balanced = "E:/Measurements/46/2025-03-06/balanced-lueftung-aus-500kSs-10s.npy"
file_autobalanced = "E:/Measurements/46/2025-03-06/autobalanced-lueftung-aus-500kSs-10s.npy"
file_osci = "E:/Measurements/46/2025-03-05/osci-dark-noise-highres.npy"
file_signal = "E:/Measurements/46/2025-03-06/balanced-signal-lueftung-aus-500kSs-10s.npy" """

""" file_balanced_800 = "E:/Measurements/46/2025-03-05/balanced-highres3-2,422mW-total.npy"
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


""" file_balanced_400 = "E:/Measurements/46/2025-03-14/bal-400microW.npy"
file_autobalanced_400 = "E:/Measurements/46/2025-03-14/autobal-400microW.npy"
file_balanced_800 = "E:/Measurements/46/2025-03-14/bal-800microW.npy"
file_autobalanced_800 = "E:/Measurements/46/2025-03-14/autobal-800microW.npy"
file_dark = "E:/Measurements/46/2025-03-14/nirvana-dark-noise.npy"
file_osci = "E:/Measurements/46/2025-03-13/osci-dark.npy"
file_signal = "E:/Measurements/46/2025-03-14/signalhalf-800microW.npy" """

""" file_balanced_400 = "E:/Measurements/46/2025-04-07/400microW-balanced-rin-20s-5MS.npy"
file_autobalanced_400 = "E:/Measurements/46/2025-04-07/400microW+nd03abinsignal-autobalanced-rin-20s-5MS.npy"
file_balanced_800 = "E:/Measurements/46/2025-03-14/bal-800microW.npy"
file_autobalanced_800 = "E:/Measurements/46/2025-03-14/autobal-800microW.npy"
file_dark = "E:/Measurements/46/2025-03-14/nirvana-dark-noise.npy"
file_osci = "E:/Measurements/46/2025-03-13/osci-dark.npy"
file_signal = "E:/Measurements/46/2025-03-14/signalhalf-800microW.npy"
file_signal_400 = "E:/Measurements/46/2025-03-14/signalhalf-400microW.npy" """

file_balanced_400 = "E:/Measurements/46/2025-04-29/rin-400microW-LO-bal-2.npy"
file_autobalanced_400 = "E:/Measurements/46/2025-04-28/rin-800microW-LO-and-NE03AB-autobal.npy"
file_balanced_800 = "E:/Measurements/46/2025-03-14/bal-800microW.npy"
file_autobalanced_800 = "E:/Measurements/46/2025-03-14/autobal-800microW.npy"
file_dark = "E:/Measurements/46/2025-03-14/nirvana-dark-noise.npy"
file_osci = "E:/Measurements/46/2025-03-13/osci-dark.npy"
file_signal = "E:/Measurements/46/2025-04-29/rin-400microW-LO-signal.npy"
file_signal_400 = "E:/Measurements/46/2025-03-14/signalhalf-400microW.npy"

# Read time data into one array
time_data = []
""" time_data.append(np.load(file_balanced)[:,0])
for i in [file_balanced, file_autobalanced, file_dark, file_osci, file_signal]:
        time_data.append(np.load(i)[:,1]) """
time_data.append(np.load(file_balanced_400)[:,0])
for i in [file_balanced_400, file_autobalanced_400, file_balanced_800,file_autobalanced_800, file_dark, file_osci, file_signal, file_signal_400]:
        time_data.append(np.load(i)[:,1])

file_500microW_bal = "E:/Measurements/46/2025-06-23/500microW-bal.isf"
file_500microW_signal = "E:/Measurements/46/2025-06-23/500microW-signal.isf"
file_500microW_signal_complete_power = "E:/Measurements/46/2025-06-23/500microWsignal-complete-power.isf"
file_500microW_autobal_no_attenuator = "E:/Measurements/46/2025-06-23/500microW-autobal-no-attenuator.isf"
file_nirvana_dark_noise = "E:/Measurements/46/2025-06-23/nirvana-dark-noise.isf"
file_osci_dark_noise = "E:/Measurements/46/2025-06-23/osci-dark-noise.isf"


time_data.append(np.linspace(0,10,5000000,endpoint=False))  # add time axis
for file in ["file_500microW_bal","file_500microW_signal","file_500microW_signal_complete_power","500microW-autobal-no-attenuator","file_nirvana_dark_noise","file_osci_dark_noise"]:
        data, header = isfread(file)
        time_data.append(data)


average_signal_400 = np.mean(time_data[8])
average_signal_800 = np.mean(time_data[7])

# Balanced
# Compute PSD using Welch's method
#test = time_data[1] - (2.422e-3 * 100e3 * 0.7)
bal_400_fluctuations = time_data[1] - np.mean(time_data[1])
frequencies, psd_balanced_400 = welch(bal_400_fluctuations, fs=fs, nperseg=samples//window)
#psd_balanced_fluctuations = psd_balanced - (2.422e-3 * 100e3 * 0.7)
#psd_balanced_norm = psd_balanced_fluctuations / (2.422e-3 * 100e3 * 0.7)#(np.mean(time_data[4])*6)
psd_balanced_400_rel = psd_balanced_400 / ((2*average_signal_400)**2)
psd_balanced_400_sqrt = np.sqrt(psd_balanced_400_rel)  # Convert to Hz^(-0.5)

bal_800_fluctuations = time_data[3] - np.mean(time_data[3])
#time = time_data[3] / average_signal_800_test
frequencies, psd_balanced_800 = welch(bal_800_fluctuations, fs=fs, nperseg=samples//window)
#psd_balanced_800 = psd_balanced_800 * psd_fitted[::-1]
psd_balanced_800_sqrt = np.sqrt((psd_balanced_800) / ((2*average_signal_800)**2))

# Autobalanced
autobal_400_fluctuations = time_data[2] - np.mean(time_data[2])
frequencies, psd_autobalanced_400 = welch(autobal_400_fluctuations, fs=fs, nperseg=samples//window)
psd_autobalanced_400_sqrt = np.sqrt(psd_autobalanced_400 / ((2*average_signal_400)**2))
#psd_autobalanced_shifted = np.where(frequencies <= 5000, psd_autobalanced * 5, psd_autobalanced)
#psd_autobalanced_shifted_sqrt = np.sqrt(psd_autobalanced_shifted)  # Convert to Hz^(-0.5)
#psd_autobalanced_norm = psd_autobalanced / 
#psd_autobalanced_sqrt_cheated = psd_autobalanced_sqrt * 0.97

autobal_800_fluctuations = time_data[4] - np.mean(time_data[4])
frequencies, psd_autobalanced_800 = welch(autobal_800_fluctuations, fs=fs, nperseg=samples//window)
#psd_autobalanced_shifted = np.where(frequencies <= 5000, psd_autobalanced * 5, psd_autobalanced)
#psd_autobalanced_shifted_sqrt = np.sqrt(psd_autobalanced_shifted)  # Convert to Hz^(-0.5)
#psd_autobalanced_norm = psd_autobalanced / 
psd_autobalanced_800_sqrt = np.sqrt((psd_autobalanced_800) / ((2*average_signal_800)**2))

# Nirvana Dark Noise
dark_fluctuations = time_data[5] - np.mean(time_data[5])
frequencies, psd_dark = welch(dark_fluctuations, fs=fs, nperseg=samples//window)
psd_dark_sqrt = np.sqrt(psd_dark / ((2*average_signal_800)**2))

# Signal
signal_fluctuations_800 = time_data[7] - np.mean(time_data[7])
frequencies, psd_signal = welch(signal_fluctuations_800, fs=fs, nperseg=samples//window)
psd_signal_800_sqrt = np.sqrt(psd_signal / ((2*average_signal_800)**2))

signal_fluctuations_400 = time_data[7] - np.mean(time_data[8])
frequencies, psd_signal_400 = welch(signal_fluctuations_400, fs=fs, nperseg=samples//window)
psd_signal_400_sqrt = np.sqrt(psd_signal_400 / ((2*average_signal_400)**2))

# Osci
frequencies, psd_osci = welch(time_data[6], fs=fs, nperseg=samples//window)
psd_osci_sqrt = np.sqrt(psd_osci)


""" ############# Quick interpolation for gain fitting ###############
xp = [0,2.5e5]
fp = [3.3, 1]
factors = np.interp(frequencies, xp, fp)
print(factors) """

""" ################# Fitting and normalizing gain ################
# **Step 1: Remove Peaks Using a Median Filter**
filtered_psd = medfilt(psd_dark_sqrt, kernel_size=3333)  # Smooth high peaks

# **Step 2: Perform Polynomial Fit**
degree = 5  # Choose polynomial degree (4-6 works well)
coeffs = np.polyfit(frequencies, filtered_psd, degree)  # Fit polynomial
psd_fitted = np.polyval(coeffs, frequencies)  # Evaluate polynomial

# **Step 3: Normalize Fit to 1 at the Middle Frequency**
f_mid = 125000  # Set the middle frequency to 125 kHz
mid_index = np.argmin(np.abs(frequencies - f_mid))  # Find index closest to f_mid
psd_fitted /= psd_fitted[mid_index]  # Normalize to make value at f_mid = 1 """

""" # **Step 3: Normalize Fit to 1 at the Highest Frequency**
psd_fitted /= psd_fitted[-1]  # Normalize so that max frequency value is 1 """


# Compute shot noise level (assuming Poisson statistics)
wavelength = 1064e-9
r = 0.7  # Photodetector responsivity (A/W)
p = 400e-6 # Total optical power (W)
g = 100e3 # V/A
shot_noise_psd_800 =  np.sqrt(2) * (np.sqrt(2 * constants.h * constants.c / (wavelength * (average_signal_800 / (g*r))))) #/ (np.sqrt(1000/400)*np.sqrt(1064/1064))# / average_signal_800_test
shot_noise_psd_400 = np.sqrt(2) * (np.sqrt(2 * constants.h * constants.c / (wavelength * (average_signal_400 / (g*r))))) # p= average_signal_400 / (g*r) #/ (np.sqrt(1000/400)*np.sqrt(1064/1064))# / average_signal_800_test
#shot_noise_psd_800 = (g * np.sqrt(2 * constants.elementary_charge * r * p))/ average_signal_800_test#/ (2.422e-3 * 100e3 * 0.7)# * np.ones_like(frequencies)
#shot_noise_psd_400 =  np.sqrt(2 * constants.elementary_charge * 2 * (average_signal_400 / g))# * np.ones_like(frequencies)
#shot_noise_psd_800 =  (g * np.sqrt(2 * constants.elementary_charge * 2 * (average_signal_800 / g))) #/ average_signal_800_test# * np.ones_like(frequencies)
#shot_noise_psd_400 =  np.sqrt(2 * constants.elementary_charge / (average_signal_400/g))# * np.ones_like(frequencies)
#shot_noise_psd_800 =  np.sqrt(2) * np.sqrt(2 * constants.elementary_charge / (average_signal_800/(1*g)))# np.sqrt(2) in beginning beacuse there are 2 detectors
shot_noise_psd_400_nonrel = (g**2) * 4 * constants.h * constants.c * 150e-6 / wavelength
""" print(average_signal_400 / (g*r))
print(average_signal_800 / (g*r))
print(shot_noise_psd_400)
print(shot_noise_psd_800) """


# Compute cumulative RMS RIN
rin_psd_norm = psd_balanced_400_sqrt**2  # Calculate the RIN PSD (RIN^2) [1/Hz]
rin_psd = psd_balanced_400      # [V^2/Hz]
df = np.diff(frequencies)   # Estimate frequency resolution (Δf)
df = np.append(df, df[-1])  # Same length as frequencies
rin_rms_cum = np.sqrt(np.cumsum(rin_psd * df))   # Cumulative integral for RMS
rin_rms_cum_reverse = np.sqrt(np.cumsum((rin_psd[::-1] * df[::-1])))[::-1]      # Reverse cumulative integration
rin_rms_cum_reverse_microV = 1e6 * rin_rms_cum_reverse
rin_rms_cum_reverse_perc = (rin_rms_cum_reverse / average_signal_400) * 100
""" # Compute integrated RMS RIN using cumulative trapezoidal integration
integrated_rms_rin = np.sqrt(np.maximum(cumulative_trapezoid(psd_balanced_400_sqrt[::-1], frequencies, initial=0), 0))    #np.maximum(x,0) to prevent neg. values for following sqrt """

# Compute Fosn (Factor above shot noise)
fosn = compute_fosn(psd_balanced_400_sqrt, shot_noise_psd_400)

# Compute CMRRs
cmrrs = compute_cmrrs(psd_signal_400_sqrt, psd_balanced_400_sqrt)

########## Plot the results ##########
fig, (axrin, axcum, axfosn, axcmrr) = plt.subplots(4, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1, 1,1]})
#fig.suptitle('Power Spectral Density vs. Shot Noise')
#plt.semilogy(gain_f_norm)
axrin.semilogy(frequencies, psd_balanced_400_sqrt, label='Balanced 200 microW each')
axrin.semilogy(frequencies, psd_autobalanced_400_sqrt, label='Autobalanced 200 microW each')
axrin.semilogy(frequencies, psd_signal_400_sqrt, label='Signal 200 microW each')
#plt.semilogy(frequencies, psd_balanced_800_sqrt, color="red", label='Balanced 400 microW')
#plt.semilogy(frequencies, psd_autobalanced_800_sqrt, label='Autobalanced 400 microW')
#plt.semilogy(frequencies, psd_osci, label='Osci')
#plt.loglog(frequencies, shot_noise_psd, '--', label='Shot Noise Level')
#ax1.axhline(shot_noise_psd_800, color='orange', linestyle='--', label="Shot Noise 800 microW")
axrin.axhline(shot_noise_psd_400, color='lightblue', linestyle='--', label="Shot Noise 400 microW")
axrin.semilogy(frequencies, psd_dark_sqrt, label='Detector Dark Noise')
axrin.set_ylabel('PSD (Hz$^{-0.5}$)')
axrin.grid(True, which='both', linestyle='--', alpha=0.6)
axrin.legend()
axrin.title.set_text('RIN, Cum RMS RIN, FOSN, CMRR')

axcum.plot(frequencies, rin_rms_cum_reverse_perc, label="Integrated RMS RIN 200 microW each")
axcum.set_ylabel('Cumulative RMS RIN [%]')
axcum.grid(True, which='both', linestyle='--', alpha=0.6)
axcum.legend()

axfosn.plot(frequencies, fosn)#(frequencies, integrated_rms_rin, label="Integrated RMS RIN 400 microW each")
axfosn.axhline(1, color='black', linestyle='--', label='Shot Noise')
axfosn.set_ylabel('Factor above shot noise')
axfosn.set_ylim(-1, 5)
axfosn.grid(True, which='major', linestyle='--', alpha=0.6)

axcmrr.plot(frequencies, cmrrs)
axcmrr.set_xlabel('Frequency (Hz)')
axcmrr.set_ylabel('CMRR [dB]')
axcmrr.grid(True, which='major', linestyle='--', alpha=0.6)

""" plt.loglog(frequencies, psd_balanced_400, label='Balanced 400 microW')
plt.axhline(shot_noise_psd_400_nonrel, color='lightblue', linestyle='--', label="Shot Noise 400 microW")
plt.loglog(frequencies, psd_dark, label='Detector Dark Noise') """

#ax2 = ax1.twinx()
#ax2.plot(psd_fitted)

plt.xlim(0,1.25e5)
#plt.title('Power Spectral Density vs. Shot Noise')
plt.show()


""" fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    #fig.suptitle('Laser RIN and Noise Contributions')
    #plt.figure(figsize=(10, 10))
    ax1.semilogx(freq_data[0], freq_data[1], label='Sig', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[3], label='Autobalanced', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[5], label='Autobalanced mod lockin', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[7], label='Balanced', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[9], label='Detector dark noise', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[11], label='Osci Dark Noise', linewidth=0.5)#, alpha= 0.3)
    #ax1.semilogx(freq_data[0], freq_data[11], label='Osci', linewidth=0.5)
    ax1.axhline(rin_shot_dBHz, color='r', linestyle='--', label="Shot Noise Limit")
    #ax1.axvline(3e3, color="black", linewidth= 0.7)
    ax1.set_xlim(left=1, right=100e3)
    ax1.xaxis.grid(visible=True, which='both')
    ax1.yaxis.grid(visible=True, which='major')
    ax1.set_ylabel('RIN (dB/Hz)')
    ax1.title.set_text('Relative Intensity Noise')

    ax2.plot(freq_data[0], freq_data[2], label='Sig', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[4], label='Autobalanced', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[6], label='Autobalanced mod lockin', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[8], label='Balanced', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[10], label='Detector dark noise', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[12], label='Osci Dark Noise', linestyle='dashed')
    #ax2.plot(freq_data[0], freq_data[12], label='Osci', linestyle='dashed')
    ax2.xaxis.grid(visible=True, which='both')
    ax2.yaxis.grid(visible=True, which='major')
    ax2.legend()
    ax2.set_ylabel('RMS RIN')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.title.set_text('Root Mean Square Relative Intensity Noise')

    ax3.plot(freq_data[0], fosn)
    ax3.set_ylim(bottom=0, top=400)
    ax3.set_ylabel('Balancing Factor above Shot Noise')
    ax3.set_xlabel('Frequency (Hz)')

    ax4.plot(freq_data[0], cmrrs)
    #ax3.set_ylim(bottom=0, top=10000)
    ax4.set_ylabel('CMRR')
    ax4.set_xlabel('Frequency (Hz)') """