import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid
from scipy import constants

# Load oscilloscope data (Assumes a two-column format: time, voltage)
def load_data(file_path):
    data = pd.read_csv(file_path, header=0, delimiter=',')  # Read CSV, first row is header
    data = data.astype(float)  # Convert all data to float
    return data

# Compute RIN or noise in dB/Hz
def compute_rin(voltage, sampling_rate, responsivity, trans_gain, mean_power):#, mean_power):
    # swtiching from voltage to optical power not necessary for relative noise
    """ optical_power = voltage / (responsivity * trans_gain)
    
    # Compute intensity fluctuations
    if mean_power == None:
        mean_optical_power = np.mean(optical_power)
    else:
        mean_optical_power = mean_power / (responsivity * trans_gain)
    #relative_intensity = fluctuations / mean_optical_power
    fluctuations = optical_power - mean_optical_power   # δP(t) """
    if mean_power != None:
        mean_voltage = mean_power
    else:
        mean_voltage = np.mean(voltage)
    fluctuations = voltage - mean_voltage  # δP(t)

    # Power Spectral Density (PSD) using Welch’s method
    freqs, psd = welch(fluctuations, fs=sampling_rate, nperseg=2000000)
    
    # Compute RIN: S_I(f) = (2 / P̄²) * S_δP(f)
    rin = psd / (mean_voltage**2) # no factor 2 needed, because welch's method already gives one-sided psd (scaled to include powers from neg and pos freq.)

    rin_dB = 10 * np.log10(rin + 1e-20)
    
    # Compute integrated RMS RIN using cumulative trapezoidal integration
    integrated_rms_rin = np.sqrt(np.maximum(cumulative_trapezoid(rin, freqs, initial=0), 0))    #np.maximum to prevent neg. values for following sqrt

    return freqs, rin_dB, integrated_rms_rin

# Osci or detector noise
def compute_oscilloscope_noise(voltage_noise, sampling_rate, responsivity, trans_gain, mean_power):
    freqs, psd_v = welch(voltage_noise, fs=sampling_rate, nperseg=2000000)

    # Convert voltage noise PSD to equivalent power noise PSD
    psd_p = psd_v / (trans_gain ** 2 * responsivity ** 2)

    # Compute relative oscilloscope noise (normalize by mean optical power squared)
    mean_optical_power = mean_power / (responsivity * trans_gain)
    rin_oscilloscope = psd_p / (mean_optical_power ** 2)

    #rin_oscilloscope = psd_v / mean_power**2
    # Convert to dB/Hz
    rin_oscilloscope_dB = 10 * np.log10(rin_oscilloscope + 1e-20)

    # Compute integrated RMS RIN using cumulative trapezoidal integration
    integrated_rms_rin = np.sqrt(np.maximum(cumulative_trapezoid(rin, freqs, initial=0), 0))

    return freqs, rin_oscilloscope_dB, integrated_rms_rin

def compute_fosn(rin_dBHz, rin_shot_noise_dBHz):
    factor = 10 ** (np.abs(rin_dBHz - rin_shot_noise_dBHz) / 10)
    return np.abs(factor)

def compute_cmrr(rin_unbalanced, rin_balanced):
    cmrrs = np.abs(rin_unbalanced - rin_balanced)# / 10) ** 10
    return cmrrs

# Plot the RIN and noise spectrum
def plot_rin(freq_data, fosn, cmrrs):#, dark_noise_dB, osc_noise_dB):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10,8), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
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
    ax4.set_xlabel('Frequency (Hz)')

    plt.show()

# Example usage
if __name__ == "__main__":

    """ file_det1 = "RIN-data/Mephisto/RIN-osci-noise-eater-off.csv"
    file_det2 = "RIN-data/Mephisto/RIN-osci-noise-eater-on.csv" """

    file_det1 = "E:/Measurements/46/2025-02-28/nirvana-sig-129microW-20MS-100s-200kSs-20kHz-osci.npy"
    #file_det2 = "balanced-detector-data/Koheron/10-02-2025/koheron-det2-unchopped-2,5V-noise-100kSs-1MS.csv"
    file_autobalanced = "E:/Measurements/46/2025-03-05/autobalanced-highres.npy"
    #file_balanced_chopped = "balanced-detector-data/Koheron/10-02-2025/koheron-balanced-modulated-3kHz-ca126microW-noise-100kSs-1MS.csv"
    file_autobalanced_mod_lockin = "E:/Measurements/46/2025-02-28/nirvana-autobal-chopped-5kHz-500mVpp-lockin-sensitivity500microV3s--129microW-20MS-100s-200kSs-20kHz-osci.npy"
    file_balanced = "E:/Measurements/46/2025-02-28/nirvana-bal-194microW-20MS-100s-200kSs-20kHz-osci.npy"
    #file_balanced_chopped = "balanced-detector-data/Koheron/10-02-2025/koheron-balanced-chopped-3kHz-ca126microW-noise-100kSs-1MS.csv"
    #file_balanced_chopped_lockin = "balanced-detector-data/Koheron/10-02-2025/koheron-balanced-chopped-3kHz-ca126microW-lockin-1000xgain-noise-100kSs-1MS.csv"
    file_dark_noise = "E:/Measurements/46/2025-02-28/nirvana-linear-output-dark-noise-20MS-100s-200kSs-20kHz-osci.npy"
    file_osci = "E:/Measurements/46/2025-02-28/osci-20MS-fixed.npy"
    time_data = []
    time_data.append(np.load(file_det1)[:,0])
    for i in [file_det1, file_autobalanced, file_autobalanced_mod_lockin, file_balanced, file_dark_noise, file_osci]:
        time_data.append(np.load(i)[:,1])

    # No offset correction needed, data shows measured value already, offset is only relevant for osci display
    """ # Remove DC offset
    time_data[1] -= 0.2124
    time_data[2] -= 0.2124 """

    """ # Compute mean laser power (proportional to mean intensity)
    mean_power_det1 = np.mean(time_data[1]**2)
    mean_power_det2 = np.mean(time_data[2]**2)
    mean_powers = [mean_power_det1, mean_power_det2]
    #mean_power_balanced = np.mean(time_data[3]**2) """

    responsivity = 0.65
    trans_gain = 39e3

    freq_data = []
    freq, rin, rms_rin = compute_rin(time_data[1], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=None)#, mean_power=mean_power_det1)    #sampling_rate=2e6
    freq_data.append(freq)
    """ for j in np.linspace(1, len(time_data)-1, len(time_data)-1, dtype=int):
        freq, rin = compute_rin(time_data[j], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=None)#, mean_power=mean_powers[j-1])
        freq_data.append(rin) """
    freq, rin, rms_rin = compute_rin(time_data[1], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=None) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_rin(time_data[2], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])*3*6.24) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_rin(time_data[3], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])*3) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_rin(time_data[4], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])*3)
    freq_data.append(rin)
    freq_data.append(rms_rin)
    print(rin)
    freq, rin, rms_rin = compute_rin(time_data[5], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])*3) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_rin(time_data[6], sampling_rate=200e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])*3) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    """ freq, rin, rms_rin = compute_rin(time_data[5], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_oscilloscope_noise(time_data[6], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_oscilloscope_noise(time_data[7], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])) 
    freq_data.append(rin)
    freq_data.append(rms_rin) """

    # Calculate shot noise level
    r = 0.65  # Responsivity in A/W
    laser_wavelength = 1064e-9
    Rf =39e3  # Transimpedance gain in ohms
    #P_avg = np.mean(time_data[1] / (responsivity * trans_gain))
    P_avg = 2.422e-3
    #I_avg = np.mean(time_data[1] / trans_gain)
    #P_avg = 100e-6
    nu = constants.c / laser_wavelength
    #photons = P_avg/(constants.h*nu)
    current = P_avg * r

    """ shot_noise_photons = np.sqrt(photons)
    shot_noise_current = constants.elementary_charge*shot_noise_photons """
    #shot_noise_current=np.sqrt(2 * constants.elementary_charge * current)
    #shot_noise_current = 2 * constants.elementary_charge * current  #np.sqrt(2 * constants.elementary_charge * current)
    #shot_noise_power = shot_noise_current * (trans_gain**2)
    #rin_shot = shot_noise_current / P_avg**2
    #rin_shot_dBHz = 10 * np.log10(rin_shot)
    rin_shot_dBHz = 10 * np.log10(2*constants.elementary_charge/current)
    """ shot_noise_voltage = shot_noise_current*Rf
    # Compute shot noise in dB/Hz
    shot_noise_dBHz = 20 * np.log10(shot_noise_voltage)
 """

    """ laser_file = "oscilloscope_data.csv"  # Replace with actual file
    detector_dark_file = "detector_dark_noise.csv"  # Replace with actual file
    osc_dark_file = "oscilloscope_dark_noise.csv"  # Replace with actual file
    
    time_laser, voltage_laser = load_data(laser_file)
    time_detector, voltage_detector = load_data(detector_dark_file)
    time_osc, voltage_osc = load_data(osc_dark_file)
    
    # Estimate sampling rate (assume same for all datasets)
    sampling_rate = 1 / np.mean(np.diff(time_laser)) """
    
    
    
    """ # Compute RIN and noise spectra
    freqs, rin_dB = compute_rin(time_laser, voltage_laser, sampling_rate, mean_power)
    _, dark_noise_dB = compute_rin(time_detector, voltage_detector, sampling_rate, mean_power)
    _, osc_noise_dB = compute_rin(time_osc, voltage_osc, sampling_rate, mean_power) """
    
    # Compute CMRRs
    cmrrs = compute_cmrr(freq_data[1], freq_data[3])

    # Compute factor above shot noise
    fosn = compute_fosn(freq_data[3], rin_shot_dBHz)

    # Plot all curves
    plot_rin(freq_data, fosn, cmrrs)