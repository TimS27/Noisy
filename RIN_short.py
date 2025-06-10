import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch
from scipy.integrate import cumulative_trapezoid
from scipy import constants

# Load oscilloscope data (Assumes a two-column format: time, voltage)
def load_data(file_path):
    data = pd.read_csv(file_path, header=0, delimiter=',')  # Read CSV, first row is header (you can put "time,voltage" as header)
    data = data.astype(float)  # Convert all data to float
    return data

# Compute RIN or noise in dB/Hz
def compute_rin(voltage, sampling_rate, responsivity, trans_gain):
    # swtiching from voltage to optical power not necessary, but makes procedure more clear
    optical_power = voltage / (responsivity * trans_gain)
    
    # Compute intensity fluctuations
    mean_optical_power = np.mean(optical_power)
    
    # δP(t)
    fluctuations = optical_power - mean_optical_power

    # Power Spectral Density (PSD) using Welch’s method
    freqs, psd = welch(fluctuations, fs=sampling_rate, nperseg=sampling_rate)  # put the window size here, maybe try 1/4 or 1/2 of the sample number
    
    # Compute RIN: S_I(f) = (2 / P̄²) * S_δP(f)
    rin = psd / (mean_optical_power**2) # no factor 2 needed, because welch's method already gives one-sided psd (scaled to include powers from neg and pos freq.)

    rin_dB = 10 * np.log10(rin)
    
    # Compute integrated RMS RIN using cumulative trapezoidal integration
    integrated_rms_rin = np.sqrt(np.maximum(cumulative_trapezoid(rin, freqs, initial=0), 0))    #np.maximum to prevent neg. values for following sqrt

    return freqs, rin_dB, integrated_rms_rin

# Osci or detector noise (takes mean_power as a parameter, so relative noise value can be calculated)
def compute_noise(voltage_noise, sampling_rate, responsivity, trans_gain, mean_power):
    freqs, psd_v = welch(voltage_noise, fs=sampling_rate, nperseg=sampling_rate)

    # Convert voltage noise PSD to equivalent power noise PSD
    psd_p = psd_v / (trans_gain ** 2 * responsivity ** 2)

    # Compute relative oscilloscope noise (normalize by mean optical power squared)
    mean_optical_power = mean_power / (responsivity * trans_gain)
    rin_oscilloscope = psd_p / (mean_optical_power ** 2)

    # Convert to dB/Hz
    rin_oscilloscope_dB = 10 * np.log10(rin_oscilloscope + 1e-20)

    # Compute integrated RMS RIN using cumulative trapezoidal integration
    integrated_rms_rin = np.sqrt(np.maximum(cumulative_trapezoid(rin, freqs, initial=0), 0))

    return freqs, rin_oscilloscope_dB, integrated_rms_rin

# Plot the RIN and noise spectrum
def plot_rin(freq_data):
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10,8))
    ax1.semilogx(freq_data[0], freq_data[1], label='RIN curve', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[3], label='Dark noise', linewidth=0.5)
    ax1.semilogx(freq_data[0], freq_data[5], label='Osci', linewidth=0.5)
    ax1.axhline(rin_shot_dBHz, color='r', linestyle='--', label="Shot Noise Limit")
    ax1.xaxis.grid(visible=True, which='both')
    ax1.yaxis.grid(visible=True, which='major')
    ax1.set_ylabel('RIN (dB/Hz)')
    ax1.title.set_text('Relative Intensity Noise')

    ax2.plot(freq_data[0], freq_data[2], label='RIN curve', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[4], label='Dark noise', linestyle='dashed')
    ax2.plot(freq_data[0], freq_data[6], label='Osci', linestyle='dashed')
    ax2.xaxis.grid(visible=True, which='both')
    ax2.yaxis.grid(visible=True, which='major')
    ax2.legend()
    ax2.set_ylabel('RMS RIN')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.title.set_text('Root Mean Square Relative Intensity Noise')
    plt.show()

# Example usage
if __name__ == "__main__":

    # Specify file paths
    file_det1 = "balanced-detector-data/Koheron/10-02-2025/koheron-det1-unchopped-2,5V-noise-100kSs-1MS.csv"
    file_background = "balanced-detector-data/Koheron/10-02-2025/koheron-dark-noise-100kSs-1MS.csv"
    file_osci = "osci-data/osci-dark-noise-100kSs-1MS.csv"

    # Loads time domain data into one array
    time_data = []
    time_data.append(load_data(file_det1)['time'].values)
    freq_data = []
    for i in [file_det1, file_background, file_osci]:
        time_data.append(load_data(i)['voltage'].values)

    # Calculates resulting frequency axis and RIN + RMS RIN values
    # Make sure to put sampling_rate, responsivity, and trans_gain
    freq, rin, rms_rin = compute_rin(time_data[1], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3)
    freq_data.append(freq)
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_noise(time_data[2], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    freq, rin, rms_rin = compute_noise(time_data[3], sampling_rate=100e3, responsivity=0.65, trans_gain=39e3, mean_power=np.mean(time_data[1])) 
    freq_data.append(rin)
    freq_data.append(rms_rin)
    

    # Calculate shot noise level
    trans_gain = 39e3 # Transimpedance gain in V/A
    responsivity = 0.65
    P_avg = np.mean(time_data[1] / (responsivity * trans_gain)) #not needed
    I_avg = np.mean(time_data[1] / trans_gain)
    rin_shot_dBHz = 10 * np.log10(2*constants.elementary_charge/I_avg)

    # Plot
    plot_rin(freq_data)