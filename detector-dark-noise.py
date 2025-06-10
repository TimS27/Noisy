import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import savgol_filter

# Load time-domain data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path, header=0)  # Read CSV assuming first row is header
    data = data.astype(float)  # Convert all data to float
    return data

# Function to calculate the power in dBm
def voltage_to_dbm(voltage, impedance=10e6):
    power_watts = (voltage**2) / impedance  # Calculate power in watts
    power_dbm = 10 * np.log10(power_watts * 1e3)  # Convert to dBm
    return power_dbm
'''
# Perform FFT and calculate spectrum in dBm
def calculate_spectrum(data, sampling_rate, impedance=10e6):
    voltage = data["voltage"].values
    n = len(voltage)

    # Perform FFT
    fft_values = fft(voltage)
    fft_magnitude = np.abs(fft_values)[:n // 2] / n  # Single-sided spectrum

    # Convert to dBm
    spectrum_dbm = voltage_to_dbm(fft_magnitude, impedance)

    # Frequency axis
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)[:n // 2]

    return frequencies, spectrum_dbm
'''
# Perform FFT and calculate spectrum in dBm/√Hz
def calculate_spectrum_dbm_sqrt_hz(data, sampling_rate, impedance=10e6):
    voltage = data["voltage"].values
    n = len(voltage)

    # Perform FFT
    fft_values = fft(voltage)
    fft_magnitude = np.abs(fft_values)[:n // 2]  # Single-sided spectrum

    # Compute frequency resolution
    freq_res = sampling_rate / n  # Bin width in Hz

    # Convert FFT magnitude to voltage noise density in V/√Hz
    voltage_noise_density = fft_magnitude / np.sqrt(n * freq_res)

    # Convert to Power Spectral Density in W/√Hz
    power_density_w_sqrt_hz = (voltage_noise_density ** 2) / impedance

    # Convert to dBm/√Hz
    spectrum_dbm_sqrt_hz = 10 * np.log10(power_density_w_sqrt_hz * 1e3)

    # Frequency axis
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)[:n // 2]

    return frequencies, spectrum_dbm_sqrt_hz
'''
# Perform FFT and calculate spectrum in dBc/√Hz
def calculate_spectrum_dbc_sqrt_hz(data, sampling_rate, carrier_freq=1000, impedance=10e6):
    voltage = data["voltage"].values
    n = len(voltage)

    # Perform FFT
    fft_values = fft(voltage)
    fft_magnitude = np.abs(fft_values)[:n // 2]  # Single-sided spectrum

    # Compute frequency resolution
    freq_res = sampling_rate / n  # Bin width in Hz

    # Convert FFT magnitude to voltage noise density in V/√Hz
    voltage_noise_density = fft_magnitude / np.sqrt(n * freq_res)

    # Convert to Power Spectral Density in W/√Hz
    power_density_w_sqrt_hz = (voltage_noise_density ** 2) / impedance

    # Convert to dBm/√Hz
    spectrum_dbm_sqrt_hz = 10 * np.log10(power_density_w_sqrt_hz * 1e3)

    # Frequency axis
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)[:n // 2]

    # Find the carrier frequency index
    carrier_idx = np.argmin(np.abs(frequencies - carrier_freq))
    carrier_power_dbm = spectrum_dbm_sqrt_hz[carrier_idx]  # Carrier Power

    # Convert to dBc/√Hz (relative to carrier)
    spectrum_dbc_sqrt_hz = spectrum_dbm_sqrt_hz - carrier_power_dbm

    return frequencies, spectrum_dbc_sqrt_hz
'''
# Plot the spectrum
def plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2):
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies1, spectrum_dbm1, label="Input 1")
    plt.plot(frequencies2, spectrum_dbm2, label="Balanced inputs")
    plt.title("Spectral Content in dBm")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.xlim(0,50e3)
    plt.grid()
    plt.legend()
    plt.show()

# Plot the smoothed spectrum
def plot_smoothed_spectrum(frequencies, spectrum_dbm, window_length=51, polyorder=3):
    smoothed_spectrum = savgol_filter(spectrum_dbm, window_length, polyorder)
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, spectrum_dbm, label="Raw Spectrum", alpha=0.5)
    plt.plot(frequencies, smoothed_spectrum, label="Smoothed Spectrum", linewidth=2)
    plt.title("Smoothed Spectral Content in dBm")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.grid()
    plt.legend()
    plt.show()

# Main workflow
def main():
    #print(load_data)
    # Specify file path and sampling rate
    #file_path1 = "balanced-detector-data/dark-noise/dark-noise-off-800microS-1MS.csv"
    #file_path2 = "balanced-detector-data/dark-noise/dark-noise-on-800microS-1MS.csv"
    #sampling_rate1 = 1.25e9 # 1e6 / 800e-6
    #sampling_rate2 = 1.25e9
    #file_path1 = "balanced-detector-data/dark-noise/dark-noise-off-1s-10MS.csv"
    #file_path2 = "balanced-detector-data/dark-noise/dark-noise-on-1s-10MS.csv"
    #sampling_rate1 = 10e6
    #sampling_rate2 = 10e6
    file_path1 = "balanced-detector-data/koheron/koheron-det1-chopped-800Hz-rin-highres-100kSs-5MS.csv"
    file_path2 = "balanced-detector-data/koheron/koheron-balanced-chopped-800Hz-rin-highres-100kSs-5MS.csv"
    sampling_rate1 = 100e3
    sampling_rate2 = 100e3

    # Load data
    data1 = load_data(file_path1)
    data2 = load_data(file_path2)

    # Calculate spectrum
    frequencies1, spectrum_dbm1 = calculate_spectrum_dbm_sqrt_hz(data1, sampling_rate1)
    frequencies2, spectrum_dbm2 = calculate_spectrum_dbm_sqrt_hz(data2, sampling_rate2)

    # Plot spectrum
    plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2)
    #plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2)
    #plot_spectrum(frequencies2, spectrum_dbm2)

if __name__ == "__main__":
    main()