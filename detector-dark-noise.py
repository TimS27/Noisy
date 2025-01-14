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
def voltage_to_dbm(voltage, impedance=50):
    power_watts = (voltage**2) / impedance  # Calculate power in watts
    power_dbm = 10 * np.log10(power_watts * 1e3)  # Convert to dBm
    return power_dbm

# Perform FFT and calculate spectrum in dBm
def calculate_spectrum(data, sampling_rate, impedance=50):
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

# Plot the spectrum
def plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2):
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies1, spectrum_dbm1, label="Input 1")
    plt.plot(frequencies2, spectrum_dbm2, label="Balanced inputs")
    plt.title("Spectral Content in dBm")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.xlim(0,10e3)
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
    file_path1 = "balanced-detector-data/balancing/input1-1s-10MS-5Vpp-5kHz-modulation.csv"
    file_path2 = "balanced-detector-data/balancing/balanced-1s-10MS-5Vpp-5kHz-modulation.csv"
    sampling_rate1 = 10e6
    sampling_rate2 = 10e6

    # Load data
    data1 = load_data(file_path1)
    data2 = load_data(file_path2)

    # Calculate spectrum
    frequencies1, spectrum_dbm1 = calculate_spectrum(data1, sampling_rate1)
    frequencies2, spectrum_dbm2 = calculate_spectrum(data2, sampling_rate2)

    # Plot spectrum
    plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2)
    #plot_spectrum(frequencies1, spectrum_dbm1, frequencies2, spectrum_dbm2)
    #plot_spectrum(frequencies2, spectrum_dbm2)

if __name__ == "__main__":
    main()