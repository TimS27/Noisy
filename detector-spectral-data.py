import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data1 = pd.read_csv('balanced-detector-data/dark-noise/dark-noise-off-ESA.csv')
data2 = pd.read_csv('balanced-detector-data/dark-noise/dark-noise-on-ESA.csv')

# Converting column data to list then array
frequency_Hz1 = np.array(data1['[Hz]'].tolist())
power_dBm1 = np.array(data1['Trace1[dBm]'].tolist())
frequency_Hz2 = np.array(data2['[Hz]'].tolist())
power_dBm2 = np.array(data2['Trace1[dBm]'].tolist())


def plot_data(frequency1, power1, frequency2, power2):
    """
    Plots spectral noise data.

    Parameters:
        frequency (pd.Series): Frequency data in Hz.
        power (pd.Series): Power data in dBm.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frequency1, power1, label="Spectral noise detector off")#, color="blue")
    plt.plot(frequency2, power2, label="Spectral noise detector on")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dBm)")
    plt.title("Spectral Noise Data")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    #plt.tight_layout()
    plt.show()

plot_data(frequency_Hz1, power_dBm1, frequency_Hz2, power_dBm2)