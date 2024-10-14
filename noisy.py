import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading CSV file
data = pd.read_csv("14102024-DANL-with-photodiode-on-blocked-1MHz-RBW.csv")
 
# Converting column data to list then array
frequencies = np.array(data['[Hz]'].tolist())
measured_electronic_power_spectral_density = np.array(data['Trace1[dBm]'].tolist())
#measured_electronic_power_spectral_density_corrected = measured_electronic_power_spectral_density  # measurements were with -19 dBm reference level

responsivity = 1    # [A/W] (actually wavelength dependend, peak responsivity at is 1.04 for ThorLabs PDA10CF(-EC))
nep = 1.2e-11       # [W/sqrt(Hz)] optical power
gain = 1e4          # [V/A]
r = 50              # [Ohm]

# noise equivalent electronic amplitude
neea = nep * responsivity * gain    # [V/sqrt(Hz)]

# square neea and use P=R*I^2
electronic_power_spectral_density = (neea ** 2) / r # [W/Hz]

# calculate electronic power spectral density per 10 kHz = RBW
electronic_power_spectral_density_per_10kHz = electronic_power_spectral_density * 1e6

# calculate electronic power spectral density in dBm
electronic_power_spectral_density_per_10kHz_dBm = 10 * np.log10(electronic_power_spectral_density_per_10kHz * 1000)

print('The NEP-calculated electronic power spectral density per 10 kHz is: ', electronic_power_spectral_density_per_10kHz_dBm, ' dBm')


# Make const. array to display NEP-calculated PSD
f = np.linspace(0, 1600000000, 1600)
electronic_power_spectral_density_per_10kHz_dBm_array = np.full(1600, electronic_power_spectral_density_per_10kHz_dBm)


# Plot Measured PSD vs. NEP calculated PSD
plt.figure()
plt.plot(f, electronic_power_spectral_density_per_10kHz_dBm_array)
plt.plot(frequencies, measured_electronic_power_spectral_density)
plt.xlim(1e5, 1.6e9)
plt.title('Measured PSD vs. NEP-calculated PSD @ 10 kHz RWB')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Electronic power spectral density [dBm]')
plt.legend(['NEP-calculated PSD', 'Measured PSD'])
plt.show()