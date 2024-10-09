import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading CSV file
data = pd.read_csv("photodiode_measured_electronic_power_spectral_density.csv")
 
# Converting column data to list then array
frequencies = np.array(data['[Hz]'].tolist())
measured_electronic_power_spectral_density = np.array(data['Trace1[dBm]'].tolist())
measured_electronic_power_spectral_density_corrected = measured_electronic_power_spectral_density + 19  # measurements were with -19 dBm reference level


responsivity = 1.04 # [A/W]
nep = 1.2e-11       # [W/sqrt(Hz)]

# noise equivalent electronic amplitude
neea = nep * responsivity

# square neea and use P=R*I^2
r = 50 # [Ohm]
electronic_power_spectral_density = (neea) * r
print(electronic_power_spectral_density)
# calculate electronic power spectral density in dBm
electronic_power_spectral_density_dBm = 10 * np.log10(electronic_power_spectral_density) + 30

f = np.linspace(0, 1600000000, 1600)
electronic_power_spectral_density_dBm_array = np.full(1600, electronic_power_spectral_density_dBm)
#print(electronic_power_spectral_density_array)

plt.figure()
plt.plot(f, electronic_power_spectral_density_dBm_array)  # 'o', markersize=2
plt.plot(frequencies, measured_electronic_power_spectral_density_corrected)
#plt.xlim(-1e-12, 1e-12)
#plt.title('Time domain electric field')
#plt.xlabel('Time [s]')
#plt.ylabel('Electric field')
plt.show()

# FEHLER IN MAXIS EINTRAG: HAT NUR EINHEITEN QUADRIERT, NICHT WERTE BEI 1e-10 A^2/Hz