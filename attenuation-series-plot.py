import numpy as np
#from isfread_py3 import isfread
import matplotlib.pyplot as plt
#from scipy import signal
#from scipy.optimize import curve_fit
#from scipy.signal import hilbert
#from scipy.signal import butter, filtfilt
import matplotlib.colors as colors
#import fourioso
from scipy import constants

wavelength = 1064e-9
g = 100e3
r = 0.67

nenir20ac = 0.0092574
nenir40ac = 0.0000585
nd40a = 0.000351
ne20ab = 0.024822
ne10ab = 0.0790946

simulated_6bit = np.load('E:\Measurements/46/2025-05-22/simulated_6bit_.npy')
simulated_nonoise = np.load('E:\Measurements/46/2025-05-22/simulated_no_noise.npy')
simulated_6bitNEP20pW = np.load('E:\Measurements/46/2025-05-22/simulated_6bitNEP20pW.npy')
simulated_onlyshot = np.load('E:\Measurements/46/2025-05-22/simulated_only_shot.npy')

series = [0.013514934087822695, 0.0002877861220757857, 0.00013562762918902484, 0.00010961890634659422, 0.0001082965896049411]
#print(series_simulated)
gain = 100e3
lo_power = 500e-6
transmission = np.array([nenir40ac, nenir40ac*nd40a, nenir40ac*nd40a*nenir20ac, nenir40ac*nd40a*nenir20ac*ne10ab, nenir40ac*nd40a*nenir20ac*ne20ab])
signal_power = np.array(transmissions_simulated) * lo_power
#print(transmission)
#v_shot = g * np.sqrt(2 * constants.elementary_charge * lo_power * r)    # 1*lo_power because shot_noise level is relevant at low signal arm powers only
#photons_shot = v_shot / (constants.h * (constants.c/wavelength))
one_photon = constants.h * constants.c / wavelength
number_photons_shot = 5 * 0.05 * lo_power / one_photon
number_photons_shot_uncertainty = np.sqrt(number_photons_shot)
v_shot_rms = np.sqrt(2*constants.elementary_charge * lo_power * r * (gain**2) * 125e3)

E_photon = constants.h * constants.c/wavelength
# Function to convert Power (W) to Photons per second
def power_to_photons(P):
    return P / E_photon

fig, ax1 = plt.subplots(figsize=(8,6))
#plt.loglog(x_axis_voltage, expected_voltage)
#plt.loglog(x_axis, expected_voltage, label='g*r*4*Es*ELO')
#plt.loglog(x_axis, y_fit, label="Balanced Fit")
plt.gca().invert_xaxis()  # Inverts the x-axis
#plt.loglog(signal_voltage, data)
plt.loglog(signal_power, series, 'o', color="green", label='Simulated Signal Balanced (Only Shot Noise)')# (Linear Output)') #'--o'
plt.loglog(signal_power, simulated_onlyshot, 'o', color="green", label='Simulated Signal Balanced (Only Shot Noise)')# (Linear Output)') #'--o'
#plt.loglog(a,b, '--o',color="green", label='test')# (Linear Output)')
#plt.loglog(signal_power_400_autobal, data_autobal_400, '--o',color="blue" , label='Measured Signal Autobalanced')# (Log Output)')
#plt.axhline(v_shot_rms, color='r', linestyle='--', label="Shot Noise")
plt.axhline(2.39e-5, color='r', linestyle='--', label="Shot Noise")
plt.axvline(one_photon, color='black', linestyle='--', label="One Photon/s")
plt.xlim(1e-2, 1e-21)
plt.ylim(1e-7, 100)
plt.xlabel('Signal Arm Power [W]')
plt.ylabel('Measured Signal [V]')
plt.title('BHD Signal vs. Signal Arm Power')
plt.legend(loc='lower left')#, bbox_to_anchor=(0,0.15))
ax1.xaxis.grid(visible=True, which='both')
ax1.yaxis.grid(visible=True, which='major')
#ax1.axvspan(signal_power_400_bal[0], 1e-12, alpha=0.1, color='green')

ax2 = ax1.secondary_xaxis("top", functions=(power_to_photons, lambda N: N * E_photon))  # transform function and its inverse
ax2.set_xlabel("Signal Arm [Photons/s]")

plt.show()