import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

# Load CSV data
file_nofilter = "E:/Measurements/46/2025-03-24/nofilter.csv"
file_nd10a = "E:/Measurements/46/2025-03-24/nd10a.csv"
file_nd20a = "E:/Measurements/46/2025-03-24/nd20a.csv"
file_nd40a = "E:/Measurements/46/2025-03-24/nd40a.csv"
file_nd40a_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab.csv"
file_nd40a_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne20ab.csv"
file_nd40a_ne10ab_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab-ne10ab.csv"
data = np.array([])

for i in [file_nofilter, file_nd10a, file_nd20a, file_nd40a, file_nd40a_ne10ab, file_nd40a_ne20ab, file_nd40a_ne10ab_ne20ab, file_nd40a_nenir20ac_ne20ab, file_nd40a_nenir20ac_ne20ab_ne10ab]:
    data = np.append(data, np.mean(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

r = 0.67                 # Photodetector responsivity (A/W)
g = 100e3                # V/A
lo_power = 0.1975e-3    # W
data_power = data / (g * r)  # Maybe factor 2 because of Interferogram peak to peak
data_voltage = data
lo_voltage = g * r * lo_power 
transmission = np.array([1, 0.1104, 0.01190, 0.0003510, 0.0003510*0.0791, 0.0003510*0.02482, 0.0003510*0.0791*0.02482, 0.0003510*0.009257*0.02482, 0.0003510*0.009257*0.02482*0.1104])
signal_power = transmission * lo_power
signal_voltage = transmission * lo_voltage
#expected = 4 * np.sqrt(signal_voltage) * np.sqrt(lo_voltage)
x_axis = np.linspace(10, 1e-20, 10000)#[::-1]  # Reverse the array
x_axis_voltage = x_axis * g * r
expected = 4 * np.sqrt(x_axis) * np.sqrt(lo_power) #/ (constants.epsilon_0 * constants.c)
expected_voltage = g * r * expected
#theory = 4 * x_axis * lo_voltage
#theory = 4 * x_axis * lo_voltage
print(4 * np.sqrt(2e-4) * np.sqrt(2e-4))
print(lo_power*4*g*r)

wavelength = 1064e-9
shot_noise = np.sqrt(2 * constants.h * constants.c / (wavelength * 2 * lo_power))
one_photon = constants.h * constants.c / wavelength
v_shot = g * np.sqrt(2 * constants.elementary_charge * lo_power * r)
#print(theory)
# Plot the results
#plt.semilogy(frequencies, psd_balanced_400_sqrt, label='Balanced 400 mW')
#ax1.axhline(shot_noise_psd_800, color='r', linestyle='--', label="Shot Noise 800 microW")
#ax2 = ax1.twinx()
#ax2.plot(psd_fitted)
""" plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (Hz$^{-0.5}$)')
plt.legend()
plt.title('Power Spectral Density vs. Shot Noise')
ax1.grid(True, which='both', linestyle='--', alpha=0.6) """
""" fig, ax1 = plt.subplots(figsize=(8,6))
ax1.loglog(x_axis, theory)
ax1.loglog(signal_power, data)
plt.show() """

####### add labels etc.
plt.figure(figsize=(10, 6))
#plt.loglog(x_axis_voltage, expected_voltage)
plt.loglog(x_axis, expected_voltage, label='g*r*4*Es*ELO')
plt.gca().invert_xaxis()  # Inverts the x-axis
#plt.loglog(signal_voltage, data)
plt.loglog(signal_power, data_voltage, '--bo', label='Measured Signal')
plt.axhline(v_shot, color='r', linestyle='--', label="Shot Noise")
plt.axvline(one_photon, color='black', linestyle='--', label="One Photon/s")
plt.xlim(1e-2, 1e-21)
plt.ylim(1e-7, 2)
plt.xlabel('Signal Arm Power [W]')
plt.ylabel('Measured Signal [V]')
plt.title('BHD Signal vs. Signal Arm Power')
plt.legend()
""" 
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (dBm)")
plt.xlim(0,50e3)
plt.grid()
plt.legend() """
plt.show()