import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

size = 100000

power_LO = 1e-3 # [W]
measurement_time = 1  # [s]
photon_energy = constants.h * constants.c / 1e-6  # Energy of 1000 nm photons
photon_flux = power_LO / photon_energy  # [W/s]
photons_LO = photon_flux * measurement_time # LO photon number

power_NEP = 12.6e-12 * np.sqrt(measurement_time ** -1)  # [NEP] = [W/sqrt(Hz)]
photons_NEP = power_NEP * measurement_time / photon_energy

print('Photons NEP: ', photons_NEP, 'W')
print('Photons LO: ', photons_LO)

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('SNR vs. # Local oscillator photons in Balanced Detection')

Ns = [1000, photons_NEP]  # Different numbers of NEP photons

#for N in [0,photons_NEP]:
for N in Ns:  # photons per measurement time from noise-equivalent power
  for S in [1, 100, 1000]:  # Different numbers of signal photons
  #for S in [100]:
    Ls = [0, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, ] # Different numbers of LO photons
    #Ls = [0, 1, 100,100000,10000000000000,1e15,1e18, 1e23]
    snrs = []
    for L in Ls:
      #S = 1
      #L = 1000
      #N = 1000 # photons per measurement time from noise-equivalent power
      #N = 0 # photons per measurement time from noise-equivalent power
      #N = photons_NEP
      
      if L < 10000:
        # One arm shows constructive, one arm shows destructive interference, depending on phase
        photons_arm1 = np.random.poisson(((np.sqrt(L/2) + np.sqrt(S/2))**2), size=size) # L/2 and S/2 photons in each arm, amplitude of the field is sqrt(photon number), fields interfere, **2 again because photon number is proportional to intensity
        photons_arm2 = np.random.poisson(((np.sqrt(L/2) - np.sqrt(S/2))**2), size=size)

      else:
        photons_arm1 = np.random.normal(((np.sqrt(L/2) + np.sqrt(S/2))**2), np.sqrt((np.sqrt(L/2) + np.sqrt(S/2))**2), size)
        photons_arm2 = np.random.normal(((np.sqrt(L/2) - np.sqrt(S/2))**2), np.sqrt((np.sqrt(L/2) + np.sqrt(S/2))**2), size)

      #print(type(photons_arm1))
      photocurrent_arm1 = photons_arm1 + np.random.normal(scale=N, size=size)
      photocurrent_arm2 = photons_arm2 + np.random.normal(scale=N, size=size)
      
      signal = np.mean(photocurrent_arm1-photocurrent_arm2)
      noise = np.std(photocurrent_arm1-photocurrent_arm2)
      
      snr = signal/noise
      snrs.append(snr)
      
      print(S,L,snr, 2*np.sqrt(S)) # Factor 2 because in homodyne detection, both the positive and negative frequency components of the signal field contribute coherently, effectively doubling the detected signal strength compared to the shot noise limit

    if N == Ns[0]:
      ax1.plot(Ls, snrs, label="{} signal photons".format(S))
    else:
      ax2.plot(Ls, snrs, label="{} signal photons".format(S))

    #plt.plot(Ls, snrs, label="{} NEP photons".format(N))


#fig.suptitle('Horizontally stacked subplots')
fig.set_size_inches(15,8)

ax1.set_title("NEP = 1000 Photons / Measurement time")
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylabel("SNR")
ax1.set_xlabel("LO photons")
ax1.legend()
ax1.axvline(photons_LO)
ax1.text(photons_LO*5, 0.1,"LO Power = 1 mW", verticalalignment='center', rotation=90)
ax1.axhline(2*np.sqrt(S))
ax1.text(1e2, 80, r'$2 \sqrt{<n_S>}$', verticalalignment='center')

ax2.set_title("NEP = 63,429,869 Photons / Measurement time")
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylabel("SNR")
ax2.set_xlabel("LO photons")
ax2.legend()
ax2.axvline(photons_LO)
ax2.text(photons_LO*5, 0.1,"LO Power = 1 mW", verticalalignment='center', rotation=90)
ax2.axhline(2*np.sqrt(S))
ax2.text(1e2, 80, r'$2 \sqrt{<n_S>}$', verticalalignment='center')
plt.show()