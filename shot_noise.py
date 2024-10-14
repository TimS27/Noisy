import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

size = 100000

power_LO = 1e-3
measurement_time = 1
photon_energy = constants.h * constants.c / 1e-6
photons_LO = power_LO * measurement_time / photon_energy

power_NEP = 12.6e-12 * np.sqrt(measurement_time ** -1)
photons_NEP = power_NEP * measurement_time / photon_energy

print(photons_NEP)
print(photons_LO)

#for N in [0,photons_NEP]:
for N in [1000]:
  for S in [1, 100, 1000]:
  #for S in [100]:
    Ls = [0, 1, 100, 1000, 10000, 100000, 1000000000000,10000000000000,1e15,1e18, ]
    #Ls = [0, 1, 100,100000,10000000000000,1e15,1e18, 1e23]
    snrs = []
    for L in Ls:
      #S = 1
      #L = 1000
      #N = 1000 # photons per measurement time from noise-equivalent power
      #N = 0 # photons per measurement time from noise-equivalent power
      #N = photons_NEP
      
      #photons_arm1 = np.array([], dtype=(np.float64))
      #photons_arm2 = np.array([], dtype=(np.float64))
      photons_arm1 = np.random.poisson(((np.sqrt(L/2) + np.sqrt(S/2))**2), size=size)   #.astype(np.float64)
      photons_arm2 = np.random.poisson(((np.sqrt(L/2) - np.sqrt(S/2))**2), size=size)
      
      print(type(photons_arm1))
      photocurrent_arm1 = photons_arm1 + np.random.normal(scale=N, size=size)
      photocurrent_arm2 = photons_arm2 + np.random.normal(scale=N, size=size)
      
      signal = np.mean(photocurrent_arm1-photocurrent_arm2)
      noise = np.std(photocurrent_arm1-photocurrent_arm2)
      
      snr = signal/noise
      snrs.append(snr)
      
      print(S,L,snr, 2*np.sqrt(S)) # why 2*?
  
    plt.plot(Ls, snrs, label="{} signal photons".format(S))
    #plt.plot(Ls, snrs, label="{} NEP photons".format(N))

plt.yscale("log")
plt.xscale("log")
plt.ylabel("SNR")
plt.xlabel("LO photons")
plt.legend()
plt.axvline(photons_LO)
plt.axhline(2*np.sqrt(S))
plt.show()