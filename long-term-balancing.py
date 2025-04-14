import numpy as np
import matplotlib.pyplot as plt
#from scipy import constants

""" ###### Load CSV data #####
file_400_balanced = "E:/Measurements/46/2025-04-07/400microW-balanced-rin-20s-5MS.c"
file_400_autobalanced = "E:/Measurements/46/2025-04-07/400microW+nd03abinsignal-autobalanced-rin-20s-5MS.csv" """

data_400 = [0,0,0,0,0,0]
data_400[0] = np.load("E:/Measurements/46/2025-04-07/balancing-drift-1000s.npy")[:,0]
data_400[1] = np.load("E:/Measurements/46/2025-04-07/balancing-drift-1000s.npy")[:,1]
data_400[2] = np.load("E:/Measurements/46/2025-04-07/autobalancing-drift-1000s.npy")[:,0]
data_400[3] = np.load("E:/Measurements/46/2025-04-07/autobalancing-drift-1000s.npy")[:,1]
data_400[4] = np.load("E:/Measurements/46/2025-04-07/balancing-drift-4000s.npy")[:,0]
data_400[5] = np.load("E:/Measurements/46/2025-04-07/balancing-drift-4000s.npy")[:,1]
""" for i in [file_400_balanced]:#, file_400_autobalanced]:
    data_400 = np.append(data_400, np.load("E:/Measurements/46/2025-04-07/400microW-balanced-rin-20s-5MS.npy"))
    data_400 = np.append(data_400, np.genfromtxt(i, delimiter=",", skip_header=21)[:,0])
    data_400 = np.append(data_400, np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]) """
#print(data_400[0])
fig = plt.subplots(figsize=(8,6))
#plt.plot(data_400[0], data_400[1])
plt.plot(data_400[2], data_400[3], label="Autobalanced")
plt.plot(data_400[4], data_400[5], label="Balanced")
plt.xlabel("Time [s]")
plt.ylabel("Signal [V]")
plt.title("Balancing Stability")
plt.legend()
plt.show()