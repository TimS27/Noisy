import pandas as pd
import numpy as np
from isfread_py3 import isfread

""" # Load CSV data
file_nofilter = "E:/Measurements/46/2025-03-24/nofilter.csv"
file_nd10a = "E:/Measurements/46/2025-03-24/nd10a.csv"
file_nd20a = "E:/Measurements/46/2025-03-24/nd20a.csv"
file_nd40a = "E:/Measurements/46/2025-03-24/nd40a.csv"
file_nd40a_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab.csv"
file_nd40a_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne20ab.csv"
file_nd40a_ne10ab_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-ne10ab-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab.csv"
file_nd40a_nenir20ac_ne20ab_ne10ab = "E:/Measurements/46/2025-03-24/nd40a-nenir20ac-ne20ab-ne10ab.csv"
data = []

for i in [file_nofilter, file_nd10a, file_nd20a, file_nd40a, file_nd40a_ne10ab, file_nd40a_ne20ab, file_nd40a_ne10ab_ne20ab, file_nd40a_nenir20ac_ne20ab, file_nd40a_nenir20ac_ne20ab_ne10ab]:
    data.append(np.mean(np.genfromtxt(i, delimiter=",", skip_header=21)[:,1]))

print(data)
#print(np.genfromtxt(file_nofilter, delimiter=",", skip_header=21)[:,1]) """

""" df = pd.read_csv("E:/Measurements/46/2025-02-28/nirvana-bal-194microW-20MS-100s-200kSs-20kHz-osci.csv")

# Convert 'inf' strings to NaN for processing
df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")

# Compute the mean excluding NaN values
mean_voltage = df["voltage"].mean()

# Replace NaN values (previously 'inf') with the mean
df["voltage"].fillna(mean_voltage, inplace=True)

# Save the cleaned data back to a CSV file
df.to_csv("E:/Measurements/46/2025-02-28/nirvana-bal-194microW-20MS-100s-200kSs-20kHz-osci-fixed2", index=False) """

#################################################

# Load CSV data
data = np.genfromtxt("E:/Measurements/46/2025-04-29/rin-400microW-LO-signal.csv", delimiter=",", skip_header=21)

# Save as NumPy binary
np.save("E:/Measurements/46/2025-04-29/rin-400microW-LO-signal.npy", data)

#################################################

""" # Load .npy file
data = np.load("E:/Measurements/46/2025-04-29/rin-400microW-LO-bal-2.npy")

print(data)  # Display first 5 rows """

#################################################

""" # Read .isf data
data, header = isfread('E:\Measurements/46/2025-05-08/500microWLO-1600microW-delay-20MS-40s-nd40a-nenir40ac-nenir20ac-calibration.isf')

# Save as NumPy binary
np.save("E:\Measurements/46/2025-05-08/500microWLO-1600microW-delay-20MS-40s-nd40a-nenir40ac-nenir20ac-calibration.npy", data) """
