import pandas as pd
import numpy as np

""" # Load CSV data
df = pd.read_csv("E:/Measurements/46/2025-02-28/nirvana-bal-194microW-20MS-100s-200kSs-20kHz-osci.csv")

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
data = np.genfromtxt("E:/Measurements/46/2025-03-14/bal-800-2.csv", delimiter=",", skip_header=21)

# Save as NumPy binary
np.save("E:/Measurements/46/2025-03-14/bal-800-2.npy", data)

#################################################

""" # Load .npy file
data = np.load("E:/Measurements/46/2025-02-28/data.npy")

print(data)  # Display first 5 rows """

