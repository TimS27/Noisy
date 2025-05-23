import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
import glob
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

font_size_label = 18
font_size_title = 20

""" fused_silica = "E:\Older-Measurements/measurements-11-09-24/messreihe3/3-5mm-SiO2-spectrum-5.csv"
no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"
no_fused_silica_stretched = "E:\Older-Measurements/measurements-24-27-08/Stretched_No_Fused_Silica_Spectrum.csv"

data_fused_silica = np.genfromtxt(fused_silica, delimiter=",", skip_header=1)
data_no_fused_silica = np.genfromtxt(no_fused_silica, delimiter=",", skip_header=1)
data_no_fused_silica_stretched = np.genfromtxt(no_fused_silica_stretched, delimiter=",", skip_header=1)

# Fused Silica
wavelength_no_fused_silica = data_no_fused_silica[:,0]
counts_no_fused_silica = data_no_fused_silica[:,1]
counts_no_fused_silica_norm = counts_no_fused_silica / np.max(counts_no_fused_silica)

# Fused Silica stretched
counts_no_fused_silica_stretched = data_no_fused_silica_stretched[:,2]  # old data is in second column, stretched data is in third column!
counts_no_fused_silica_stretched_norm = counts_no_fused_silica_stretched / np.max(counts_no_fused_silica_stretched)

#No Fused Silica
wavelength_fused_silica = data_fused_silica[:,0]
counts_fused_silica = data_fused_silica[:,1]
counts_fused_silica_norm = counts_fused_silica / np.max(counts_fused_silica)

# Plot spectrum
plt.semilogy(wavelength_no_fused_silica, counts_no_fused_silica_stretched_norm, label="Air")
plt.semilogy(wavelength_fused_silica, counts_fused_silica_norm, label="5 mm SiO2")
#plt.semilogy(wavelength_no_fused_silica, counts_no_fused_silica_norm)
plt.xlim(750, 1150)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Normalized Counts")
plt.title("Spectral Broadening in 5 mm Fused Silica")
plt.legend(loc='upper left')
plt.show() """

#################################################

# Folder where the .asc files are stored
folder_path1 = "E:\Older-Measurements/measurements-11-09-24/messreihe2"
folder_path2 = "E:\Older-Measurements/measurements-11-09-24/messreihe3"

# Optional: manually define distances in mm (must match file order!)
distances = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])#, 14, 15])#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#, 16 ,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]#, 29, 30, 31, 32, 33]
step = 1.5
distances_mm = distances * step
distances_mm = distances_mm

# Get sorted list of .asc files (ensure order matches distances!)
files1 = sorted(glob.glob(os.path.join(folder_path1, "*.asc")))
files2 = sorted(glob.glob(os.path.join(folder_path2, "*.asc")))

# === Load spectra ===
spectra = []
wavelength_reference = None

for file in files1:
    df = pd.read_csv(file, delimiter=r"\s+", decimal=",", header=None, names=["Wavelength", "Counts"])
    if wavelength_reference is None:
        wavelength_reference = df["Wavelength"].values
    spectra.append(df["Counts"].values)
spectra = spectra[2:10]  # Shape: (num_files, num_wavelengths)

for file in files2:
    df = pd.read_csv(file, delimiter=r"\s+", decimal=",", header=None, names=["Wavelength", "Counts"])
    if wavelength_reference is None:
        wavelength_reference = df["Wavelength"].values
    spectra.append (df["Counts"].values)
spectra_array = np.array(spectra[:-2])  # Shape: (num_files, num_wavelengths)

# Convert to numpy array
#print(np.max(spectra_array, axis=1))
spectra_array = spectra_array / np.max(spectra_array, axis=1)[:, np.newaxis]  # normalize each spectrum
#spectra_array = spectra_array[::-1]

print(spectra_array.shape)
print(len(distances))
print(wavelength_reference.shape)

spectra_array = spectra_array[::-1]

""" # === Plotting ===
plt.figure(figsize=(10, 6))
plt.pcolormesh(distances_mm, wavelength_reference, spectra_array.T, shading='gouraud', cmap='magma', norm=PowerNorm(gamma=0.35, vmin=1e-3, vmax=1))
plt.xlabel("Distance from focus to plate (mm)", fontsize=20)
plt.ylabel("Wavelength (nm)", fontsize=20)
plt.ylim(850,1100)
plt.title("Spectral Broadening vs Distance in Fused Silica", fontsize=22)
plt.colorbar(label="Normalized intensity", fontsize=22)
plt.tight_layout()
plt.show() """

""" tick_locs = np.linspace(distances_mm[-1], distances_mm[0], 13)  # 13 evenly spaced ticks
plt.xticks(tick_locs, fontsize=16) """
plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(distances_mm, wavelength_reference, spectra_array.T,
                      shading='gouraud', cmap='magma',
                      norm=PowerNorm(gamma=0.35, vmin=1e-3, vmax=1))

plt.xlabel("Distance between laser focus and SiO2 plate (mm)", fontsize=font_size_label)
plt.ylabel("Wavelength (nm)", fontsize=font_size_label)
plt.ylim(850, 1100)
plt.title("Spectral Broadening vs Distance in Fused Silica", fontsize=font_size_title)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Set colorbar and label font size
cbar = plt.colorbar(mesh)
cbar.set_label("Normalized intensity", fontsize=font_size_label)
cbar.ax.tick_params(labelsize=16)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()