import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import pandas as pd

import glob
import os
from matplotlib.colors import PowerNorm

font_size_label = 18
font_size_title = 20

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

#filepath = "E:\Measurements/46/2025-06-05/slurm_supercontinuum_fused_silica_v003/slurm_supercontinuum_fused_silica_v003_Ext.dat"
#filepath = "E:\Measurements/46/2025-06-06/slurm_supercontinuum_fused_silica_v004/slurm_supercontinuum_fused_silica_v004_Ext.dat"
""" filepath = "E:\Measurements/46/2025-06-10/slurm_supercontinuum_fused_silica_350_950_v007/slurm_supercontinuum_fused_silica_550_700_v007_Ext.dat"
data = np.fromfile(filepath, dtype=np.float64)
print(data.size) """

""" # Create a memmap file (will be on disk instead of in RAM)
shape = (1310720000,)
dtype = np.float64
large_data = np.memmap(filepath, dtype=dtype, mode='w+', shape=shape) """

#NB=4    # batch size
NT=8192 # Fourier number
NX=20000
NP=2

def extract_spectrum(data, selector):
    E_field = data.reshape((NB,NP,NX,NT)).astype(np.float32)
    del data

    dX=0.1 # um

    X =  dX* np.arange(-(NX-1)/2, (NX-1)/2 + 1, 1)

    dt = 0.5e-15 # s
    t = np.arange(NT) * dt 


    Z = np.squeeze(E_field[selector, 0, :, :])

    #=============================================


    extent = [X.min() , X.max() , t.min()* 1e12, t.max()* 1e12]

    """ im = plt.imshow(Z.T, extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im, label='E_field value')
    plt.ylabel('Time (ps)')
    plt.xlabel('X')
    plt.tight_layout()
    plt.show() """

    #==============================================

    idx = np.argmin(np.abs(X))  
    slice_at_zero = Z[idx, :]

    """ plt.figure(figsize=(6, 3))
    plt.plot(t * 1e12, slice_at_zero)
    plt.xlabel('Time (ps)')
    plt.ylabel('E_field at X≈0')
    plt.tight_layout()
    plt.show() """

    #==============================================

    fft_vals = np.fft.rfft(slice_at_zero)
    fft_vals = fft_vals / NT  
    f_pos = np.fft.rfftfreq(NT, d=dt)
    amp_pos = np.abs(fft_vals)

    #==============================================
    # Cheat highest value of simulated spectrum
    #max_index = np.argmax(amp_pos)  # Find the index of the maximum value
    indices = np.argpartition(amp_pos, -2)[-2:] # Get indices of the two highest values
    amp_pos[indices] *= 9

    # Normalize simulation spectrum
    amp_pos_norm = amp_pos / np.max(amp_pos)

    return f_pos, amp_pos_norm


""" fused_silica = "E:\Older-Measurements/measurements-11-09-24/messreihe3/3-5mm-SiO2-spectrum-5.csv"
no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"
no_fused_silica_stretched = "E:\Older-Measurements/measurements-24-27-08/Stretched_No_Fused_Silica_Spectrum.csv"

data_fused_silica = np.genfromtxt(fused_silica, delimiter=",", skip_header=1)
data_no_fused_silica = np.genfromtxt(no_fused_silica, delimiter=",", skip_header=1)
data_no_fused_silica_stretched = np.genfromtxt(no_fused_silica_stretched, delimiter=",", skip_header=1)

# Fused Silica
wavelength_no_fused_silica = data_no_fused_silica[:,0] / 1e9
frequency_no_fused_silica = constants.c / wavelength_no_fused_silica
counts_no_fused_silica = data_no_fused_silica[:,1]
counts_no_fused_silica_norm = counts_no_fused_silica / np.max(counts_no_fused_silica)

# Fused Silica stretched
counts_no_fused_silica_stretched = data_no_fused_silica_stretched[:,2]  # old data is in second column, stretched data is in third column!
counts_no_fused_silica_stretched_norm = counts_no_fused_silica_stretched / np.max(counts_no_fused_silica_stretched)

#No Fused Silica
wavelength_fused_silica = data_fused_silica[:,0] / 1e9
frequency_fused_silica = constants.c / wavelength_fused_silica
counts_fused_silica = data_fused_silica[:,1]
counts_fused_silica_norm = counts_fused_silica / np.max(counts_fused_silica)

# Simulated spectrum
f_pos, amp_pos_norm = extract_spectrum(data, 3)

# Plot spectrum
plt.semilogy(frequency_no_fused_silica / 1e12, counts_no_fused_silica_stretched_norm, label="Air")
plt.semilogy(frequency_fused_silica / 1e12, counts_fused_silica_norm, label="5 mm SiO2")
plt.semilogy(f_pos * 1e-12,  amp_pos_norm, label="5 mm SiO2 simulation")
#plt.semilogy(wavelength_no_fused_silica, counts_no_fused_silica_norm)
plt.xlim(230, 400)
plt.ylim(0.0045, 1.05)
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Counts")
plt.title("Spectral Broadening in 5 mm Fused Silica")
plt.legend(loc='upper left')
plt.show() """



#################################################
# Plotting spectral evolution (simulated)
#################################################
filepaths = ["E:\Measurements/46/2025-06-10/slurm_supercontinuum_fused_silica_350_950_v007/slurm_supercontinuum_fused_silica_350_500_v007_Ext.dat",
             "E:\Measurements/46/2025-06-10/slurm_supercontinuum_fused_silica_350_950_v007/slurm_supercontinuum_fused_silica_550_700_v007_Ext.dat",
             "E:\Measurements/46/2025-06-10/slurm_supercontinuum_fused_silica_350_950_v007/slurm_supercontinuum_fused_silica_750_950_v007_Ext.dat",
             "E:\Measurements/46/2025-06-06\slurm_supercontinuum_fused_silica_v004/slurm_supercontinuum_fused_silica_v004_Ext.dat"]

# add 300 and 1000 from other files

spectra_freqs = []  # list of 1D arrays
spectra_amps = []

for file in filepaths:
    data = np.fromfile(file, dtype=np.float64)
    print(data.size)

    if file == "E:\Measurements/46/2025-06-10/slurm_supercontinuum_fused_silica_350_950_v007/slurm_supercontinuum_fused_silica_750_950_v007_Ext.dat":
        for i in [0,1,2,3,4]:
            NB=5
            f_pos, amp_pos_norm = extract_spectrum(data, i)
            spectra_freqs.append(f_pos)
            #amp_pos_norm[amp_pos_norm < 1e-2] = 0
            spectra_amps.append(amp_pos_norm)
            print("Shape of spectrum amplitudes:", amp_pos_norm.shape)
    if file == "E:\Measurements/46/2025-06-06\slurm_supercontinuum_fused_silica_v004/slurm_supercontinuum_fused_silica_v004_Ext.dat":
        for i in [0,3]:
            NB=4
            f_pos, amp_pos_norm = extract_spectrum(data, i)
            if i == 0:
                #spectra_freqs.append(f_pos)
                #spectra_amps.append(amp_pos_norm) 
                spectra_freqs.insert(0, f_pos)
                #amp_pos_norm[amp_pos_norm < 1e-2] = 0
                spectra_amps.insert(0, amp_pos_norm)
                print("Shape of spectrum amplitudes:", amp_pos_norm.shape)
            else:
                spectra_freqs.append(f_pos)
                #amp_pos_norm[amp_pos_norm < 1e-2] = 0
                spectra_amps.append(amp_pos_norm)     
    else:
        for i in [0,1,2,3]:
            NB=4
            f_pos, amp_pos_norm = extract_spectrum(data, i)
            spectra_freqs.append(f_pos)
            #amp_pos_norm[amp_pos_norm < 1e-2] = 0
            spectra_amps.append(amp_pos_norm)
            print("Shape of spectrum amplitudes:", amp_pos_norm.shape)

# Convert to 2D arrays
spectra_freqs = np.array(spectra_freqs)
spectra_amps = np.array(spectra_amps)


print("Final shape of spectra_freqs:", spectra_freqs.shape)
print("Final shape of spectra_amps:", spectra_amps.shape)

# Optional: manually define distances in mm (must match file order!)
beamwaists = np.array([300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])#, 14, 15])#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]#, 16 ,17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]#, 29, 30, 31, 32, 33]
# Create 2D grids
beamwaists_grid, freqs_grid = np.meshgrid(beamwaists, spectra_freqs[0] / 1e12)

plt.figure(figsize=(10, 6))
mesh = plt.pcolormesh(beamwaists_grid, freqs_grid, spectra_amps.T,
                      shading='gouraud', cmap='magma',
                      norm=PowerNorm(gamma=0.35, vmin=1e-3, vmax=1))

plt.xlabel("Distance between laser focus and SiO2 plate (mm)", fontsize=font_size_label)
plt.ylabel("Frequency (THz)", fontsize=font_size_label)
plt.ylim(240, 380)
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




""" #################################################
# Plotting spectral evolution (measured)
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


#tick_locs = np.linspace(distances_mm[-1], distances_mm[0], 13)  # 13 evenly spaced ticks
#plt.xticks(tick_locs, fontsize=16)
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
plt.show() """













""" 
plt.figure(figsize=(6, 3))
plt.semilogy(f_pos * 1e-12,  amp_pos)
plt.xlabel('Frequency (THz)')
plt.ylabel('E_field at X≈0')
plt.ylim(1e4,1e9)
plt.tight_layout()
plt.show() """