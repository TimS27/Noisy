import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

#filepath = "E:\Measurements/46/2025-06-05/slurm_supercontinuum_fused_silica_v003/slurm_supercontinuum_fused_silica_v003_Ext.dat"
filepath = "E:\Measurements/46/2025-06-06/slurm_supercontinuum_fused_silica_v004/slurm_supercontinuum_fused_silica_v004_Ext.dat"
data = np.fromfile(filepath, dtype=np.float64)
print(data.size)

NB=4
NT=8192
NX=20000
NP=2

E_field = data.reshape((NB,NP,NX,NT)).astype(np.float32)
del data

dX=0.1 # um

X =  dX* np.arange(-(NX-1)/2, (NX-1)/2 + 1, 1)

dt = 0.5e-15 # s
t = np.arange(NT) * dt 


Z = np.squeeze(E_field[1, 0, :, :])

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

#==============================================

fused_silica = "E:\Older-Measurements/measurements-11-09-24/messreihe3/3-5mm-SiO2-spectrum-5.csv"
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
plt.show()



""" 
plt.figure(figsize=(6, 3))
plt.semilogy(f_pos * 1e-12,  amp_pos)
plt.xlabel('Frequency (THz)')
plt.ylabel('E_field at X≈0')
plt.ylim(1e4,1e9)
plt.tight_layout()
plt.show() """