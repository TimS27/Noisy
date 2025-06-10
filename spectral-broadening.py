import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
import glob
import os
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from scipy.fft import fft, ifft, fftshift, fftfreq

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "serif",
#   "font.size": 11,
#   "text.latex.preamble": r"\usepackage{lmodern}"
#})

font_size_label = 18
font_size_title = 20


#################################################
# Plotting final and initial spectrum
#################################################

fused_silica = "E:\Older-Measurements/measurements-11-09-24/messreihe3/3-5mm-SiO2-spectrum-5.csv"
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
plt.show()


#################################################
# Preparing FT-limited pulse corresponding to spectrum for simulations
#################################################

""" # Load and Sort Spectrum
no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"
df = pd.read_csv(no_fused_silica)
wavelengths_nm = np.sort(df["Wavelength"].values)
intensities = df["Counts"].values[np.argsort(df["Wavelength"].values)]

# Convert to Frequency Domain
c = 3e8
frequencies_Hz = c / (wavelengths_nm * 1e-9)

# Extend and Interpolate
pad_fraction = 1
bandwidth = frequencies_Hz.max() - frequencies_Hz.min()
f_min_ext = frequencies_Hz.min() - pad_fraction * bandwidth
f_max_ext = frequencies_Hz.max() + pad_fraction * bandwidth
num_points = 2**14  # half spectrum size
freq_half = np.linspace(f_min_ext, f_max_ext, num_points)
delta_f = (f_max_ext - f_min_ext) / (num_points - 1)    # Frequency resolution

interp_func = interp1d(frequencies_Hz, intensities, kind='linear', bounds_error=False, fill_value=0)
spectrum_half = interp_func(freq_half)

# === Apply Window Centered at Peak ===
threshold = 0.001 * np.max(spectrum_half)
signal_region = spectrum_half > threshold
first_idx = np.argmax(signal_region)
last_idx = len(signal_region) - np.argmax(signal_region[::-1])
flat_length = last_idx - first_idx
peak_idx = np.argmax(spectrum_half)
narrow_width = int(0.6 * flat_length)
start_idx = max(0, peak_idx - narrow_width // 2)
end_idx = min(len(spectrum_half), peak_idx + narrow_width // 2)
window_length = end_idx - start_idx

taper = tukey(window_length, alpha=0.9)
window = np.zeros_like(spectrum_half)
window[start_idx:end_idx] = taper
spectrum_half_windowed = spectrum_half * window

# Plot interpolated, extended and windowed spectrum
plt.figure(figsize=(10, 4))
plt.plot(freq_half * 1e-12, spectrum_half, label="Extended Interpolated Spectrum")
plt.plot(freq_half * 1e-12, spectrum_half_windowed, label="Windowed Extended Spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Spectral Intensity (a.u.)")
plt.title("Extended and Windowed Spectrum")
plt.legend()
plt.tight_layout()
plt.show()


# Number of points in the gap between pos. and neg. frequncies of spectrum
# Don't know if this frequency axis leads to errors later on!
gap_points = int(np.round((2*190e12) / delta_f))

# Create zero array for the gap
zero_gap = np.zeros(gap_points)

# Construct Hermitian-Symmetric Spectrum with gap filled with zeros
amp_half = np.sqrt(spectrum_half_windowed)
amp_full_filled_gap = np.concatenate([
    np.conj(amp_half[::-1]),  # negative frequencies (−440 THz to −190 THz)
    zero_gap,                 # zero gap (−190 THz to +190 THz)
    amp_half                  # positive frequencies (+190 THz to +440 THz)
])

# Physical lab-frame frequencies (your measured data range)
freq_positive = freq_half  # from f_min_ext to f_max_ext

# Negative frequencies mirror: -f_max_ext to -f_min_ext
freq_negative = -freq_half[::-1]

# Concatenate to get the real physical axis
freq_full_physical = np.concatenate([freq_negative, freq_positive])

# Number of points for the full spectrum
N_full = len(amp_full_filled_gap)

# Build a full, uniformly spaced frequency axis with the same Δf
freq_full_with_gap = np.linspace(
    -freq_half[-1],     # −f_max_ext
    freq_half[-1],      # +f_max_ext
    N_full              # total number of points
)



# Check if zero_gap needs to be adjusted
missing_points = len(freq_negative) + len(freq_positive)
if missing_points < N_full:
    zero_gap = np.zeros(N_full - missing_points)

# Build final frequency axis
freq_full_physical = np.concatenate([freq_negative, zero_gap * 0 + freq_negative[-1] + delta_f, freq_positive])

# Build final spectrum (Hermitian-symmetric)
amp_full_filled_gap = np.concatenate([np.conj(amp_half[::-1]), zero_gap, amp_half])

# Create final uniform frequency grid
freq_uniform = np.linspace(freq_full_physical.min(), freq_full_physical.max(), N_full)

# Interpolate real and imaginary parts separately to this uniform grid
amp_interp_real = interp1d(freq_full_physical, np.real(amp_full_filled_gap), kind='linear', fill_value=0, bounds_error=False)(freq_uniform)
amp_interp_imag = interp1d(freq_full_physical, np.imag(amp_full_filled_gap), kind='linear', fill_value=0, bounds_error=False)(freq_uniform)

# Combine to final complex spectrum for IFFT
amp_full_uniform = amp_interp_real + 1j * amp_interp_imag

plt.figure(figsize=(10, 4))
plt.plot(freq_uniform * 1e-12, np.abs(amp_full_uniform)**2 / np.max(np.abs(amp_full_uniform)**2), label="Prepared Spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Intensity")
plt.title("Final Prepared Hermitian-Symmetric Spectrum (for IFFT)")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(freq_full_with_gap * 1e-12, np.abs(amp_full_filled_gap)**2 / np.max(np.abs(amp_full_filled_gap)**2))
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Spectral Intensity")
plt.title("Hermitian-Symmetric Spectrum with Zero-Filled Gap")
plt.tight_layout()
plt.show()


# Time-Domain Electric Field
E_time = fftshift(ifft(amp_full_uniform))#filled_gap))

dt = 1 / (N_full * delta_f)  # in seconds
t = np.linspace(-N_full/2, N_full/2 - 1, N_full) * dt

f0 = np.sum(freq_half * spectrum_half_windowed) / np.sum(spectrum_half_windowed)
# Modulate the baseband pulse to the lab-frame center frequency
E_time_lab = np.real(E_time * np.exp(2j * np.pi * f0 * t))
# Use complex field for FFT
E_time_lab_complex = E_time * np.exp(2j * np.pi * f0 * t)



############## Exporting Electric field for LWE ##############
# Convert time to fs
#time_fs = t * 1e15

# Two-column data: time (fs) and real part of the field
data_to_save = np.column_stack((t, np.real(E_time_lab_complex)))

# Save to ASCII file
np.savetxt("E_time_real_field.asc", data_to_save, comments='')  # , header="time_fs  E_field_real"
##########################################################



plt.figure(figsize=(10, 4))
plt.plot(t * 1e15, np.real(E_time_lab))  # time in fs
plt.xlabel("Time (fs)")
plt.ylabel("Electric Field (a.u.)")
plt.title("Fourier-Transform-Limited Time-Domain Electric Field")
plt.tight_layout()
plt.show()


# === Time-Domain Electric Field (baseband) ===
E_time = fftshift(ifft(amp_full_uniform))  # final spectrum → time domain

# Time axis
freq_range = freq_full_with_gap[-1] - freq_full_with_gap[0]
dt = 1 / freq_range
t = np.linspace(-N_full/2, N_full/2 - 1, N_full) * dt

# Center frequency (lab frame)
spectrum_power = np.abs(amp_full_uniform)**2
f0 = np.sum(freq_full_with_gap * spectrum_power) / np.sum(spectrum_power)

# Modulate to lab-frame center frequency
E_time_lab_complex = E_time * np.exp(2j * np.pi * f0 * t)

# Reconstructed spectrum (complex field)
E_freq_lab_complex = fftshift(fft(E_time_lab_complex))
spectrum_lab_complex = np.abs(E_freq_lab_complex)**2
spectrum_lab_complex /= np.max(spectrum_lab_complex)

# FFT frequency bins
freq_bins = fftshift(fftfreq(N_full, d=dt))  # in Hz
freq_lab_frame = freq_bins + f0              # physically shifted to lab frame

# Plot final spectrum (jut shift manually for visualiztion, doesnt matter for simulations)
plt.figure(figsize=(10, 4))
plt.plot(freq_lab_frame * 1e-12, spectrum_lab_complex, '--', label="Reconstructed Spectrum (complex)")
plt.plot(freq_full_with_gap * 1e-12, np.abs(amp_full_uniform)**2 / np.max(np.abs(amp_full_uniform)**2), label="Original Hermitian Spectrum")
plt.axvline(f0 * 1e-12, color='r', linestyle='--', label=f"f0 = {f0*1e-12:.2f} THz")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Intensity")
plt.title("Lab-frame Spectrum (Single Peaks at ±290 THz)")
plt.legend()
plt.tight_layout()
plt.show() """













""" # === Time-Domain Electric Field (baseband) ===
E_time = fftshift(ifft(amp_full_filled_gap))
#dt = 1 / (N_full * delta_f)
freq_range = freq_full_with_gap[-1] - freq_full_with_gap[0]
dt = 1 / freq_range
t = np.linspace(-N_full/2, N_full/2 - 1, N_full) * dt

# Center frequency (lab frame)
#f0 = np.sum(freq_half * spectrum_half_windowed) / np.sum(spectrum_half_windowed)
spectrum_power = np.abs(amp_full_filled_gap)**2
f0 = np.sum(freq_half * spectrum_half_windowed) / np.sum(spectrum_half_windowed)

# Modulate to lab frame (complex field)
E_time_lab_complex = E_time * np.exp(2j * np.pi * f0 * t)

# Reconstructed spectrum (complex field)
E_freq_lab_complex = fftshift(fft(E_time_lab_complex))
spectrum_lab_complex = np.abs(E_freq_lab_complex)**2
spectrum_lab_complex /= np.max(spectrum_lab_complex)

# FFT frequency bins
freq_bins = fftshift(fftfreq(N_full, d=dt))  # in Hz
freq_lab_frame = freq_bins + f0              # physically shifted to lab frame

# Plot (on correct frequency axis for each!)
plt.figure(figsize=(10, 4))
plt.plot(freq_lab_frame * 1e-12, spectrum_lab_complex, '--', label="Reconstructed Spectrum (complex)")
plt.plot(freq_full_with_gap * 1e-12, np.abs(amp_full_filled_gap)**2 / np.max(np.abs(amp_full_filled_gap)**2), label="Original Hermitian Spectrum")
plt.axvline(f0 * 1e-12, color='r', linestyle='--', label=f"f0 = {f0*1e-12:.2f} THz")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Intensity")
plt.title("Lab-frame Spectrum (Single Peaks at ±290 THz)")
plt.legend()
plt.tight_layout()
plt.show() """

""" # Reconstruct Spectrum from Time-Domain Pulse: FFT of complex field → single-sided spectrum
E_freq_lab_complex = fftshift(fft(E_time_lab_complex))
spectrum_lab_complex = np.abs(E_freq_lab_complex)**2
spectrum_lab_complex /= np.max(spectrum_lab_complex)

freq_bins = fftshift(fftfreq(N_full, d=dt))  # in Hz, centered at 0
freq_lab_frame = freq_bins + f0              # shift up to lab frame

plt.figure(figsize=(10, 4))
plt.plot(freq_lab_frame * 1e-12, spectrum_lab_complex / np.max(spectrum_lab_complex), '--', label="Reconstructed Spectrum (complex)")
plt.plot(freq_full_with_gap * 1e-12, np.abs(amp_full_filled_gap)**2 / np.max(np.abs(amp_full_filled_gap)**2), label="Original Hermitian Spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Intensity")
plt.title("Lab-frame Spectrum (Single Peaks at ±290 THz)")
plt.legend()
plt.tight_layout()
plt.show() """



#################################################
# Plotting spectral evoltion
#################################################

""" # Folder where the .asc files are stored
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

#################################################

