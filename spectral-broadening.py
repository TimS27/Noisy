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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

font_size_label = 18
font_size_title = 20


#################################################
# Plotting final and initial spectrum
#################################################

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
# Preparing FT-limited pulse corresponding to spectrum for simulations
#################################################

# === Load and Sort Spectrum ===
no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"
df = pd.read_csv(no_fused_silica)
wavelengths_nm = np.sort(df["Wavelength"].values)
intensities = df["Counts"].values[np.argsort(df["Wavelength"].values)]

# === Convert to Frequency Domain ===
c = 3e8
frequencies_Hz = c / (wavelengths_nm * 1e-9)

# === Extend and Interpolate ===
pad_fraction = 0.2
bandwidth = frequencies_Hz.max() - frequencies_Hz.min()
f_min_ext = frequencies_Hz.min() - pad_fraction * bandwidth
f_max_ext = frequencies_Hz.max() + pad_fraction * bandwidth
num_points = 2**13  # half spectrum size
freq_half = np.linspace(f_min_ext, f_max_ext, num_points)

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

# === Construct Hermitian-Symmetric Spectrum ===
amp_half = np.sqrt(spectrum_half_windowed)
# Build full symmetric spectrum: [negative frequencies | positive frequencies]
amp_full = np.concatenate([
    amp_half,                        # positive frequencies
    np.conj(amp_half[::-1])         # negative frequencies
])

# === Time-Domain Electric Field ===
E_time = fftshift(ifft(amp_full))

""" # Shift electric field to lab frame
E_time_shifted = E_time * np.exp(2j * np.pi * f0 * t) """

# === Time Axis ===
dfreq = freq_half[1] - freq_half[0]
N = len(amp_full)
dt = 1 / (N * dfreq)
t = np.linspace(-N/2, N/2 - 1, N) * dt

# === Plot: Spectrum ===
freq_full = np.linspace(-1, 1, N) * (dfreq * N / 2)  # synthetic symmetric frequency axis for plotting
#freq_full_symmetric = np.linspace(-f_max_ext, f_max_ext, N)
#freq_bins = fftshift(fftfreq(N, d=dt))  # now centered on 0 Hz
#f0 = np.sum(freq_half * spectrum_half_windowed) / np.sum(spectrum_half_windowed)    # center frequency ~290 THz
#freq_full_lab = freq_bins + f0
plt.figure(figsize=(10, 4))
plt.plot(np.fft.fftshift(freq_full * 1e-12), np.abs(amp_full)**2 / np.max(np.abs(amp_full)**2), label="Full Hermitian Spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Spectral Intensity")
plt.title("Hermitian-Symmetric Spectrum")
plt.tight_layout()
plt.show()

# === Plot: Time-Domain Electric Field ===
plt.figure(figsize=(10, 4))
plt.plot(t * 1e15, np.real(E_time), label="E(t)")
plt.xlabel("Time (fs)")
plt.ylabel("Electric Field (a.u.)")
plt.title("Real Oscillating Electric Field")
plt.tight_layout()
plt.show()

# === Forward FFT to reconstruct spectrum ===
E_freq_reconstructed = fftshift(fft(E_time))
spectrum_reconstructed = np.abs(E_freq_reconstructed)**2
spectrum_reconstructed /= np.max(spectrum_reconstructed)

# Extract frequency axis corresponding to fftshifted FFT
df = freq_half[1] - freq_half[0]
freq_full = np.linspace(-N/2, N/2 - 1, N) * df  # in Hz

# Take positive half
positive_freq_indices = freq_full >= 0
freq_positive = freq_full[positive_freq_indices]
spectrum_positive = spectrum_reconstructed[positive_freq_indices]

# === Plot reconstructed vs original windowed spectrum ===
plt.figure(figsize=(10, 4))
plt.plot(freq_half * 1e-12, spectrum_half_windowed / np.max(spectrum_half_windowed), label="Original Spectrum (windowed)")
plt.plot(freq_positive * 1e-12, spectrum_positive / np.max(spectrum_positive), '--', label="Reconstructed Spectrum (from E(t))")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Intensity")
plt.title("Reconstructed Spectrum Check")
plt.legend()
plt.tight_layout()
plt.show()





""" no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"

# Load spectrum from CSV
df = pd.read_csv(no_fused_silica)

# Extract data
wavelengths_nm = df["Wavelength"].values
intensities = df["Counts"].values

# Sort in increasing wavelength (if not already sorted)
sort_idx = np.argsort(wavelengths_nm)
wavelengths_nm = wavelengths_nm[sort_idx]
intensities = intensities[sort_idx]

# Convert wavelength (nm) to frequency (Hz)
c = 3e8  # speed of light in m/s
frequencies_Hz = c / (wavelengths_nm * 1e-9)

# Define extended frequency range
bandwidth = frequencies_Hz.max() - frequencies_Hz.min()
pad_fraction = 0.2  # 20% extra frequencies on both ends
freq_min_ext = frequencies_Hz.min() - pad_fraction * bandwidth
freq_max_ext = frequencies_Hz.max() + pad_fraction * bandwidth

# Uniform extended frequency grid
num_points = 2**14
freq_uniform_ext = np.linspace(freq_min_ext, freq_max_ext, num_points)

# Interpolate and extend with zeros at the edges
interp_func_ext = interp1d(frequencies_Hz, intensities, kind='linear', bounds_error=False, fill_value=0)
interp_spectrum_ext = interp_func_ext(freq_uniform_ext)


# Identify significant region
threshold = 0.001 * np.max(interp_spectrum_ext)  # 5% of max intensity
signal_region = interp_spectrum_ext > threshold
first_idx = np.argmax(signal_region)
last_idx = len(signal_region) - np.argmax(signal_region[::-1])
flat_length = last_idx - first_idx

# Find the spectral peak index (center of window)
peak_idx = np.argmax(interp_spectrum_ext)

# Reduce flat region manually
narrow_width = int(0.6 * flat_length)  # reduce length of window
start_idx = max(0, peak_idx - narrow_width // 2)
end_idx = min(len(interp_spectrum_ext), peak_idx + narrow_width // 2)
window_length = end_idx - start_idx

taper = tukey(window_length, alpha=0.9)  # sharper than alpha=0.6
window = np.zeros_like(interp_spectrum_ext)
window[start_idx:end_idx] = taper

# Apply the window
windowed_spectrum_ext = interp_spectrum_ext * window

# Plot interpolated, extended and windowed spectrum
plt.figure(figsize=(10, 4))
plt.plot(freq_uniform_ext * 1e-12, interp_spectrum_ext, label="Extended Interpolated Spectrum")
plt.plot(freq_uniform_ext * 1e-12, windowed_spectrum_ext, label="Windowed Extended Spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Spectral Intensity (a.u.)")
plt.title("Extended and Windowed Spectrum")
plt.legend()
plt.tight_layout()
plt.show()

# === Construct Hermitian-Symmetric Spectrum ===
amp_half = np.sqrt(windowed_spectrum_ext)
spectrum_hermitian = np.concatenate([
    amp_half,
    np.conj(amp_half[::-1])  # Complex conjugate of mirrored positive half
])

# === Inverse FFT to Time Domain ===
E_time_real = fftshift(ifft(spectrum_hermitian))

# === Time Axis ===
dfreq = freq_uniform_ext[1] - freq_uniform_ext[0]
N = len(spectrum_hermitian)
dt = 1 / (N * dfreq)
t = np.linspace(-N/2, N/2 - 1, N) * dt  # Time in seconds

# Step 8: Plot Real Electric Field
plt.figure(figsize=(10, 4))
plt.plot(t * 1e15, np.real(E_time_real)**2)
plt.xlabel("Time (fs)")
plt.ylabel("Electric Field (a.u.)")
plt.title("Reconstructed Real Electric Field (Hermitian-Symmetric Spectrum)")
plt.tight_layout()
plt.show() """

#time_domain_centered = np.roll(time_domain, -np.argmax(np.abs(time_domain)))    # Optional: Center time-domain electric field before FFT

""" # === Step 1: Forward Fourier transform (back to frequency domain) ===
E_freq_reconstructed = fftshift(fft(E_time))

# === Step 2: Compute spectral intensity ===
S_reconstructed = np.abs(E_freq_reconstructed)**2

# === Step 3: Normalize both spectra for comparison ===
S_reconstructed /= np.max(S_reconstructed)
windowed_spectrum_norm = windowed_spectrum_ext / np.max(windowed_spectrum_ext)

# === Step 4: Plot comparison ===
plt.figure(figsize=(10, 4))
plt.plot(freq_uniform_ext * 1e-12, windowed_spectrum_norm, label="Original Windowed Spectrum")
plt.plot(freq_uniform_ext * 1e-12, S_reconstructed, '--', label="Reconstructed via FFT")
plt.xlabel("Frequency (THz)")
plt.ylabel("Normalized Spectral Intensity")
plt.title("Spectrum Reconstruction via FFT")
plt.legend()
plt.tight_layout()
plt.show()
 """


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

