import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey
from scipy.fft import ifft, fftshift
from scipy import constants
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter1d

# Font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

# Load and sort spectrum (nm and counts)
no_fused_silica = "E:\Older-Measurements/measurements-24-27-08/No_Fused_Silica_Spectrum_Data.csv"
fused_silica_5mm = "E:\Older-Measurements/measurements-11-09-24/messreihe3/3-5mm-SiO2-spectrum-5.csv"

# Read first file
df = pd.read_csv(no_fused_silica)
wavelengths_nm = np.sort(df["Wavelength"].values)
intensities_ = df["Counts"].values[np.argsort(df["Wavelength"].values)]

# Read second file
df = pd.read_csv(fused_silica_5mm)
wavelengths_nm2 = np.sort(df["Wavelength"].values)
intensities_2 = df["Counts"].values[np.argsort(df["Wavelength"].values)]

""" plt.plot(wavelengths_nm, intensities_)
plt.show() """


def prepare_spectrum(wavelengths_nm, intensities_):
    # Convert to frequency domain
    frequencies_Hz = constants.c / (wavelengths_nm * 1e-9)
    intensities = intensities_ - intensities_.min() # Set noise floor to 0

    # Sort by increasing frequency
    idx_f = np.argsort(frequencies_Hz)
    f_sorted = frequencies_Hz[idx_f]
    i_sorted = intensities[idx_f]

    ##### Extend and Interpolate
    t = 10000e-15
    dt = 0.1e-15
    N_full = int(np.round(t / dt))  # Compute number of FTT points
    if N_full % 2:                  # Make sure number is even
        N_full += 1
    N_half = N_full // 2
    delta_f = 1.0 / t               # Define frequency resolution
    freq_half = np.arange(N_half) * delta_f     #Creates the positive frequency grid (freq_half) up to N_half points
    spectrum_half = PchipInterpolator(f_sorted, i_sorted, extrapolate=False)(freq_half) # PCHIP interpolation (shape-preserving) to get the spectrum on the new frequency grid
    spectrum_half[np.isnan(spectrum_half)] = 0      #   Fills values outside the measured range with zeros

    ###### Apply Window Centered at Peak
    f_orig_min = f_sorted.min()
    f_orig_max = f_sorted.max()
    idx_start = np.searchsorted(freq_half, f_orig_min, side='left') # Finds the indices in the new frequency grid that correspond to the original measured spectrum’s start and end.
    idx_end = np.searchsorted(freq_half, f_orig_max, side='right') - 1
    if idx_end < idx_start:                                         # Check if there is valid data
        raise ValueError("No valid overlap region found! Check your frequency grid.")
    else:                                                           # Creates a Tukey window over the region of real measured data and applies it to the interpolated spectrum
        length = idx_end - idx_start + 1
        taper = tukey(length, alpha=0.2)
        window = np.zeros_like(spectrum_half)
        window[idx_start:idx_end + 1] = taper
        spectrum_half_windowed = spectrum_half * window

    return frequencies_Hz, intensities, freq_half, N_full, delta_f, window, spectrum_half_windowed

frequencies_Hz, intensities, freq_half, N_full, delta_f, window, spectrum_half_windowed = prepare_spectrum(wavelengths_nm, intensities_)
frequencies_Hz2, intensities2, freq_half2, N_full2, delta_f2, window2, spectrum_half_windowed2 = prepare_spectrum(wavelengths_nm2, intensities_2)

##### Plot interpolated, extended and windowed spectrum
plt.figure(figsize=(10, 4))
plt.plot(frequencies_Hz * 1e-12, intensities_, label="Original spectrum")
plt.plot(frequencies_Hz * 1e-12, intensities, label="Original spectrum - noise floor")
plt.plot(freq_half * 1e-12, 2e4 * window, label="Window")
#plt.plot(freq_half * 1e-12, spectrum_half, label="Extended and interpolated spectrum")
plt.plot(freq_half * 1e-12, spectrum_half_windowed, label="Windowed, extended, and interpolated spectrum")
plt.xlabel("Frequency (THz)")
plt.ylabel("Spectral Intensity (Counts)")
plt.xlim(215, 420)
plt.title("Extending, Interpolating, and Windowing the Measured Initial Spectrum")
plt.legend()
plt.tight_layout()
plt.show()

##### IFFT to obtain time-domain pulse
def time_domain_pulse(spectrum_half_windowed, N_full, delta_f):
    amp_pos = np.sqrt(spectrum_half_windowed)   # Computes the square root of the spectrum intensity to get field amplitudes (assuming zero phase)
    # Construct a Hermitian-symmetric spectrum for the full frequency range
    N_pos = len(amp_pos)
    E_f = np.zeros(N_full, dtype=complex)
    E_f[:N_pos] = amp_pos
    E_f[N_pos:] = np.conj(amp_pos[::-1])
    time_signal = fftshift(ifft(E_f))
    dt = 1.0 / (N_full * delta_f)                   # Compute time step 'dt'
    time = (np.arange(N_full) - N_full / 2) * dt    # Compute time array 'time'

    """ data = np.column_stack((time, np.real(time_signal)))
    np.savetxt("pulse.txt", data, fmt="%.6e %.6e") """

    # Compute normalized real time-domian electric field
    time_signal_real_norm = np.real(time_signal)/np.max(np.real(time_signal))

    # Compute analytic signal
    analytic_signal = hilbert(time_signal_real_norm)     # Calculate intensity envelope (square of absolute electric field)
    envelope = np.abs(analytic_signal)
    envelope /= np.max(envelope)                        # Normalize intensity envelope

    # Smooth the envelope numerically
    envelope_smoothed = envelope#gaussian_filter1d(envelope, sigma=20)  # adjust sigma as needed

    # Compute FWHM of the smoothed envelope
    half_max = 0.5 * np.max(envelope_smoothed)
    indices_above_half = np.where(envelope_smoothed >= half_max)[0]

    if len(indices_above_half) > 0:
        fwhm_start = time[indices_above_half[0]]
        fwhm_end = time[indices_above_half[-1]]
        fwhm_duration = fwhm_end - fwhm_start
        print(f"FWHM duration: {fwhm_duration*1e15:.2f} fs")
    else:
        print("No region found at half maximum intensity.")

    return time, time_signal_real_norm, envelope_smoothed, fwhm_start, fwhm_end

time, time_signal_real_norm, envelope_smoothed, fwhm_start, fwhm_end = time_domain_pulse(spectrum_half_windowed, N_full, delta_f)
time2, time_signal_real_norm2, envelope_smoothed2, fwhm_start2, fwhm_end2 = time_domain_pulse(spectrum_half_windowed2, N_full2, delta_f2)


fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First pulse (your existing field & envelope)
axs[0].plot(time * 1e15, np.real(time_signal_real_norm), label="Real Part of E(t) (normalized)", alpha=0.7)
axs[0].plot(time * 1e15, envelope_smoothed, 'g', label="Envelope", linewidth=2)
axs[0].axvline(fwhm_start * 1e15, color='b', linestyle='--', lw=0.8)
axs[0].axvline(fwhm_end * 1e15, color='b', linestyle='--', lw=0.8)
axs[0].fill_between(time * 1e15, -1.5, 1.5, 
                     where=((time >= fwhm_start) & (time <= fwhm_end)), 
                     color='b', alpha=0.15, label="FWHM duration")
axs[0].set_ylabel("Amplitude (a.u.)")
axs[0].set_xlim(-200, 200)
axs[0].set_ylim(-1, 1.05)
axs[0].legend()
#axs[0].set_title("Fourier-Transform-Limited Time-Domain Pulse (Air)")
fwhm_duration1_fs = (fwhm_end - fwhm_start) * 1e15
axs[0].text(0.05, 0.9, f"FWHM: {fwhm_duration1_fs:.1f} fs", transform=axs[0].transAxes,
             fontsize=12, color='black')

# Second pulse (replace with your second field & envelope variables)
# E.g., time2, time_signal2, envelope_smoothed2, fwhm_start2, fwhm_end2
axs[1].plot(time2 * 1e15, np.real(time_signal_real_norm2), label="Real Part of E(t) (normalized)", alpha=0.7)
axs[1].plot(time2 * 1e15, envelope_smoothed2, 'g', label="Envelope", linewidth=2)
axs[1].axvline(fwhm_start2 * 1e15, color='b', linestyle='--', lw=0.8)
axs[1].axvline(fwhm_end2 * 1e15, color='b', linestyle='--', lw=0.8)
axs[1].fill_between(time2 * 1e15, -1.5, 1.5, 
                     where=((time2 >= fwhm_start2) & (time2 <= fwhm_end2)), 
                     color='b', alpha=0.15, label="FWHM duration")
axs[1].set_xlabel("Time (fs)")
axs[1].set_ylabel("Amplitude (a.u.)")
axs[1].set_xlim(-200, 200)
axs[1].set_ylim(-1, 1.05)
#axs[1].legend()
#axs[1].set_title("Fourier-Transform-Limited Time-Domain Pulse (5 mm Fused Silica)")
fwhm_duration2_fs = (fwhm_end2 - fwhm_start2) * 1e15
axs[1].text(0.05, 0.9, f"FWHM: {fwhm_duration2_fs:.1f} fs", transform=axs[1].transAxes,
             fontsize=12, color='black')#, bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))


plt.suptitle("Real Electric Field and Pulse Duration Comparison for FT-Limited pulses corresponding to Measured Spectra")
plt.tight_layout()
plt.show()




""" plt.figure(figsize=(10, 4))
plt.plot(time * 1e15, np.real(time_signal_real_norm), label="Real Part of E(t) (normalized)", alpha=0.7)
#plt.plot(time * 1e15, envelope, 'r', label="Raw Envelope (Hilbert)", linewidth=1)
plt.plot(time * 1e15, envelope_smoothed_scaled, 'g', label="Envelope", linewidth=2)
plt.axvline(fwhm_start * 1e15, color='b', linestyle='--', lw='0.8')#, label="FWHM start")
plt.axvline(fwhm_end * 1e15, color='b', linestyle='--', lw='0.8')#, label="FWHM end")
plt.fill_between(time * 1e15, -1.5, 1.5, 
                  where=((time >= fwhm_start) & (time <= fwhm_end)), 
                  color='b', alpha=0.15, label="FWHM duration")
plt.xlabel("Time (fs)")
plt.ylabel("Amplitude (a.u.)")
plt.xlim(-200,200)
plt.legend()
plt.title("Fourier-Transform-Limited Time-Domain Pulse, Envelope, and FWHM")
plt.tight_layout()
plt.show() """