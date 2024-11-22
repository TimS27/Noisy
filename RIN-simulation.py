import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

###################### Simulating Time Domain Data ########################

# Simulation parameters
fs = 5e6 #2.5e9  # Oscilloscope sampling frequency (2.5 GHz) -> determines highest frequency that can be observed, f_Nyquist = fs/2
delta_t = 1 / fs    # time resolution
T = 0.001   # Total duration of the signal (1 ms) -> determines frequency resolution
delta_f = 1 / T   # frequency resolution
N = int(fs * T)  # Total number of samples
time = np.linspace(0, T, N, endpoint=False)  # Time vector

# Laser parameters
laser_wavelength = 1064e-9  # [nm]
P_avg = 230e-3  # Mean optical power (230 mW) [J/s], for 1s energy equals average power in Joules
photon_energy = const.h * const.c / laser_wavelength    # [J]
avg_photon_number = P_avg / photon_energy  # [1/s]
relative_fluctuation_amplitude = 0.01  # Relative fluctuation amplitude (1% of mean power)

# Photodiode parameters
R = 1 # Photodetector responsivity [A/W]

# Relaxation oscillation parameters
f_ro = 0.65e6  # Relaxation oscillation frequency (approx. 150 kHz for Nd:YAG)
damping_factor = 2e3  # Damping factor for oscillations
relaxation_amplitude = 0.00001  # Amplitude of relaxation oscillations

# Noise contributions
white_noise_amplitude = 1e-5  # Amplitude of white noise
flicker_noise_amplitude = 1e-4  # Amplitude of 1/f noise
shot_noise_amplitude = np.sqrt((2 * const.eV * P_avg * delta_f) / R)

shot_noise_linear = 2 * const.h * (const.c / laser_wavelength) / P_avg
shot_noise_dBHz = 10 * np.log10(shot_noise_linear)
shot_noise_dBcHz = 10 * np.log10(shot_noise_linear / P_avg)

# Detector noise
# noise equivalent electronic amplitude
nep = 1.2e-11   # [W/sqrt(Hz)] optical power
gain = 1e4      # [V/A]
neea = nep * R * gain    # [V/sqrt(Hz)]
# square neea and use P=R*I^2
electronic_power_spectral_density = (neea ** 2) / R # [W/Hz]
# calculate electronic power spectral density per delta_f = RBW
#electronic_power_spectral_density_per_delta_f = electronic_power_spectral_density * (delta_f)
# calculate electronic power spectral density in dB/Hz
electronic_power_spectral_density_dBHz = 10 * np.log10(electronic_power_spectral_density / P_avg)

# Generate noise components
np.random.seed(42)  # For reproducibility
white_noise = white_noise_amplitude * np.random.randn(N)  # Gaussian white noise
flicker_noise = flicker_noise_amplitude * np.cumsum(np.random.normal(0, 1, N)) / fs  # 1/f noise
#shot_noise_photons = np.random.normal(avg_photon_number, np.sqrt(avg_photon_number), size = N)   # or np.random.poisson(avg_photon_number, number_of_samples) for low means
shot_noise = shot_noise_amplitude * np.random.normal(0, 1, N)   # normal(mean, standard dev, number of samples), choose standard dev = 1 and scale with schot_noise_amplitude
acoustic_frequency = 100  # Acoustic noise at 100 Hz
acoustic_noise = 0.001 * P_avg * np.sin(2 * np.pi * acoustic_frequency * time)

# Generate relaxation oscillations
omega_ro = 2 * np.pi * f_ro  # Angular frequency
envelope = np.exp(-damping_factor * time)  # Damped envelope
relaxation_oscillations = relaxation_amplitude * np.sin(omega_ro * time) * envelope

# Combine all components
fluctuations = (
    relative_fluctuation_amplitude * P_avg * (white_noise + flicker_noise)
    + relaxation_oscillations
    + shot_noise
    #+ acoustic_noise
)
intensity_signal = P_avg + fluctuations

'''
# Plot the time-domain signal
plt.figure(figsize=(10, 4))
plt.plot(time[:5000], intensity_signal[:5000])  # Plot a short segment for clarity
plt.xlabel("Time [s]")
plt.ylabel("Intensity [W]")
plt.title("Simulated Laser Intensity with Relaxation Oscillations")
plt.grid()
plt.show()
'''

# Save data for FFT analysis
#np.savetxt('time_data.txt', time)  # Save time vector
#np.savetxt('signal_data.txt', intensity_signal)  # Save intensity signal

###################### RIN ########################

# Load time-domain data
#time = np.loadtxt('time_data.txt')  # Time vector
#signal = np.loadtxt('signal_data.txt')  # Signal vector

# Parameters
mean_voltage = np.mean(intensity_signal)  # DC level (mean voltage)
power = mean_voltage / R  # Optical power (Watts)

# Remove DC offset
signal_fluctuations = intensity_signal - mean_voltage

# FFT computation
fft_result = np.fft.fft(signal_fluctuations)
freqs = np.fft.fftfreq(len(intensity_signal), d=(time[1] - time[0]))  # Frequency axis
psd = (np.abs(fft_result) ** 2) / len(intensity_signal)  # Power spectral density

# Normalize and convert to RIN
rin = 10 * np.log10(psd / power**2)

'''
# Plot RIN
plt.plot(freqs[1:len(freqs)//2], rin[1:len(rin)//2])  # Plot positive frequencies only
plt.xlabel('Frequency (Hz)')
plt.ylabel('RIN (dB/Hz)')
plt.title('Relative Intensity Noise (RIN)')
plt.grid()
plt.show()
'''

###################### Plotting ########################

plt.figure(figsize=(11, 5))

# Plot the time-domain signal
plt.subplot(1, 2, 1)
plt.plot(time[:100000], intensity_signal[:100000])  # Plot a short segment for clarity
plt.xlabel("Time [s]")
plt.ylabel("Intensity [W]")
plt.title("Simulated Laser Intensity with Relaxation Oscillations")
plt.grid()

# Plot RIN
plt.subplot(1, 2, 2)
plt.plot(freqs[1:len(freqs)//2], rin[1:len(rin)//2])  # Plot positive frequencies only
plt.hlines(shot_noise_dBcHz, 0, 2.5e6)
plt.hlines(electronic_power_spectral_density_dBHz, 0, 2.5e6)
plt.text(0.25e6, -170, 'shot noise') #verticalalignment='center')
plt.text(0.5e6, -140, 'NEP') #verticalalignment='center')
plt.xlabel('Frequency [Hz]')
plt.ylabel('RIN [dBc/Hz]')
plt.title('Relative Intensity Noise (RIN)')
plt.grid()

#plt.tight_layout()
plt.show()