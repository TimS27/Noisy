import numpy as np
from isfread_py3 import isfread
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import matplotlib.colors as colors
import fourioso
from scipy import constants

# Read .isf data
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-delay-2.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-inter-2.isf') """
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-nd40a-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-nd40a-inter.isf') """
data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-inter.isf')
#data_move, header_move = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-move.isf')

""" # Save as NumPy binary
np.save("E:/Measurements/46/2025-05-15/t.npy", t)
np.save("E:/Measurements/46/2025-05-15/calibration.npy", calibration)
np.save("E:/Measurements/46/2025-05-15/sig.npy", sig) """

""" t = np.arange(data_cal[0].size)*dt
where = (t>0.32) & (t<0.38)
t = t[where]
calibration = data_cal[1][where]
sig = data_inter[1][where] """

""" t = np.load("E:/Measurements/46/2025-05-15/t.npy")
calibration = np.load("E:/Measurements/46/2025-05-15/calibration.npy")
sig = np.load("E:/Measurements/46/2025-05-15/sig.npy") """

sample_rate = 500e3
omega_expected = 2*np.pi*92315
dt = sample_rate**-1
wavelength = 1064e-9

t = np.arange(data_cal[0][0:500000].size)*dt
calibration = data_cal[1][0:500000]
sig = data_inter[1][0:500000]

""" plt.plot(calibration)
plt.plot(sig+0.2)
plt.show() """

# Gets rid of data near the turning points of the scan
def isolate_linear(t, calibration, sig):
    # bandpass
    bandpass_freq = omega_expected / (2*np.pi) / 2
    bandwidth = bandpass_freq/2
    sos = signal.butter(10, (bandpass_freq-bandwidth/2, bandpass_freq+bandwidth/2), 'bandpass', fs=sample_rate, output="sos")
    sig_bandpass = signal.sosfiltfilt(sos, sig)
    sig_bandpass_abs = np.abs(sig_bandpass)
    sos2 = signal.butter(10, (50), 'lowpass', fs=sample_rate, output="sos")
    sig_trigger = signal.sosfiltfilt(sos2, sig_bandpass_abs)

    # Isolate interferogram data not at turning point via indices
    threshold_triggerpoints = np.max(sig_trigger)/3
    indices_no_turningpoints = np.where(sig_trigger < threshold_triggerpoints)[0]

    t_no_turningpoints = t[indices_no_turningpoints]
    calibration_no_turningpoints = calibration[indices_no_turningpoints]
    sig_no_turningpoints = sig[indices_no_turningpoints]

    return [t_no_turningpoints, calibration_no_turningpoints, sig_no_turningpoints]


def lockin(t, calibration, sig):
    # negative shift of frequency peak to 0
    shift = np.exp(1j*omega_expected*t)
    calibration_shifted = calibration * shift.conj()

    # lowpass
    cut_off_freq = omega_expected/(2*np.pi)/2
    sos = signal.butter(10, (cut_off_freq), 'lowpass', fs=sample_rate, output="sos")
    calibration_shifted_lowpass = signal.sosfiltfilt(sos, calibration_shifted)

    calibration_reconstructed = calibration_shifted_lowpass * shift

    # normalize
    fast_oscillation = np.exp(1j*omega_expected*t)
    phase_slow = np.unwrap(np.angle(calibration_reconstructed/ fast_oscillation))
    calibration_normalized = np.exp(1j*phase_slow) * fast_oscillation

    # scaling delay axis
    phase_fast = phase_slow + omega_expected * t
    n_oscillations = phase_fast / (2*np.pi)
    delay_axis = n_oscillations * wavelength / constants.c
    delay_axis_evenly = np.linspace(np.min(delay_axis), np.max(delay_axis), delay_axis.size*4)
    sig_interp = np.interp(delay_axis_evenly, delay_axis, sig)

    """ print(delay_axis_evenly)
    print(delay_axis) """

    sig_i = sig_interp * np.sin(2*np.pi*constants.c/wavelength * delay_axis_evenly)
    sig_q = sig_interp * np.cos(2*np.pi*constants.c/wavelength * delay_axis_evenly)

    cut_off_freq = 2*np.pi*constants.c/wavelength/20
    sos = signal.butter(10, (cut_off_freq), 'lowpass', fs=1/(np.mean(np.diff(delay_axis_evenly))), output="sos")
    sig_i_lowpass = signal.sosfiltfilt(sos, sig_i)
    sig_q_lowpass = signal.sosfiltfilt(sos, sig_q)

    result = sig_i_lowpass + 1j*sig_q_lowpass
    result_abs = np.abs(result)

    return result_abs

isolated_data = isolate_linear(t, calibration, sig)
lockin_amplitude = lockin(isolated_data[0], isolated_data[1], isolated_data[2])

""" freq,spec = fourioso.transform(t-t[t.size//2], calibration)
freq,spec_shifted = fourioso.transform(t-t[t.size//2], calibration_shifted)
freq,spec_lowpass = fourioso.transform(t-t[t.size//2], calibration_shifted_lowpass)
freq,spec_reconstructed = fourioso.transform(t-t[t.size//2], calibration_reconstructed) """

""" plt.figure()
plt.plot(freq, np.abs(spec)**2)
#plt.plot(freq, np.abs(spec_shifted)**2)
#plt.plot(freq, np.abs(spec_lowpass)**2)
plt.plot(freq, np.abs(spec_reconstructed)**2) """
""" plt.plot(freq, spec_shifted.real)
plt.plot(freq, spec_lowpass.real+1e-6) """


""" plt.figure()
f, t, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
plt.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
plt.axhline(omega_expected/(2*np.pi), color='r')
plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show() """

#plt.plot(calibration)
#plt.plot(delay_axis)
#plt.plot(delay_axis_evenly)
#plt.plot(sig)
#plt.plot(sig_i_lowpass)
#plt.plot(sig_q_lowpass)
#plt.plot(sig_interp)
#plt.plot(delay_axis_evenly, result_abs)
#plt.ylim(0, 0.04)
#plt.plot(sig_bandpass)
#plt.plot(sig_trigger)
#plt.plot(sig_no_turningpoints[0:100000])
plt.plot(lockin_amplitude[lockin_amplitude>np.max(lockin_amplitude)/2])
#plt.plot(calibration_reconstructed)
#plt.plot(calibration_shifted)
#plt.plot(calibration_shifted_lowpass)
#plt.plot(calibration_normalized*0.002)
plt.show()