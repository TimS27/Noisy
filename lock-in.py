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
import frshelpers.plot
import glob

# Read .isf data
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne20ab-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne20ab-inter.isf') """
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne10ab-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne10ab-inter.isf') """
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-delay-2.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-inter-2.isf') """
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-nd40a-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-nd40a-inter.isf') """
""" data_cal, header_cal = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-delay.isf')
data_inter, header_inter = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-inter.isf') """
#data_move, header_move = isfread('E:\Measurements/46/2025-05-12/500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-move.isf')

series_simulated = []
transmissions_simulated = []
for file_name in glob.glob("E:\Measurements/46/2025-05-22/tim_onlyshot*.npz"):

    data = np.load(file_name)    # diff, cal entries
    sig = data["diff"]
    calibration = data["cal"]

    wavelength = 1064e-9
    g = 100e3
    r = 0.67

    nenir20ac = 0.0092574
    nenir40ac = 0.0000585
    nd40a = 0.000351
    ne20ab = 0.024822
    ne10ab = 0.0790946

    power=500e-6*nd40a*nenir40ac#nenir20ac*nenir40ac*nd40a*ne20ab
    #print(power)
    #print(power*(1/20)/(constants.h*constants.c/wavelength))

    #plt.plot(data_inter[0][:50000], data_inter[1][:50000])
    #plt.show()

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
    #omega_expected = 2*np.pi*92315
    omega_expected = 2*np.pi*49e3
    dt = sample_rate**-1
    wavelength = 1064e-9

    t = np.arange(calibration.size)*dt
    #t = np.arange(data_cal[0][0:500000].size)*dt
    #calibration = data_cal[1][0:500000]
    #sig = data_inter[1][0:500000]

    """ plt.plot(calibration)
    plt.plot(sig+0.2)
    plt.show() """

    """ ax = plt.subplot(2,1,1)
    f, t_short, Zxx = signal.stft(sig, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
    plt.axhline(omega_expected/(2*np.pi), color='r')
    plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
    plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.axvline(t[i_trigger], color='r')
    plt.axvline(t[i_start], color='r')
    plt.axvline(t[i_stop], color='r')

    plt.subplot(2,1,2, sharex=ax)
    f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
    plt.axhline(omega_expected/(2*np.pi), color='r')
    plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
    plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show() """

    # Gets rid of data near the turning points of the scan
    #def isolate_linear(t, calibration, sig):
    # bandpass
    bandpass_freq = omega_expected / (2*np.pi) / 2
    bandwidth = bandpass_freq/2
    sos = signal.butter(10, (bandpass_freq-bandwidth/2, bandpass_freq+bandwidth/2), 'bandpass', fs=sample_rate, output="sos")
    sig_bandpass = signal.sosfiltfilt(sos, calibration)# sig)
    sig_bandpass_abs = np.abs(sig_bandpass)
    sos2 = signal.butter(10, (15), 'lowpass', fs=sample_rate, output="sos")
    sig_trigger = signal.sosfiltfilt(sos2, sig_bandpass_abs)

    # Isolate interferogram data not at turning point via indices
    threshold_triggerpoints = np.max(sig_trigger)/3

    sig_no_turningpoints = []
    t_no_turningpoints = []
    cal_no_turningpoints = []
    while True:
        i_trigger = np.argmax(sig_trigger > threshold_triggerpoints)
        i_start = i_trigger + int(0.08 * sample_rate)
        i_stop = i_start + int(0.12 * sample_rate)

        """ plt.plot(sig)
        plt.axvline(i_trigger)
        plt.axvline(i_start)
        plt.axvline(i_stop)
        plt.show() """

        """ ax = plt.subplot(2,1,1)
        f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
        plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
        plt.axhline(omega_expected/(2*np.pi), color='r')
        plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
        plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.axvline(t[i_trigger]-t[0], color='r')
        plt.axvline(t[i_start]-t[0], color='r')
        plt.axvline(t[i_stop]-t[0], color='r')
        plt.subplot(2,1,2, sharex=ax)
        plt.plot(t-t[0], sig_trigger)
        plt.axvline(t[i_trigger]-t[0], color='r')
        plt.axvline(t[i_start]-t[0], color='r')
        plt.axvline(t[i_stop]-t[0], color='r')

        plt.show()  """

        sig_no_turningpoints.append(sig[i_start:i_stop])
        sig = sig[i_start:]
        t_no_turningpoints.append(t[i_start:i_stop])
        t = t[i_start:]
        cal_no_turningpoints.append(calibration[i_start:i_stop])
        calibration = calibration[i_start:]
        sig_trigger = sig_trigger[i_start:]
        if(i_stop > sig.size):
            break

    sig_no_turningpoints = sig_no_turningpoints[1:]
    t_no_turningpoints = t_no_turningpoints[1:]
    cal_no_turningpoints = cal_no_turningpoints[1:]

    #plt.plot(np.concatenate(t_no_turningpoints), np.concatenate(cal_no_turningpoints))



    """ plt.figure()
    ax = plt.subplot(2,1,1)
    f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
    plt.axhline(omega_expected/(2*np.pi), color='r')
    plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
    plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

    plt.subplot(2,1,2, sharex=ax)
    plt.plot(t, sig_trigger)

    plt.show() """

    #indices_no_turningpoints = np.where(sig_trigger < threshold_triggerpoints)[0]

    """ t_no_turningpoints = t[indices_no_turningpoints]
    calibration_no_turningpoints = calibration[indices_no_turningpoints]
    sig_no_turningpoints = sig[indices_no_turningpoints] """

    #return [t_no_turningpoints, calibration_no_turningpoints, sig_no_turningpoints]


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
        cal_interp = np.interp(delay_axis_evenly, delay_axis, calibration)

        """ print(delay_axis_evenly)
        print(delay_axis) """

        sig_i = sig_interp * np.sin(2*np.pi*constants.c/wavelength * delay_axis_evenly)
        sig_q = sig_interp * np.cos(2*np.pi*constants.c/wavelength * delay_axis_evenly)

        """ ax = plt.subplot(2,1,1)
        f, t_short, Zxx = signal.stft(cal_interp, constants.c/wavelength, nperseg=256)
        plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
        plt.axhline(omega_expected/(2*np.pi), color='r')
        plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
        plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #plt.axvline(t[i_trigger], color='r')
        #plt.axvline(t[i_start], color='r')
        #plt.axvline(t[i_stop], color='r') """

        """ plt.subplot(2,1,2, sharex=ax)
        f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
        plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
        plt.axhline(omega_expected/(2*np.pi), color='r')
        plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
        plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
        plt.title('STFT Magnitude')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]') """

        #plt.plot(delay_axis_evenly, sig_interp)

        #plt.show()

        cut_off_freq = 2*np.pi*constants.c/wavelength/20
        sos = signal.butter(10, (cut_off_freq), 'lowpass', fs=1/(np.mean(np.diff(delay_axis_evenly))), output="sos")
        sig_i_lowpass = signal.sosfiltfilt(sos, sig_i)
        sig_q_lowpass = signal.sosfiltfilt(sos, sig_q)

        result = sig_i_lowpass + 1j*sig_q_lowpass
        result_abs = np.abs(result)
        result_phase = np.angle(result)
        #print(delay_axis.size, sig.size) # delay_axis_evenly ahs more values because we interpolated
        frequencies, spectrum = fourioso.transform(delay_axis_evenly, sig_interp)

        """ plt.plot(frequencies,(np.abs(spectrum))**2)
        #print(np.mean(result_abs))
        plt.show() """

        return result_abs, result_phase#, frequencies, spectrum]

        #isolated_data = isolate_linear(t, calibration, sig)
        #lockin_amplitude = lockin(isolated_data[0], isolated_data[1], isolated_data[2])
        #lockin_amplitude = lockin(t_no_turningpoints, calibration_no_turningpoints, sig_no_turningpoints)

    result_abs_mean = []
    result_phase_mean = []
    for t,cal,sig in zip(t_no_turningpoints, cal_no_turningpoints, sig_no_turningpoints):
        result_abs, result_phase = lockin(t, cal, sig)
        result_abs_mean.append(np.mean(result_abs))
        result_phase_mean.append(np.mean(result_phase))

    #print(result_abs_mean, result_phase_mean)

    series_simulated.append(result_abs_mean)
    result_abs_mean = np.array(result_abs_mean)
    #frshelpers.plot.plot_allan(result_abs_mean[None,:])
    #plt.show()
    series_simulated_means = np.mean(series_simulated, axis=1)
    #np.save('E:\Measurements/46/2025-05-22/simulated_6bit_.npy', series_simulated_means)

    #print(np.mean(result_abs_mean))
    transmissions_simulated += [data["transmission"]]

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
    #plt.plot(lockin_amplitude[lockin_amplitude>np.max(lockin_amplitude)/2])
    #plt.plot(lockin_amplitude[1], lockin_amplitude[2])
    #plt.plot(calibration_reconstructed)
    #plt.plot(calibration_shifted)
    #plt.plot(calibration_shifted_lowpass)
    #plt.plot(calibration_normalized*0.002)
    #plt.show()

""" simulated_6bit = np.load('E:\Measurements/46/2025-05-22/simulated_6bit_.npy')
simulated_nonoise = np.load('E:\Measurements/46/2025-05-22/simulated_no_noise.npy')
simulated_6bitNEP20pW = np.load('E:\Measurements/46/2025-05-22/simulated_6bitNEP20pW.npy')
simulated_onlyshot = np.load('E:\Measurements/46/2025-05-22/simulated_only_shot.npy') """

series = [0.013514934087822695, 0.0002877861220757857, 0.00013562762918902484, 0.00010961890634659422, 0.0001082965896049411]
#print(series_simulated)
gain = 100e3
lo_power = 500e-6
transmission = np.array([nenir40ac, nenir40ac*nd40a, nenir40ac*nd40a*nenir20ac, nenir40ac*nd40a*nenir20ac*ne10ab, nenir40ac*nd40a*nenir20ac*ne20ab])
signal_power = np.array(transmissions_simulated) * lo_power
#print(transmission)
#v_shot = g * np.sqrt(2 * constants.elementary_charge * lo_power * r)    # 1*lo_power because shot_noise level is relevant at low signal arm powers only
#photons_shot = v_shot / (constants.h * (constants.c/wavelength))
one_photon = constants.h * constants.c / wavelength
number_photons_shot = 5 * 0.05 * lo_power / one_photon
number_photons_shot_uncertainty = np.sqrt(number_photons_shot)
v_shot_rms = np.sqrt(2*constants.elementary_charge * lo_power * r * (gain**2) * 125e3)

E_photon = constants.h * constants.c/wavelength
# Function to convert Power (W) to Photons per second
def power_to_photons(P):
    return P / E_photon

fig, ax1 = plt.subplots(figsize=(8,6))
#plt.loglog(x_axis_voltage, expected_voltage)
#plt.loglog(x_axis, expected_voltage, label='g*r*4*Es*ELO')
#plt.loglog(x_axis, y_fit, label="Balanced Fit")
plt.gca().invert_xaxis()  # Inverts the x-axis
#plt.loglog(signal_voltage, data)
plt.loglog(signal_power, series_simulated_means, 'o', color="green", label='Simulated Signal Balanced (Only Shot Noise)')# (Linear Output)') #'--o'
#plt.loglog(signal_power, simulated_onlyshot, 'o', color="green", label='Simulated Signal Balanced (Only Shot Noise)')# (Linear Output)') #'--o'
#plt.loglog(a,b, '--o',color="green", label='test')# (Linear Output)')
#plt.loglog(signal_power_400_autobal, data_autobal_400, '--o',color="blue" , label='Measured Signal Autobalanced')# (Log Output)')
#plt.axhline(v_shot_rms, color='r', linestyle='--', label="Shot Noise") # Calculated shot noise level
plt.axhline(2.39e-5, color='r', linestyle='--', label="Shot Noise") # Shot noise level from simulations
plt.axvline(one_photon, color='black', linestyle='--', label="One Photon/s")
plt.xlim(1e-2, 1e-21)
plt.ylim(1e-7, 100)
plt.xlabel('Signal Arm Power [W]')
plt.ylabel('Measured Signal [V]')
plt.title('BHD Signal vs. Signal Arm Power')
plt.legend(loc='lower left')#, bbox_to_anchor=(0,0.15))
ax1.xaxis.grid(visible=True, which='both')
ax1.yaxis.grid(visible=True, which='major')
#ax1.axvspan(signal_power_400_bal[0], 1e-12, alpha=0.1, color='green')

ax2 = ax1.secondary_xaxis("top", functions=(power_to_photons, lambda N: N * E_photon))  # transform function and its inverse
ax2.set_xlabel("Signal Arm [Photons/s]")

plt.show()