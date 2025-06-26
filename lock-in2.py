import numpy as np
from isfread_py3 import isfread
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from scipy.signal import butter, filtfilt
import matplotlib.colors as colors
import fourioso, binreduce
from scipy import constants
import frshelpers.plot
import glob
import os
import re
from scipy.signal import windows
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

def plot_stft(data, sample_rate, data2=None):
  t = np.arange(data.size)*sample_rate**-1

  # Plot stft of signal, spectrum, and raw signal data
  if 0:
    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], width_ratios=[3, 1])  # top row taller

    # Top-left: STFT
    ax0 = fig.add_subplot(gs[0, 0])
    f, t_short, Zxx = signal.stft(data, sample_rate, nperseg=256)
    pcm = ax0.pcolormesh(t_short, f / 1e3, np.abs(Zxx), shading='auto',
                        norm=colors.LogNorm(vmin=np.abs(Zxx).max() * 1e-8, vmax=np.abs(Zxx).max()))
    ax0.set_title('Short-time Fourier transform')
    ax0.set_ylabel('Frequency [kHz]')
    ax0.set_xlabel('Time [s]')

    # Top-right: Spectrum
    ax1 = fig.add_subplot(gs[0, 1])
    nu, spectrum = fourioso.transform(t, data)
    nur, specr = binreduce.multi_binreduce(nu, 5000, abs(spectrum) ** 2)
    ax1.plot(specr[nur > 0], nur[nur > 0] / 1e3)
    ax1.set_xscale("log")
    ax1.set_xlim((1e-15, 1e-6))
    ax1.set_xlabel("PSD")
    ax1.set_ylabel("Frequency [kHz]")
    ax1.set_title("Spectrum")

    # Bottom (spanning both): Raw signal
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, data)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Balanced signal (V)")
    ax2.set_ylim((-.12, .12))
    ax2.set_title("Raw balanced-detector output (signal: {:.1f} nW, LO: 500 µW)".format(500e-6 * T / 1e-9))

    """ fig = plt.figure(figsize=(6,2.), constrained_layout=True)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    ax.plot(t, data)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("balanced signal (V)")
    ax.set_ylim((-.12,.12))
    ax.set_title("raw balanced-detector output (signal: {:.1f} nW, LO: 500 µW)".format(500e-6*T/1e-9))
    #plt.savefig("rawall.png")

    fig = plt.figure(figsize=(6,3), constrained_layout=True)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    f, t_short, Zxx = signal.stft(data, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f/1e3, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*1e-8, vmax=np.abs(Zxx).max()))
    ax.set_title('Short-time Fourier transform')
    ax.set_ylabel('frequency [kHz]')
    ax.set_xlabel('t (s)')
    #plt.savefig("2dall.png")

    fig = plt.figure(figsize=(1.5,3), constrained_layout=True)
    gs = fig.add_gridspec(1,1)
    ax = fig.add_subplot(gs[0,0])
    nu, spectrum = fourioso.transform(t, data)
    nur, specr = binreduce.multi_binreduce(nu, 5000, abs(spectrum)**2)
    ax.plot(specr[nur>0], nur[nur>0]/1e3)
    ax.set_xscale("log")
    ax.set_xlim((1e-15, 1e-6))
    ax.set_xlabel("PSD")
    ax.set_ylabel("f (kHz)")
    ax.set_title("spectrum")
    #plt.savefig("spec.png") """

    plt.show()

  """ ax=plt.subplot(2,2,1)
  f, t_short, Zxx = signal.stft(data, sample_rate, nperseg=256)
  plt.pcolormesh(t_short, f/1e3, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*1e-8, vmax=np.abs(Zxx).max()))
  print(np.abs(Zxx).max())
  plt.title('STFT Magnitude')
  plt.ylabel('Frequency [kHz]')
  plt.xlabel('Time [sec]')
  plt.subplot(2,2,2,sharey=ax)
  nu, spectrum = fourioso.transform(t, data)
  nur, specr = binreduce.multi_binreduce(nu, 5000, abs(spectrum)**2)
  plt.plot(specr[nur>0], nur[nur>0]/1e3)
  plt.xscale("log")
  plt.xlim((1e-15, 1e-6))
  plt.subplot(2,2,3)
  i0 = np.argmin(abs(t-0.25))
  l = 10000
  plt.plot(t[i0:i0+l], data[i0:i0+l])
  plt.ylim((-1.2e-1, 1.2e-1))

  if data2 is not None:
    plt.subplot(2,2,4)
    f, t_short, Zxx = signal.stft(data2, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f/1e3, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*1e-8, vmax=np.abs(Zxx).max()))
    print(np.abs(Zxx).max())
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [kHz]')
    plt.xlabel('Time [sec]') """

def lockin(t, calibration, sig, sample_rate, omega_expected, wavelength):
  # shift the carrier frequency of calibration signal down to DC
  shift = np.exp(1j*omega_expected*t)
  calibration_shifted = calibration * shift.conj()

  # lowpass to isolate the slowly-varying envelope
  cut_off_freq = omega_expected/(2*np.pi)/2
  sos = signal.butter(10, (cut_off_freq), 'lowpass', fs=sample_rate, output="sos")
  calibration_shifted_lowpass = signal.sosfiltfilt(sos, calibration_shifted)

  # Shift the low-passed data back to the original frequency, reconstructing the “cleaned” calibration signal
  calibration_reconstructed = calibration_shifted_lowpass * shift

  # Normalize/smoothen clibraiton signal
  # Unwrap the phase (removing jumps of 2π) and build a normalized calibration signal containing only the slowly varying phase drift (e.g., from scan nonlinearities), but preserving the carrier
  fast_oscillation = np.exp(1j*omega_expected*t)
  phase_slow = np.unwrap(np.angle(calibration_reconstructed/ fast_oscillation))
  calibration_normalized = np.exp(1j*phase_slow) * fast_oscillation

  # Scaling delay axis
  # Map the time axis to optical delay in meters
  phase_fast = phase_slow + omega_expected * t
  n_oscillations = phase_fast / (2*np.pi)
  delay_axis = n_oscillations * wavelength / constants.c
  delay_axis_evenly = np.linspace(np.min(delay_axis), np.max(delay_axis), delay_axis.size*4)
  sig_interp = np.interp(delay_axis_evenly, delay_axis, sig)
  cal_interp = np.interp(delay_axis_evenly, delay_axis, calibration)

  """ print(delay_axis_evenly)
  print(delay_axis) """

  # Demodulate the signal in the delay domain by applying sine and cosine terms at the carrier frequency in delay units (spatial oscillations in delay, not in time)
  sig_i = sig_interp * np.sin(2*np.pi*constants.c/wavelength * delay_axis_evenly)
  sig_q = sig_interp * np.cos(2*np.pi*constants.c/wavelength * delay_axis_evenly)

  if 0:
    ax = plt.subplot(2,1,1)
    f, t_short, Zxx = signal.stft(cal_interp, constants.c/wavelength, nperseg=256)
    plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    #plt.axvline(t[i_trigger], color='r')
    #plt.axvline(t[i_start], color='r')
    #plt.axvline(t[i_stop], color='r')
  
    plt.subplot(2,1,2, sharex=ax)
    f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
    plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
    plt.axhline(omega_expected/(2*np.pi), color='r')
    plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
    plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

  #plt.plot(delay_axis_evenly, sig_interp)

  #plt.show()

  cut_off_freq = 2*np.pi*constants.c/wavelength/20
  sos = signal.butter(10, (cut_off_freq), 'lowpass', fs=1/(np.mean(np.diff(delay_axis_evenly))), output="sos")
  filter_reaction_time = cut_off_freq**-1
  i0 = int( filter_reaction_time/np.mean(np.diff(delay_axis_evenly)) )  # Accounts for filter reaction time

  # final complex result (for envelope amplitude and phase)
  sig_i_lowpass = signal.sosfiltfilt(sos, sig_i)
  sig_q_lowpass = signal.sosfiltfilt(sos, sig_q)


#  sig_iq_lowpass = signal.sosfiltfilt(sos, sig_interp * (sin+i cos))
#  (-i sig_iq_lowpass) * np.exp(...) = signal.sosfiltfilt(sos, sig_interp * (cos - i sin))

  # physically reconstructed interferogram
  siginterp_reconstructed = 2*np.real( (-1j)*(sig_i_lowpass+1j*sig_q_lowpass)*np.exp(1j*2*np.pi*constants.c/wavelength*delay_axis_evenly) )

  if 0:
    # Plot interpolated signal vs. interpolated even delay
    plt.figure()
    #ax=plt.subplot(2,1,1)
    plt.plot(delay_axis_evenly, sig_interp)
    plt.plot(delay_axis_evenly, siginterp_reconstructed)
    #plt.subplot(2,1,2,sharex=ax)
    #plt.plot(np.real( 2*(sig_i_lowpass+1j*sig_q_lowpass)*np.exp(1j*2*np.pi*constants.c/wavelength * delay_axis_evenly - 1j*np.pi/2) ))
    #plt.xlim((20000,21000))
    plt.axvline(delay_axis_evenly[0]+filter_reaction_time)
    plt.show()

  result = sig_i_lowpass + 1j*sig_q_lowpass
  result_abs = np.abs(result)
  result_phase = np.angle(result)
  #print(delay_axis.size, sig.size) # delay_axis_evenly has more values because we interpolated
  frequencies, spectrum = fourioso.transform(delay_axis_evenly, sig_interp)
  frequencies, spectrum1 = fourioso.transform(delay_axis_evenly, siginterp_reconstructed)


  if 0:
    plt.figure()
    plt.plot(frequencies,(np.abs(spectrum))**2)
    plt.plot(frequencies,(np.abs(spectrum1))**2)
    plt.yscale("log")
    #print(np.mean(result_abs))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(result_abs)
    plt.subplot(2,1,2)
    plt.plot(result_phase)
    plt.show()

  #return result_abs, result_phase
  return result[i0:]

sample_rate = 500e3

ne03ab = 0.37945067
nenir20ac = 0.0092574
nenir40ac = 0.0000585
nd40a = 0.000351
ne20ab = 0.024822
ne10ab = 0.0790946
nenir240b = 0.0000478
nenir260b = 0.00000017



############ get transmission dict
folder_path = "C:\\Users/Admin/Documents/bhd-simulations/only_shot"

# regex pattern
pattern = re.compile(r"tim_onlyshot_([0-9.eE+-]+)_[0-9]*npz")#r"tim_test5_([0-9.eE+-]+)_[0-9]*npz"#r"tim_test3_([0-9.eE+-]+)\.npz"

file_dict = {}
for filename in os.listdir(folder_path):
    match = pattern.match(filename)
    if match:
        transmission_str = match.group(1)
        try:
            transmission = float(transmission_str)
            filepath = os.path.join(folder_path, filename)
            file_dict[filepath] = transmission
        except ValueError:
            pass  # skip if conversion fails
############
print(file_dict)

#transmissions = file_dict

#transmissions = {"E:\Measurements/46/2025-06-05/BHD1.dat": 0}
transmissions = {
  #"E:\\Measurements/46/2025-06-03-mod\\tek0014CH1.isf": nenir40ac*ne20ab,
    
############################ Files and transmissions from 2025-06-03 (1 mW LO) bal
  "E:\\Measurements/46/2025-06-03-mod\\tek0002CH1.isf": ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0004CH1.isf": ne20ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0006CH1.isf": nenir20ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0008CH1.isf": nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0010CH1.isf": nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0012CH1.isf": nenir40ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0014CH1.isf": nenir40ac*ne20ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0016CH1.isf": nenir40ac*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0018CH1.isf": nenir40ac*nd40a*ne20ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0020CH1.isf": nenir40ac*nd40a*nenir20ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0022CH1.isf": nenir240b*nenir20ac*ne20ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0024CH1.isf": nenir20ac*nd40a*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0026CH1.isf": nenir240b*nenir40ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0028CH1.isf": nenir240b*nd40a*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0030CH1.isf": nenir240b*nenir20ac*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0032CH1.isf": nenir240b*nd40a*ne20ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0034CH1.isf": nenir260b*nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0036CH1.isf": nenir40ac*nd40a*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0038CH1.isf": nenir240b*nenir40ac*nenir20ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0040CH1.isf": nenir40ac*nd40a*nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0042CH1.isf": nenir260b*nenir40ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0044CH1.isf": nenir240b*nenir40ac*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-03-mod\\tek0046CH1.isf": nenir40ac*nd40a*ne20ab*nenir20ac,
  "E:\\Measurements/46/2025-06-03-mod\\tek0048CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\tek0050CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-03-mod\\tek0052CH1.isf": ne20ab*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0054CH1.isf": nenir40ac*nenir240b *nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0056CH1.isf": ne10ab*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\tek0058CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-03-mod\\tek0060CH1.isf": nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0062CH1.isf": ne10ab*ne20ab*nenir20ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0064CH1.isf": ne10ab*ne20ab*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0066CH1.isf": nenir20ac*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\tek0068CH1.isf": ne10ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0070CH1.isf": ne20ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0072CH1.isf": ne20ab*nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0074CH1.isf": ne10ab*ne20ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\tek0076CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0000CH1.isf": ne03ab,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0002CH1.isf": 0.6745,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0004CH1.isf": 0.9, # check this
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0006CH1.isf": 0.9*0.6745,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0008CH1.isf": 0.3208,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0010CH1.isf": 0.2019,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0012CH1.isf": ne03ab*ne03ab,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0014CH1.isf": ne10ab*nenir240b*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0016CH1.isf": ne03ab*nenir240b*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0018CH1.isf": ne20ab*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0020CH1.isf": ne20ab*nenir240b*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0022CH1.isf": nenir20ac*nenir240b*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0024CH1.isf": ne03ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0026CH1.isf": ne10ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0028CH1.isf": ne10ab*nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0030CH1.isf": ne03ab*nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0032CH1.isf": ne03ab*ne20ab*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0034CH1.isf": ne10ab*ne20ab*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0036CH1.isf": nenir20ac*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0038CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0040CH1.isf": nenir40ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0042CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0044CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0046CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0048CH1.isf": nenir40ac*nenir240b*nenir260b,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0050CH1.isf": ne10ab*nenir40ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0052CH1.isf": ne03ab*ne10ab*nenir40ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new2-tek0000CH1.isf": 1,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0000CH1.isf": 557/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0002CH1.isf": 605/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0005CH1.isf": 672/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0007CH1.isf": 750/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0009CH1.isf": 858/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0011CH1.isf": 942/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0013CH1.isf": 1046/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0015CH1.isf": 1185/550,
  "E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0018CH1.isf": 1477/550,
}


""" ############################ Files and transmissions from 2025-06-03 (1 mW LO) autobal
"E:\\Measurements/46/2025-06-03-mod2\\tek0003CH1.isf": ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0005CH1.isf": ne20ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0007CH1.isf": nenir20ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0009CH1.isf": nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0011CH1.isf": nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0013CH1.isf": nenir40ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0015CH1.isf": nenir40ac*ne20ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0017CH1.isf": nenir40ac*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0019CH1.isf": nenir40ac*nd40a*ne20ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0021CH1.isf": nenir40ac*nd40a*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0023CH1.isf": nenir240b*nenir20ac*ne20ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0025CH1.isf": nenir20ac*nd40a*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0027CH1.isf": nenir240b*nenir40ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0029CH1.isf": nenir240b*nd40a*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0031CH1.isf": nenir240b*nenir20ac*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0033CH1.isf": nenir240b*nd40a*ne20ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0035CH1.isf": nenir260b*nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0037CH1.isf": nenir40ac*nd40a*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0039CH1.isf": nenir240b*nenir40ac*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0041CH1.isf": nenir40ac*nd40a*nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0043CH1.isf": nenir260b*nenir40ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0045CH1.isf": nenir240b*nenir40ac*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod2\\tek0047CH1.isf": nenir40ac*nd40a*ne20ab*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod2\\tek0049CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\tek0051CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod2\\tek0053CH1.isf": ne20ab*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0055CH1.isf": nenir40ac*nenir240b *nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0057CH1.isf": ne10ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\tek0059CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod2\\tek0061CH1.isf": nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0063CH1.isf": ne10ab*ne20ab*nenir20ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0065CH1.isf": ne10ab*ne20ab*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0067CH1.isf": nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\tek0069CH1.isf": ne10ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0071CH1.isf": ne20ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0073CH1.isf": ne20ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0075CH1.isf": ne10ab*ne20ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\tek0077CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0001CH1.isf": ne03ab,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0003CH1.isf": 0.6745,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0005CH1.isf": 0.9, # check this
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0007CH1.isf": 0.9*0.6745,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0009CH1.isf": 0.3208,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0011CH1.isf": 0.2019,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0013CH1.isf": ne03ab*ne03ab,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0015CH1.isf": ne10ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0017CH1.isf": ne03ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0019CH1.isf": ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0021CH1.isf": ne20ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0023CH1.isf": nenir20ac*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0025CH1.isf": ne03ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0027CH1.isf": ne10ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0029CH1.isf": ne10ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0031CH1.isf": ne03ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0033CH1.isf": ne03ab*ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0035CH1.isf": ne10ab*ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0037CH1.isf": nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0039CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0041CH1.isf": nenir40ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0043CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0045CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0047CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0049CH1.isf": nenir40ac*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0051CH1.isf": ne10ab*nenir40ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new2-tek0001CH1.isf": 1,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0001CH1.isf": 557/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0003CH1.isf": 605/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0006CH1.isf": 672/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0008CH1.isf": 750/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0010CH1.isf": 858/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0012CH1.isf": 942/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0014CH1.isf": 1046/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0016CH1.isf": 1185/550,
"E:\\Measurements/46/2025-06-03-mod2\\1mW-new3-tek0019CH1.isf": 1477/550,

#################################################### """

""" ############################ Files and transmissions from 2025-06-03 (1 mW LO) bal
"E:\\Measurements/46/2025-06-03-mod\\tek0002CH1.isf": ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0004CH1.isf": ne20ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0006CH1.isf": nenir20ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0008CH1.isf": nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0010CH1.isf": nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0012CH1.isf": nenir40ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0014CH1.isf": nenir40ac*ne20ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0016CH1.isf": nenir40ac*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0018CH1.isf": nenir40ac*nd40a*ne20ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0020CH1.isf": nenir40ac*nd40a*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0022CH1.isf": nenir240b*nenir20ac*ne20ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0024CH1.isf": nenir20ac*nd40a*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0026CH1.isf": nenir240b*nenir40ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0028CH1.isf": nenir240b*nd40a*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0030CH1.isf": nenir240b*nenir20ac*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0032CH1.isf": nenir240b*nd40a*ne20ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0034CH1.isf": nenir260b*nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0036CH1.isf": nenir40ac*nd40a*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0038CH1.isf": nenir240b*nenir40ac*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0040CH1.isf": nenir40ac*nd40a*nenir20ac*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0042CH1.isf": nenir260b*nenir40ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0044CH1.isf": nenir240b*nenir40ac*ne20ab*ne10ab,
"E:\\Measurements/46/2025-06-03-mod\\tek0046CH1.isf": nenir40ac*nd40a*ne20ab*nenir20ac,
"E:\\Measurements/46/2025-06-03-mod\\tek0048CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\tek0050CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod\\tek0052CH1.isf": ne20ab*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0054CH1.isf": nenir40ac*nenir240b *nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0056CH1.isf": ne10ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\tek0058CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod\\tek0060CH1.isf": nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0062CH1.isf": ne10ab*ne20ab*nenir20ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0064CH1.isf": ne10ab*ne20ab*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0066CH1.isf": nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\tek0068CH1.isf": ne10ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0070CH1.isf": ne20ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0072CH1.isf": ne20ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0074CH1.isf": ne10ab*ne20ab*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\tek0076CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0000CH1.isf": ne03ab,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0002CH1.isf": 0.6745,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0004CH1.isf": 0.9, # check this
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0006CH1.isf": 0.9*0.6745,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0008CH1.isf": 0.3208,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0010CH1.isf": 0.2019,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0012CH1.isf": ne03ab*ne03ab,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0014CH1.isf": ne10ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0016CH1.isf": ne03ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0018CH1.isf": ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0020CH1.isf": ne20ab*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0022CH1.isf": nenir20ac*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0024CH1.isf": ne03ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0026CH1.isf": ne10ab*ne20ab*nenir20ac*nenir40ac*nenir240b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0028CH1.isf": ne10ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0030CH1.isf": ne03ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0032CH1.isf": ne03ab*ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0034CH1.isf": ne10ab*ne20ab*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0036CH1.isf": nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0038CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0040CH1.isf": nenir40ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0042CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0044CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0046CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0048CH1.isf": nenir40ac*nenir240b*nenir260b,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0050CH1.isf": ne10ab*nenir40ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0052CH1.isf": ne03ab*ne10ab*nenir40ac*nenir260b*nd40a,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new2-tek0000CH1.isf": 1,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0000CH1.isf": 557/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0002CH1.isf": 605/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0005CH1.isf": 672/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0007CH1.isf": 750/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0009CH1.isf": 858/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0011CH1.isf": 942/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0013CH1.isf": 1046/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0015CH1.isf": 1185/550,
"E:\\Measurements/46/2025-06-03-mod\\1mW-new3-tek0018CH1.isf": 1477/550, """

############### Files and transmissions from 2025-06-02
""" "E:\\Measurements/46/2025-06-02-mod\\tek0000CH1.isf": 0,
  "E:\\Measurements/46/2025-06-02-mod\\tek0004CH1.isf": ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0006CH1.isf": ne20ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0008CH1.isf": nenir20ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0010CH1.isf": nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0013CH1.isf": nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0016CH1.isf": nenir40ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0019CH1.isf": nenir40ac*ne20ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0022CH1.isf": nenir40ac*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0025CH1.isf": nenir40ac*nd40a*ne20ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0028CH1.isf": nenir40ac*nd40a*nenir20ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0031CH1.isf": nenir240b*nenir20ac*ne20ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0034CH1.isf": nenir20ac*nd40a*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0037CH1.isf": nenir240b*nenir40ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0041CH1.isf": nenir240b*nd40a*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0044CH1.isf": nenir240b*nenir20ac*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0047CH1.isf": nenir240b*nd40a*ne20ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0050CH1.isf": nenir260b*nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0053CH1.isf": nenir40ac*nd40a*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0056CH1.isf": nenir240b*nenir40ac*nenir20ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0059CH1.isf": nenir40ac*nd40a*nenir20ac*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0062CH1.isf": nenir260b*nenir40ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0065CH1.isf": nenir240b*nenir40ac*ne20ab*ne10ab,
  "E:\\Measurements/46/2025-06-02-mod\\tek0068CH1.isf": nenir40ac*nd40a*ne20ab*nenir20ac,
  "E:\\Measurements/46/2025-06-02-mod\\tek0069CH1.isf": ne10ab*ne20ab*nenir20ac*nenir260b,
  "E:\\Measurements/46/2025-06-02-mod\\tek0070CH1.isf": ne10ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-02-mod\\tek0071CH1.isf": ne20ab*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0072CH1.isf": nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0073CH1.isf": ne10ab*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-02-mod\\tek0074CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b,
  "E:\\Measurements/46/2025-06-02-mod\\tek0075CH1.isf": nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0076CH1.isf": ne10ab*ne20ab*nenir20ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0077CH1.isf": ne10ab*ne20ab*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0078CH1.isf": nenir20ac*nenir40ac*nenir260b,
  "E:\\Measurements/46/2025-06-02-mod\\tek0079CH1.isf": ne10ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0080CH1.isf": ne20ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0081CH1.isf": ne20ab*nenir20ac*nenir260b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0082CH1.isf": ne10ab*ne20ab*nenir40ac*nenir240b*nd40a,
  "E:\\Measurements/46/2025-06-02-mod\\tek0083CH1.isf": ne20ab*nenir20ac*nenir40ac*nenir240b*nd40a, """
####################################################


############################ Files and transmissions from 2025-06-03 (1 mW LO) bal
  ###################################################

#slow_files = [
#  "500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne10ab-delay.isf",
#  "500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-ned40a-nenir20ac-ne20ab-delay.isf",
#]

import joblib # pip install joblib
memory = joblib.Memory("cache-joblib")

@memory.cache
def load_isf(filename):
  data_cal, header_cal = isfread(filename)
  #inter_filename = "inter".join(filename.rsplit("delay", 1))
  inter_filename = filename.replace("CH1","CH2")
  data_inter, header_inter = isfread(inter_filename)
  calibration = data_cal[1]#[0:750000]
  sig = data_inter[1]#[0:750000]
  if 0:
    plt.figure()
    plt.plot(sig[:10000])
    plt.title("signal")
    plt.figure()
    plt.plot(calibration[:10000])
    plt.title("calibration")
    plt.show()
    sys.exit()
  return transmissions[filename], sig, calibration,

def load_dat(filename):
  data = np.memmap(filename, dtype="<i2")
  ch1 = data[0::4]
  ch2 = data[1::4]
  ch3 = data[2::4]
  ch4 = data[3::4]
  """ plt.subplot(4,1,1)
  plt.plot(ch1[:1000000])
  plt.subplot(4,1,2)
  plt.plot(ch2[:1000000])
  plt.subplot(4,1,3)
  plt.plot(ch3[:1000000])
  plt.subplot(4,1,4)
  plt.plot(ch4[:1000000])
  plt.show() """
  return transmissions[filename], ch4[:100000000], ch2[:100000000],

def load(filename):
  if filename.endswith(".npz"):
    data = np.load(file_name)    # diff, cal entries
    sig = data["diff"]
    calibration = data["cal"]
    return data["transmission"], sig, calibration
  elif filename.endswith(".isf"):
    return load_isf(filename)
  elif filename.endswith(".dat"):
    return load_dat(filename)
  else:
    raise Exception("invalid extension")

#filenames = glob.glob("tim_onlyshot*.npz")
#filenames = glob.glob("*-delay.isf")

# Simulation data
#filenames = glob.glob("C:\\Users/Admin/Documents/bhd-simulations/only_shot/tim_onlyshot_*.npz")

# PLotting test
#filenames = glob.glob("E:\Measurements/46/2025-06-03-mod\\tek0014CH1.isf")

#filenames = glob.glob("E:\Measurements/46/2025-06-03-mod2/tek*CH1.isf")#)"tek*CH1.isf")

# Bal 1mW
filenames1 = glob.glob("E:\Measurements/46/2025-06-03-mod/tek*CH1.isf")#BHD1.dat)
filenames2 = glob.glob("E:\Measurements/46/2025-06-03-mod/1mW-new-tek*CH1.isf")
filenames3 = glob.glob("E:\Measurements/46/2025-06-03-mod/1mW-new2-tek*CH1.isf")
filenames4 = glob.glob("E:\Measurements/46/2025-06-03-mod/1mW-new3-tek*CH1.isf")

# AutoBal 1mW
#filenames1 = glob.glob("E:\Measurements/46/2025-06-03-mod2/tek*CH1.isf")#BHD1.dat)
#filenames2 = glob.glob("E:\Measurements/46/2025-06-03-mod2/1mW-new-tek*CH1.isf")
#filenames3 = glob.glob("E:\Measurements/46/2025-06-03-mod2/1mW-new2-tek*CH1.isf")
#filenames4 = glob.glob("E:\Measurements/46/2025-06-03-mod2/1mW-new3-tek*CH1.isf")

filenames = filenames1 + filenames2 + filenames3 + filenames4

#filenames = ["500microWLO-1,6V-delayPD-20MS-40s-nenir40ac-delay.isf"]
#filenames = ["tek0013CH1.isf"]
#filenames = ["tek0022CH1.isf"]

#filenames = glob.glob("tim_twinshotNEPwiggleENOBs_*.npz")

#filenames = ["tim_onlyshot_5.85000e-05.npz"]
#filenames = ["tim_twinshotNEPwiggleENOBs_5.85000e-05.npz"]
#filenames = ["tim_twinshotNEPwiggle_5.85000e-05.npz"]
#filenames = ["tim_twinshotNEP_5.85000e-05.npz"]
#filenames = ["tim_twinonlyshot_5.85000e-05.npz"]
#filenames = ["tim_twinnonoise_5.85000e-05.npz"]

series_simulated = []
transmissions_simulated = []
for file_name in filenames:
#    data = np.load(file_name)    # diff, cal entries
#    sig = data["diff"]
#    calibration = data["cal"]
    T, sig, calibration = load(file_name)
    print(file_name, T)

    #slow = file_name in slow_files
    #slow = True

    if 0:
      fig = plt.figure()
      plot_stft(sig, sample_rate, calibration)
      plt.savefig(file_name+".png")
      plt.close(fig)

    wavelength = 1064e-9
#    g = 100e3
#    r = 0.67



#    power=500e-6*nd40a*nenir40ac#nenir20ac*nenir40ac*nd40a*ne20ab
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

    omega_expected = 2*np.pi*65e3
    dt = sample_rate**-1
    wavelength = 1064e-9

    t = np.arange(calibration.size)*dt  # no allan
    #t = np.arange(data_cal[0][0:500000].size)*dt
    #calibration = data_cal[1][0:500000]
    #sig = data_inter[1][0:500000]

    """ plt.plot(calibration)
    plt.plot(sig+0.2)
    plt.show() """

    # Plot STFT with omega expected and area around it
    if 0:
      ax = plt.subplot(2,1,1)
      f, t_short, Zxx = signal.stft(sig, sample_rate, nperseg=256)
      plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-15, vmax= np.abs(Zxx).max()))
      plt.axhline(omega_expected/(2*np.pi), color='r')
      plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
      plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
      plt.title('STFT Magnitude')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      #plt.axvline(t[i_trigger], color='r')
      #plt.axvline(t[i_start], color='r')
      #plt.axvline(t[i_stop], color='r')
  
      plt.subplot(2,1,2, sharex=ax)
      f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
      plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-15, vmax= np.abs(Zxx).max()))
      plt.axhline(omega_expected/(2*np.pi), color='r')
      plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
      plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
      plt.title('STFT Magnitude')
      plt.ylabel('Frequency [Hz]')
      plt.xlabel('Time [sec]')
      plt.show()



    #######################################################
    # Preprocessing: Remove Turning Point Data
    #######################################################
    
    # Bandpass to extract main frequency
    bandpass_freq = omega_expected / (2*np.pi) / 2
    bandwidth = bandpass_freq/2
    sos = signal.butter(10, (bandpass_freq-bandwidth/2, bandpass_freq+bandwidth/2), 'bandpass', fs=sample_rate, output="sos")
    sig_bandpass = signal.sosfiltfilt(sos, calibration)# sig)
    sig_bandpass_abs = np.abs(sig_bandpass)

    # Low-pass to extract trigger envelope
    sos2 = signal.butter(10, (10), 'lowpass', fs=sample_rate, output="sos")
    #if slow:
    #  sos2 = signal.butter(10, (15), 'lowpass', fs=sample_rate, output="sos")
    #else:
    #  sos2 = signal.butter(10, (30), 'lowpass', fs=sample_rate, output="sos")
    sig_trigger = signal.sosfiltfilt(sos2, sig_bandpass_abs)

    # Isolate interferogram data not at turning point via indices
    threshold_triggerpoints = np.max(sig_trigger)/2

    sig_no_turningpoints = []
    t_no_turningpoints = []
    t_starts = []
    cal_no_turningpoints = []
    while True:
        i_trigger = np.argmax(sig_trigger > threshold_triggerpoints)
#        if slow:
#          i_start = i_trigger + int(0.04*2 * sample_rate)
#          i_stop = i_start + int(0.06*2 * sample_rate)
#        else:
#          i_start = i_trigger + int(0.04 * sample_rate)
#          i_stop = i_start + int(0.06 * sample_rate)
        i_start = i_trigger + int(0.04*3.3 * sample_rate)
        i_stop = i_start + int(0.06*4.2 * sample_rate)  #0.06*4.5

        """ plt.plot(sig)
        plt.axvline(i_trigger)
        plt.axvline(i_start)
        plt.axvline(i_stop)
        plt.show() """

        # Raw signal data
        if 0:
          # zoomemd out
          fig = plt.figure(figsize=(6,2.0), constrained_layout=True)
          gs = fig.add_gridspec(1,1)
          ax = fig.add_subplot(gs[0,0])
          ax.plot(t[i_start:i_stop], sig[i_start:i_stop])
          ax.set_xlabel("t (s)")
          ax.set_ylabel("balanced signal (V)")
          ax.set_title("raw balanced-detector output (signal: {:.1f} nW, LO: 500 µW)".format(500e-6*T/1e-9))
          #plt.savefig("withtp.png")

          # zoomed in
          fig = plt.figure(figsize=(6,2.0), constrained_layout=True)
          gs = fig.add_gridspec(1,1)
          ax = fig.add_subplot(gs[0,0])
          ax.plot(t[i_start:i_stop], sig[i_start:i_stop])
          ax.set_xlabel("t (s)")
          ax.set_ylabel("balanced signal (V)")
          ax.set_title("raw balanced-detector output (signal: {:.1f} nW, LO: 500 µW)".format(500e-6*T/1e-9))
          ax.set_xlim((0.06,0.061))
          #plt.savefig("zoom.png")

          plt.show()

        # Plot stft of calibration signal and trigger signal
        if 0:
          plt.figure(figsize=(8,6))
          ax = plt.subplot(2,1,1)
          f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
          plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
          plt.axhline(omega_expected/(2*np.pi), color='r')
          plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
          plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
          plt.title('STFT Magnitude and Scanning Trigger Signal for the Calibration Signal')
          plt.ylabel('Frequency [Hz]')
          plt.axvline(t[i_trigger]-t[0], color='r')
          plt.axvline(t[i_start]-t[0], color='orange')
          plt.axvline(t[i_stop]-t[0], color='blue')
          plt.subplot(2,1,2, sharex=ax)
          plt.xlabel('Time (s)')
          plt.ylabel('Trigger (a.u.)')
          plt.plot(t-t[0], sig_trigger)
          plt.axvline(t[i_trigger]-t[0], color='r')
          plt.axvline(t[i_start]-t[0], color='r')
          plt.axvline(t[i_stop]-t[0], color='r')
  
          plt.show()

        sig_no_turningpoints.append(sig[i_start:i_stop])
        sig = sig[i_start:]
        t_no_turningpoints.append(t[i_start:i_stop])  # no allan
        #if len(t_starts) > 0:                        # allan
        #  t_starts.append(i_start*dt + t_starts[-1]) # allan
        #else:                                        # allan
        #  t_starts.append(i_start*dt)                # allan
        t = t[i_start:] # no allan
        cal_no_turningpoints.append(calibration[i_start:i_stop])
        calibration = calibration[i_start:]
        sig_trigger = sig_trigger[i_start:]
        if(i_stop > sig.size):
            break

        #print(i_start)
    sig_no_turningpoints = sig_no_turningpoints[1:]
    t_no_turningpoints = t_no_turningpoints[1:]
    cal_no_turningpoints = cal_no_turningpoints[1:]

    sig_no_turningpoints = sig_no_turningpoints[:5]
    t_no_turningpoints = t_no_turningpoints[:5]
    cal_no_turningpoints = cal_no_turningpoints[:5]

    #sig_no_turningpoints = sig_no_turningpoints[:-1]  # allan
    #t_no_turningpoints = t_no_turningpoints[:-1]      # allan
    #cal_no_turningpoints = cal_no_turningpoints[:-1]  # allan

    #plt.plot(np.concatenate(t_no_turningpoints), np.concatenate(cal_no_turningpoints))

    #######################################################################

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


    
    #result_abs = []
    result_abs_mean = []
    result_phase_mean = []
    """ for tstart,cal,sig in zip(t_starts, cal_no_turningpoints, sig_no_turningpoints):    # allan
        t = tstart + dt * np.arange(cal.size)
        #print(tstart)
        result = lockin(t, cal, sig, sample_rate, omega_expected, wavelength) """
    for t,cal,sig in zip(t_no_turningpoints, cal_no_turningpoints, sig_no_turningpoints):    # allan
        result = lockin(t, cal, sig, sample_rate, omega_expected, wavelength)
        #result_abs.append(np.abs(result))
        # windowing slved problem: dc wasnt averaged out sufficiently
        result_abs_mean.append(np.abs(np.average(result, weights=windows.tukey(result.size, alpha=0.5)))) # changed abs() to np.abs()  #weights=windows.tukey(result.size, alpha=0.5
        result_phase_mean.append(np.angle(np.mean(result)))

    #print(result_abs_mean, result_phase_mean)

    series_simulated.append(result_abs_mean)
    print("Length: "+ str(len(result_abs_mean)))
    #result_abs_mean = np.array(result_abs_mean)
    #print(result_abs)
    #print(len(result_abs))
    #print(result_abs_mean[None,:])


    ########## Allan Dev
    #print(result_abs_mean)
    #frshelpers.plot.plot_allan(np.atleast_2d(result_abs_mean))#result_abs_mean[None,:])
    #plt.show()

    #for i in np.array(series_simulated).T:
       
    #series_simulated_std = np.mean(series_simulated, axis=1)
    series_simulated_means = np.mean(series_simulated, axis=1)
    #np.save('E:\Measurements/46/2025-05-22/simulated_6bit_.npy', series_simulated_means)

    #print(np.mean(result_abs_mean))
    transmissions_simulated += [T]

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

# Extract different scans for plotting statistics
series_simulated_means = np.array(series_simulated).ravel()
transmissions_simulated = np.repeat(transmissions_simulated, np.array(series_simulated).shape[1])
#print(series_simulated_means)
#print(transmissions_simulated)

""" simulated_6bit = np.load('E:\Measurements/46/2025-05-22/simulated_6bit_.npy')
simulated_nonoise = np.load('E:\Measurements/46/2025-05-22/simulated_no_noise.npy')
simulated_6bitNEP20pW = np.load('E:\Measurements/46/2025-05-22/simulated_6bitNEP20pW.npy')
simulated_onlyshot = np.load('E:\Measurements/46/2025-05-22/simulated_only_shot.npy') """

series = [0.013514934087822695, 0.0002877861220757857, 0.00013562762918902484, 0.00010961890634659422, 0.0001082965896049411]
#print(series_simulated)
wavelength = 1064 *1e-9
gain = 100e3
lo_power = 500e-6
transmission = np.array([nenir40ac, nenir40ac*nd40a, nenir40ac*nd40a*nenir20ac, nenir40ac*nd40a*nenir20ac*ne10ab, nenir40ac*nd40a*nenir20ac*ne20ab])
signal_power = np.array(transmissions_simulated) * lo_power
#print(transmission)
#v_shot = g * np.sqrt(2 * constants.elementary_charge * lo_power * r)    # 1*lo_power because shot_noise level is relevant at low signal arm powers only
#photons_shot = v_shot / (constants.h * (constants.c/wavelength))
one_photon = constants.h * constants.c / wavelength
number_photons_shot = 5 * 0.14 * lo_power / one_photon
number_photons_shot_uncertainty = np.sqrt(number_photons_shot)
#v_shot_rms = np.sqrt(2*constants.elementary_charge * lo_power * r * (gain**2) * 125e3)

E_photon = constants.h * constants.c/wavelength
# Function to convert Power (W) to Photons per second
def power_to_photons(P):
    return P / E_photon

print(signal_power.shape)
print(series_simulated_means.shape)
# Standard Deviation
df = pd.DataFrame(np.array([signal_power,series_simulated_means]).T, columns=["key", "val"])
df = df.groupby("key").val.apply(pd.Series.tolist)
print(df)
keys = df.index.to_numpy()
std_array = df.apply(np.std).to_numpy()

# Plot attenuation series
if 1:
  fig, ax1 = plt.subplots(figsize=(8,6))
  #plt.loglog(x_axis_voltage, expected_voltage)
  #plt.loglog(x_axis, expected_voltage, label='g*r*4*Es*ELO')
  #plt.loglog(x_axis, y_fit, label="Balanced Fit")
  plt.gca().invert_xaxis()  # Inverts the x-axis
  #plt.loglog(signal_voltage, data)
  #plt.loglog([signal_power[len(series_simulated_means)//4], series_simulated_means[len(series_simulated_means)//4]], [signal_power[3*len(series_simulated_means)//4], series_simulated_means[3*len(series_simulated_means)//4]], 'g--',alpha=0.3)# (Linear Output)') #'--o' #series_simulated_means.max()*np.logspace(-15,0)**.5
  plt.loglog(signal_power[len(series_simulated_means)//4]*np.logspace(-15,8), series_simulated_means[len(series_simulated_means)//4]*np.logspace(-15,8)**.5, 'g--',alpha=0.3)# (Linear Output)') #'--o' #series_simulated_means.max()*np.logspace(-15,0)**.5
  #plt.loglog(signal_power.max()*np.logspace(-15,0), series_simulated_means.max()*np.logspace(-15,0)**.5, 'g--',alpha=0.3)# (Linear Output)') #'--o' #series_simulated_means.max()*np.logspace(-15,0)**.5
  plt.loglog(signal_power, series_simulated_means, 'o', color="green", label='Signal Balanced')# (Linear Output)') #'--o'
  plt.loglog(keys, std_array, 'o', color="orange", label='Signal Balanced')# (Linear Output)') #'--o'
  #plt.loglog(signal_power, simulated_onlyshot, 'o', color="green", label='Simulated Signal Balanced (Only Shot Noise)')# (Linear Output)') #'--o'
  #plt.loglog(a,b, '--o',color="green", label='test')# (Linear Output)')
  #plt.loglog(signal_power_400_autobal, data_autobal_400, '--o',color="blue" , label='Measured Signal Autobalanced')# (Log Output)')
  #plt.axhline(v_shot_rms, color='r', linestyle='--', label="Shot Noise") # Calculated shot noise level
  #plt.axhline(2.39e-5, color='r', linestyle='--', label="Shot Noise") # Shot noise level from simulations
  plt.axvline(one_photon, color='black', linestyle='--', label="One Photon/s")
  #signal_photons = 12e-6 / 500e-6 + 246e-6 / 500e-6
  plt.axvline(one_photon/(0.06*4.2) , color='black', linestyle='--', label="One Photon/0.252 s", alpha=0.5)
  plt.xlim(1e-2, 1e-21)
  plt.ylim(1e-7, 100)
  plt.xlabel('Signal Arm Power [W]')
  plt.ylabel('Measured Signal [V]')
  #plt.title('BHD Signal vs. Signal Arm Power ')
  plt.title('Experimental data (Balanced, 1 mW LO)')
  plt.legend(loc='lower left')#, bbox_to_anchor=(0,0.15))
  ax1.xaxis.grid(visible=True, which='both')
  ax1.yaxis.grid(visible=True, which='major')
  #ax1.axvspan(signal_power_400_bal[0], 1e-12, alpha=0.1, color='green')

  ax2 = ax1.secondary_xaxis("top", functions=(power_to_photons, lambda N: N * E_photon))  # transform function and its inverse
  ax2.set_xlabel("Signal Arm [Photons/s]")

  plt.show()
