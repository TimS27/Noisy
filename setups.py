import sys, functools, itertools, uuid, warnings, textwrap, inspect
from typing import Union
from collections.abc import Iterable, Callable

import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import scipy.ndimage, scipy.signal

import joblib # pip install joblib
memory = joblib.Memory("cache-joblib")

import fourioso, fourioso.tools # pip install git+https://gitlab.com/leberwurscht/fourioso.git@v0.0.5
import chunkiter, chunkiter.tools # pip install git+https://gitlab.com/leberwurscht/chunkiter.git@v0.0.17
import fwhm # pip install git+https://gitlab.com/leberwurscht/fwhm.git@v0.0.1
import frshelpers.plot, frshelpers.ops # pip install git+https://gitlab.com/leberwurscht/frshelpers.git@v0.0.5

from .parts import *

def tims_setup(**kwargs):
  shot_noise = kwargs.get("shot_noise", False) # whether to consider shot noise
  OD_filter_transmissions = kwargs.get("OD_filter_transmissions", np.repeat(np.logspace(-7, -17, 15), 1))#[5.85000000e-05, 2.05335000e-08, 1.90086823e-10, 1.50348412e-11, 4.71833512e-12, 4e-13, 4e-14, 4e-15, 4e-16])
  power_lo = kwargs.get("power_lo", 500e-6)

  digitization_noise = kwargs.get("digitization_noise", False) # whether to consider the noise of the oscilloscope (in the code, this is desactivated for the calibration PD and only used for the balanced detector)
  digitization_max_level = kwargs.get("digitization_max_level", 5e-3) # the max level of the oscilloscope in V (goes from -max_level to +max_level. Example: 1mV/Div means 10 mV range, so max_level=5mV)
  digitization_enobs = kwargs.get("digitization_enobs", 6) # effective number of bits of the oscilloscope (check)

  calibration_phase = kwargs.get("calibration_phase", np.pi*.35) # phase shift between calibration and signal arm
  sample_rate = kwargs.get("sample_rate", 500e3)
  wavelength = kwargs.get("wavelength", 1064e-9)
  attenuation = kwargs.get("attenuation", 1) # we introduced this articicial attenuation (~factor 10) to get agreement with the observed responsivity*gain factor
  file_suffix = kwargs.get("file_suffix", "") # to append onlyshot etc. to the filenames of the npz files
  max_delay = kwargs.get("max_delay", 46e-12) # maximum delay of the delay stage motion (values will go from -max_delay to max_delay)
  scan_frequency = kwargs.get("scan_frequency", 0.8**-1) # frequency of the delay stage motion
  mirror_bandwidth = kwargs.get("delaystage_bandwidth", 0.07**-1) # inverse reaction time of the delay stage to velocity changes (to avoid/smooth out abrupt turnings)

  measurement_time = kwargs.get("measurement_time", 2.) # measurement time in seconds

  responsivity = kwargs.get("responsivity", .67e5) # product of responsivity (in A/W) and gain (V/A), of balanced detector
  dc_responsivity = kwargs.get("dc_responsivity", .67e5/5)
  responsivity_cal = kwargs.get("responsivity_cal", 4e1) # product of responsivity (in A/W) and gain (V/A), of calibration PD
  NEP = kwargs.get("NEP", 0)#3e-12) # NEP of balanced detector; data sheet says 3e-12
  NEP_cal = kwargs.get("NEP_cal", 0) # NEP of calibration PD - it probably won't affect our data in any way

  delay_jitter = kwargs.get("delay_jitter", False) # whether to add delay jitter (sinusoidal)
  max_delay_modulation = kwargs.get("max_delay_modulation", .02e-12) # maximum delay deviation of this jitter
  delay_modulation_frq = kwargs.get("delay_modulation_frq", 0.01**-1) # frequency of the sinusoidal jitter

  plot = kwargs.get("plot", True) # whether to plot the signals of the last entry in OD_filter_transmissions
  show = kwargs.get("show", True) # whether to call plt.show after plotting

  block_PD2 = kwargs.get("block_PD2", False) # whether to block the second photodiode
  disconnect_BNC = kwargs.get("disconnect_BNC", False) # whether to disconnect the balanced detector's BNC cable

  for j,OD_filter_transmission in enumerate(OD_filter_transmissions):
    power_signal = power_lo*OD_filter_transmission
  
    nu = constants.c/wavelength
  
    t = np.arange(int(measurement_time/sample_rate**-1))*sample_rate**-1
  
    delay = max_delay*scipy.signal.sawtooth(2*np.pi*t/scan_frequency**-1, width=0.5)

    # smooth the turning points
    kernel = np.ones(int(mirror_bandwidth**-1/sample_rate**-1))
    delay = scipy.signal.convolve(delay, kernel/kernel.sum(), mode="same")
  
    kernel = np.ones(int(mirror_bandwidth**-1*3/7/sample_rate**-1))
    delay = scipy.signal.convolve(delay, kernel/kernel.sum(), mode="same")
  
    # add jitter to the delay
    if delay_jitter: delay += max_delay_modulation*np.sin(2*np.pi*delay_modulation_frq*t)
  
  #  plt.figure()
  #  plt.plot(t[1:], abs(np.diff(delay)))
  #  plt.show()

    # Polarizations
    if 1:
      # Beamsplitter
      rel_I_LO_to_pd1_p = 440e-6 / 1000e-6
      rel_I_LO_to_pd2_p = 26e-6 / 1000e-6
      rel_I_LO_to_pd1_s = 0e-6 / 1000e-6
      rel_I_LO_to_pd2_s = 389e-6 / 1000e-6
      rel_I_S_to_pd1_p = 12e-6 / 500e-6
      rel_I_S_to_pd2_p = 246e-6 / 500e-6
      rel_I_S_to_pd1_s = 241e-6 / 500e-6
      rel_I_S_to_pd2_s = 0e-6 / 500e-6

      # PBS implementation
      omega = 2 * np.pi * nu
      k = omega/constants.c
      #np.exp(1j*omega*t) cancels
      #power_pd1_p = np.abs(np.sqrt(rel_I_LO_to_pd1_p * power_lo) * np.exp(1j*omega*t - 1j*k*delay*constants.c) + np.sqrt(rel_I_S_to_pd1_p * power_signal) * np.exp(1j*omega*t))**2#power_signal/2 + power_lo/2 + np.sqrt(power_signal*power_lo)*np.sin(2*np.pi*nu*delay)
      power_pd1_p = np.abs(np.sqrt(rel_I_LO_to_pd1_p * power_lo) * np.exp(- 1j*k*delay*constants.c) + np.sqrt(rel_I_S_to_pd1_p * power_signal))**2
      power_pd2_p = np.abs(np.sqrt(rel_I_LO_to_pd2_p * power_lo) * np.exp(- 1j*k*delay*constants.c) - np.sqrt(rel_I_S_to_pd2_p * power_signal))**2

      #power_pd1_p = rel_I_LO_to_pd1_p * power_lo + rel_I_S_to_pd1_p * power_signal + 2 * np.sqrt(rel_I_LO_to_pd1_p * power_lo * rel_I_S_to_pd1_p * power_signal) * np.cos(k*delay*constants.c)
      #power_pd2_p = rel_I_LO_to_pd2_p * power_lo + rel_I_S_to_pd2_p * power_signal - 2 * np.sqrt(rel_I_LO_to_pd2_p * power_lo * rel_I_S_to_pd2_p * power_signal) * np.cos(k*delay*constants.c)

      power_pd1_s = np.abs(np.sqrt(rel_I_LO_to_pd1_s * power_lo) * np.exp(- 1j*k*delay*constants.c) + np.sqrt(rel_I_S_to_pd1_s * power_signal))**2
      power_pd2_s = np.abs(np.sqrt(rel_I_LO_to_pd2_s * power_lo) * np.exp(- 1j*k*delay*constants.c) - np.sqrt(rel_I_S_to_pd2_s * power_signal))**2 

      power_calpd = power_lo/2 + power_lo/2 - np.sqrt(power_lo*power_lo)*np.sin(2*np.pi*nu*delay + calibration_phase)
      
      if shot_noise:
        power_pd1_p = apply_shot_noise(sample_rate, iter([power_pd1_p]), central_wavelength=wavelength)
        power_pd1_s = apply_shot_noise(sample_rate, iter([power_pd1_s]), central_wavelength=wavelength)
        power_pd2_p = apply_shot_noise(sample_rate, iter([power_pd2_p]), central_wavelength=wavelength)
        power_pd2_s = apply_shot_noise(sample_rate, iter([power_pd2_s]), central_wavelength=wavelength)
        power_calpd = apply_shot_noise(sample_rate, iter([power_calpd]), central_wavelength=wavelength)
      else:
        power_pd1_p = iter([power_pd1_p])
        power_pd1_s = iter([power_pd2_s])
        power_pd2_p = iter([power_pd2_p])
        power_pd2_s = iter([power_pd2_s])
        power_calpd = iter([power_calpd])

      power_pd1 = ((d1+d2) for d1,d2 in zip(power_pd1_p, power_pd1_s))#power_pd1_p + power_pd1_s
      power_pd2 = ((d1+d2) for d1,d2 in zip(power_pd2_p, power_pd2_s))#power_pd2_p + power_pd2_s

    else:
      power_pd1 = power_signal/2 + power_lo/2 + np.sqrt(power_signal*power_lo)*np.sin(2*np.pi*nu*delay)
      power_pd2 = power_signal/2 + power_lo/2 - np.sqrt(power_signal*power_lo)*np.sin(2*np.pi*nu*delay)
    
      power_calpd = power_lo/2 + power_lo/2 - np.sqrt(power_lo*power_lo)*np.sin(2*np.pi*nu*delay + calibration_phase)
    
      if shot_noise:
        power_pd1 = apply_shot_noise(sample_rate, iter([power_pd1]), central_wavelength=wavelength)
        power_pd2 = apply_shot_noise(sample_rate, iter([power_pd2]), central_wavelength=wavelength)
        power_calpd = apply_shot_noise(sample_rate, iter([power_calpd]), central_wavelength=wavelength)
      else:
        power_pd1 = iter([power_pd1])
        power_pd2 = iter([power_pd2])
        power_calpd = iter([power_calpd])
      
      power_pd1 = (pd1_*attenuation for pd1_ in power_pd1)
      power_pd2 = (pd2_*attenuation*int(not block_PD2) for pd2_ in power_pd2)




    """ plt.plot(chunkiter.tools.concatenate(power_pd1)[:50000])
    plt.plot(chunkiter.tools.concatenate(power_pd2)[:50000])
    plt.axhline(0)
    plt.show() """

  
  
  #  plt.figure()
  #  plt.plot(t, chunkiter.tools.concatenate(power_calpd)/1e-6)
  #  plt.show()
  
    power_pd1 = (pd1_*attenuation for pd1_ in power_pd1)
    power_pd2 = (pd2_*attenuation*int(not block_PD2) for pd2_ in power_pd2)
  
    # TOdo: NEP after differcence
    voltage_pd1 = photodiode(sample_rate, power_pd1, NEP_1Hz=NEP, responsivity=responsivity)
    #voltage_pd_1 = voltage_pd1 - np.mean(voltage_pd1) + np.mean(voltage_pd1) * (dc_responsivity / responsivity)
    voltage_pd2 = photodiode(sample_rate, power_pd2, NEP_1Hz=NEP, responsivity=responsivity)
    #voltage_pd_2 = voltage_pd2 - np.mean(voltage_pd2) + np.mean(voltage_pd2) * (dc_responsivity / responsivity)
    voltage_calpd = photodiode(sample_rate, power_calpd, NEP_1Hz=NEP_cal, responsivity=responsivity_cal)
  
    # Implementing different gains for dc and ac
    dc_freq = 50

    # AD DC gain difference
    if 1:
      voltage_pd1_copy1, voltage_pd1_copy2 = itertools.tee(voltage_pd1)  # creating copies
      voltage_pd1_dc = lowpass(sample_rate, voltage_pd1_copy1, dc_freq)
      voltage_pd1_dc_copy1, voltage_pd1_dc_copy2 = itertools.tee(voltage_pd1_dc)   # creating copies
      voltage_pd1_ac = ((d1-d2) for d1,d2 in zip(voltage_pd1_copy2, voltage_pd1_dc_copy1)) #voltage_pd1 - voltage_pd1_dc
      voltage_pd1 = ((d1+d2/5) for d1,d2 in zip(voltage_pd1_ac, voltage_pd1_dc_copy2))

      voltage_pd2_copy1, voltage_pd2_copy2 = itertools.tee(voltage_pd2)  # creating copies
      voltage_pd2_dc = lowpass(sample_rate, voltage_pd2_copy1, dc_freq)
      voltage_pd2_dc_copy1, voltage_pd2_dc_copy2 = itertools.tee(voltage_pd2_dc)   # creating copies
      voltage_pd2_ac = ((d1-d2) for d1,d2 in zip(voltage_pd2_copy2, voltage_pd2_dc_copy1))
      voltage_pd2 = ((d1+d2/5) for d1,d2 in zip(voltage_pd2_ac, voltage_pd2_dc_copy2))#voltage_pd2_ac + voltage_pd2_dc / 5

    """ # Plot mod depth
    voltage_pd1 = chunkiter.tools.concatenate(voltage_pd1)
    voltage_pd2 = chunkiter.tools.concatenate(voltage_pd2)

    plt.plot(voltage_pd1)
    plt.plot(voltage_pd2)
    plt.axhline(0)
    plt.show() """

    diff = ((d1-d2)*int(not disconnect_BNC) for d1,d2 in zip(voltage_pd1, voltage_pd2))

    # Detector saturation
    if 0:
      V_max = 200e-6 * responsivity # saturation power (diff between PDs) [W] * responsivity [V/W]
      diff = ((V_max * np.tanh(d1/ V_max))*int(not disconnect_BNC) for d1 in diff) #V_max * np.tanh(diff / V_max)
    #diff = (np.clip(d1, -V_max/2, V_max/2)*int(not disconnect_BNC) for d1 in zip(diff)) #V_max * np.tanh(diff / V_max)
  

    #diff = digitize(diff, max_level=5e-3, enobs=8, bits=11)
    #if not onlyshot: diff = digitize(diff, max_level=5e-3, enobs=8)
    if digitization_noise: diff = digitize(diff, max_level=digitization_max_level, enobs=digitization_enobs)

    diff = chunkiter.tools.concatenate(diff)
    calpd = chunkiter.tools.concatenate(voltage_calpd)
  
    #np.savez("tim.npz", diff=diff, cal=calpd)
    np.savez("test11/tim_{}_{:.5e}_{}.npz".format(file_suffix, OD_filter_transmission, j), diff=diff, cal=calpd, transmission=OD_filter_transmission, kwargs=kwargs)
  
  """ if plot:
    plt.figure()
    plt.plot(t,calpd)
    plt.title("calibration signal on oscilloscope")
  
    plt.figure()
    plt.plot(t,diff)
    plt.title("balanced-detection signal on oscilloscope")
    if show: plt.show() """