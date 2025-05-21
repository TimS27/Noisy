import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import matplotlib.font_manager as fm
#from matplotlib.font_manager import FontProperties

#import matplotlib.font_manager as fm
""" import matplotlib as mpl
print(mpl.get_cachedir()) """



# Load 2D intensity data (e.g., tab- or comma-separated)
""" beam_data = np.loadtxt("E:/Measurements/46/2025-04-24/beam-shape-arm1.csv", delimiter=';')  # adjust delimiter as needed
beam_data2 = np.loadtxt("E:/Measurements/46/2025-04-24/beam-shape-arm2.csv", delimiter=';') """
""" beam_data = np.loadtxt("E:/Measurements/46/2025-05-06/beam-profile-referencebeam-before-beam-splitter-no-header.csv", delimiter=';')  # adjust delimiter as needed
beam_data2 = np.loadtxt("E:/Measurements/46/2025-05-06/beam-profile-delayedbeam-before-beam-splitter-no-header.csv", delimiter=';') """
beam_data = np.loadtxt("E:/Measurements/46/2025-04-24/wincam-overlap.csv", delimiter=';')  # adjust delimiter as needed
beam_data2 = np.loadtxt("E:/Measurements/46/2025-04-24/wincam-iris2.csv", delimiter=';')


""" f, t_short, Zxx = signal.stft(calibration, sample_rate, nperseg=256)
plt.pcolormesh(t_short, f, np.abs(Zxx), norm=colors.LogNorm(vmin= np.abs(Zxx).max()*1e-8, vmax= np.abs(Zxx).max()))
plt.axhline(omega_expected/(2*np.pi), color='r')
plt.axhline(omega_expected/(2*np.pi)+(omega_expected/(2*np.pi))/4, color='r')
plt.axhline(omega_expected/(2*np.pi)-(omega_expected/(2*np.pi))/4, color='r')
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
 """
""" #fig, (axrin, axcum, axfosn, axcmrr) = plt.subplots(4, sharex=True, figsize=(10,10), gridspec_kw={'height_ratios': [2, 1, 1,1]})
fig, (ax1, ax2) = plt.subplots(2, sharey=True)
fig.suptitle('Power Spectral Density vs. Shot Noise')
ax1.imshow(beam_data, cmap='hot', origin='lower')  # other cmaps: gnuplot2, magma, hot, ...
ax1.colorbar(label='Intensity')
#plt.title("Beam Profile")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")

#plt.subplot(2,2,1, sharey=ax)
ax2.imshow(beam_data2, cmap='hot', origin='lower')  # other cmaps: gnuplot2, magma, hot, ...

plt.show() """


""" plt.rcParams.update({
    "font.family": "Latin Modern Roman",  # Use exact name from your list
    "font.size": 11,
    "text.usetex": False                  # Must be False to use .otf directly
}) """

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 11,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

""" # Create figure and grid layout
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Beam Profiles in LO (ScanDelay) and Sample Arm measured before Recombining Beamsplitter')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)#, height_ratios=[1,1,0.8])

# Subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])  # Colorbar axis

# Plot beam profiles
im1 = ax1.imshow(beam_data, cmap='hot', origin='lower')
ax1.set_title("LO Arm")
#ax1.set_ylabel("Y Position [Pixel]")

im2 = ax2.imshow(beam_data2, cmap='hot', origin='lower')
ax2.set_title("Sample Arm")

# Shared colorbar
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Intensity", fontsize="11")

# Common x-axis label
fig.supxlabel("X Position [Pixel]", fontsize=11)
fig.supylabel("Y Position [Pixel]", fontsize=11)

# --- Resize colorbar manually ---
# Get the current position
pos = cax.get_position()

# Adjust height and vertical position: [x0, y0, width, height]
# E.g. reduce height by 20% and center it
new_height = pos.height * 0.82
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, pos.width, new_height])

plt.tight_layout()
plt.show() """


# Create figure and grid layout
fig = plt.figure(figsize=(10, 5))
fig.suptitle('Overlapping Beamsplitter Outputs for optimal Interference')
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)#, height_ratios=[1,1,0.8])

# Subplots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
cax = fig.add_subplot(gs[2])  # Colorbar axis

# Plot beam profiles
im1 = ax1.imshow(beam_data, cmap='hot', origin='lower')
ax1.set_title("Displaced Beams")
#ax1.set_ylabel("Y Position [Pixel]")

im2 = ax2.imshow(beam_data2, cmap='hot', origin='lower')
ax2.set_title("Overlapped Beams")

# Shared colorbar
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Intensity", fontsize="11")

# Common x-axis label
fig.supxlabel("X Position [Pixel]", fontsize=11)
fig.supylabel("Y Position [Pixel]", fontsize=11)

# --- Resize colorbar manually ---
# Get the current position
pos = cax.get_position()

# Adjust height and vertical position: [x0, y0, width, height]
# E.g. reduce height by 20% and center it
new_height = pos.height * 0.82
new_y0 = pos.y0 + (pos.height - new_height) / 2
cax.set_position([pos.x0, new_y0, pos.width, new_height])

plt.tight_layout()
plt.show()