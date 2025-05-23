import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

# === Parameter ===
x = np.linspace(-5, 5, 1000)
alpha = 1  # amplitude of coherent state

# === Homodyne quadrature distribution ===

# Fock state |1⟩ quadrature distribution: P(x) = (1/sqrt(pi)) * x^2 * exp(-x^2)
P_fock_1 = (1 / np.sqrt(np.pi)) * x**2 * np.exp(-x**2)

# Coherent state |α⟩ quadrature distribution (approximately Gaussian around sqrt(2)*alpha)
P_coherent = (1 / np.sqrt(np.pi)) * np.exp(-(x - np.sqrt(2) * alpha)**2)

# === Photon-counting distributions ===

# Photon number distribution for Fock state |1⟩
n_vals_fock = np.arange(0, 10)
P_photon_fock = np.zeros_like(n_vals_fock, dtype=float)
P_photon_fock[1] = 1.0

# Photon number distribution for coherent state |α⟩ (Poissonian)
n_vals_coh = np.arange(0, 20)
P_photon_coh = poisson.pmf(n_vals_coh, mu=alpha**2)

# Create side-by-side plots
fig, axs = plt.subplots(ncols=2, figsize=(12, 4))

# Left: Homodyne Quadrature Distributions
axs[0].plot(x, P_fock_1, label=r"Fock state $|1\rangle$", linewidth=2)
axs[0].plot(x, P_coherent, label=r"Coherent state $|\alpha=1\rangle$", linewidth=2, linestyle='--')
axs[0].set_xlim(-4,4)
axs[0].set_title("Homodyne Quadrature Distributions", fontsize=22)
axs[0].set_xlabel("Quadrature amplitude $x$", fontsize=20)
axs[0].set_ylabel("Probability density $P(x)$", fontsize=20)
axs[0].legend()
axs[0].grid(True)

# Right: Photon-Counting Histograms
width = 0.35
axs[1].bar(n_vals_fock, P_photon_fock, width=width, label=r"Fock state $|1\rangle$")
axs[1].bar(n_vals_coh, P_photon_coh, width=width, alpha=1, label=r"Coherent state $|\alpha=1\rangle$")
axs[1].set_xlim(-0.5,8)
axs[1].set_ylim(0,1.2)
axs[1].set_title("Photon-Counting Histograms", fontsize=22)
axs[1].set_xlabel("Photon number $n$", fontsize=20)
axs[1].set_ylabel("Probability $P(n)$", fontsize=20)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

""" # === Plot 1: Homodyne Quadrature Distributions ===
plt.figure(figsize=(10, 5))
#plt.suptitle("")
ax1 = plt.subplot(2,1,1)
ax1.plot(x, P_fock_1, label=r"Fock state $|1\rangle$", linewidth=2)
ax1.plot(x, P_coherent, label=r"Coherent state $|\alpha=1\rangle$", linewidth=2, linestyle='--')
ax1.set_xlim(-4,4)
ax1.set_title("Homodyne Quadrature Distributions")
ax1.set_xlabel("Quadrature amplitude $x$")
ax1.set_ylabel("Probability density $P(x)$")
ax1.legend()
ax1.grid(True)


# === Plot 2: Photon-Counting Histograms ===
ax2 = plt.subplot(2,1,2)
ax2.bar(n_vals_fock - 0.2, P_photon_fock, width=0.4, label=r"Fock state $|1\rangle$")
ax2.bar(n_vals_coh + 0.2, P_photon_coh, width=0.4, alpha=0.6, label=r"Coherent state $|\alpha=1\rangle$")
ax2.set_xlim(0,10)
ax2.set_title("Photon-Counting Histograms")
ax2.set_xlabel("Photon number $n$")
ax2.set_ylabel("Probability $P(n)$")
ax2.legend()
ax2.grid(True)

#plt.tight_layout()
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