import isfread_py3
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 20,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

data_sig_bal, header_sig_bal = isfread_py3.isfread("E:\\Measurements/46/2025-06-03-mod2\\1mW-new-tek0051CH2.isf")
data_sig_autobal, header_sig_autobal = isfread_py3.isfread("E:\\Measurements/46/2025-06-03-mod\\1mW-new-tek0052CH2.isf")

# Apply Savitzky–Golay smoothing (window_length must be odd)
smoothed_bal = savgol_filter(data_sig_bal[1][10000000:11000000], window_length=20001, polyorder=2)
smoothed_autobal = savgol_filter(data_sig_autobal[1][10000000:11000000], window_length=20001, polyorder=2)

plt.figure()
plt.title("Scan-Induced Beam Pointing Translates to BD Signal Modulation")
plt.plot(data_sig_bal[0][10000000:11000000], data_sig_bal[1][10000000:11000000], color="tab:blue", alpha=0.3, label="BHD Signal (Balanced)")
plt.plot(data_sig_bal[0][10000000:11000000], smoothed_bal[:1000000], color="tab:blue", label="Smoothed BHD Signal (Balanced)")
plt.plot(data_sig_autobal[0][10000000:11000000], data_sig_autobal[1][10000000:11000000], color="tab:orange", alpha=0.3, label="BHD Signal (Autoalanced)")
plt.plot(data_sig_autobal[0][10000000:11000000], smoothed_autobal[:1000000], color="tab:orange", label="Smoothed BHD Signal (Autoalanced)")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend(loc="lower right")
plt.show()