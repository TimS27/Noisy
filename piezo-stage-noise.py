import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 26,
    "text.latex.preamble": r"\usepackage{lmodern}"
})

data = np.load("E:\\Measurements/46/2025-05-02/nirvana-dark-noise-with-piezo-stage-moving-in-1MS-2s.npy")

time = data[:,0][270000:287500]
time_no_offset = time - data[:,0][270000]
sig = data[:,1][270000:287500]

#print(data[:,0])

plt.title("BD Excess Noise based on Piezo Electromagnetic Fields")
plt.plot(time_no_offset, sig, label="BD Dark Noise")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()

# Background colors
closest = min(time_no_offset, key=lambda v: abs(v - 0.01626))
#half = time_no_offset[len(time_no_offset)//2]
plt.axvspan(time_no_offset[0], closest, facecolor='lightgray', alpha=0.3)
plt.axvspan(closest, time_no_offset[-1], facecolor='lightblue', alpha=0.3)

# Annotations
y_min, y_max = plt.ylim()
text_y = y_min + 0.05 * (y_max - y_min)
plt.text(time_no_offset[0] + 0.5 * (closest - time_no_offset[0]), text_y, 
         "Piezo stage not moving", ha='center', va='bottom')
plt.text(closest + 0.5 * (time_no_offset[-1] - closest), text_y, 
         "Piezo stage moving", ha='center', va='bottom')

plt.show()