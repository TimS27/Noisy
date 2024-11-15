import sif_parser
import matplotlib.pyplot as plt
import numpy as np

# Open the .sif file
filepath = "your_file.sif"
data, info = sif_parser.np_open('split-detector-206-100-206-final3-10min.sif')  # dataset has 30000 spectra, measured for 10 min = 600 s with 50 Hz

# Get rid of unnecessary array structure of read sif file
flattened_arrays = [np.array(sublist).flatten() for sublist in data]
#print(len(flattened_arrays))

# Split data into reference and sample array, where each array within sample and reference is one spectrum at one time
# E.g. reference = [[reference_spectrum_t1], [reference_spectrum_t2], ...]
# Each row in reference or sample represents counts recorded at one time.
reference = []
sample = []

for i in range(len(flattened_arrays)):
    reference.append(flattened_arrays[i][:2048])
    sample.append(flattened_arrays[i][2048:])

reference = np.array(reference)

'''
plt.figure()
plt.plot(reference[1])
plt.plot(sample[1])
plt.show()
'''

def calculate_allan_variance(data):
    """
    Calculate Allan variance and deviation for spectral data.
    Each row in 'data' represents counts recorded at one time.
    """
    num_times, num_points = data.shape
    taus = np.unique(np.logspace(0, np.log10(num_times // 2), num=50, dtype=int))
    allan_variances = []

    for tau in taus:
        num_blocks = num_times // tau
        reshaped_data = data[:num_blocks * tau].reshape(num_blocks, tau, num_points)
        tau_means = np.mean(reshaped_data, axis=1)
        diff = np.diff(tau_means, axis=0)
        allan_var = 0.5 * np.mean(diff**2, axis=0)
        allan_variances.append(allan_var)

    allan_variances = np.array(allan_variances)
    allan_deviation = np.sqrt(allan_variances)
    return taus, allan_variances, allan_deviation

def plot_allan(taus, allan_variances, allan_deviation, num_points):
    """
    Plot Allan variance and deviation.
    """
    plt.figure(figsize=(12, 6))

    # Plot Allan variance
    plt.subplot(1, 2, 1)
    for i in range(num_points):
        plt.loglog(taus, allan_variances[:, i], label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau)')
    plt.ylabel('Allan Variance')
    plt.title('Allan Variance for Different Spectral Points')
    plt.grid()
    plt.legend()

    # Plot Allan deviation
    plt.subplot(1, 2, 2)
    for i in range(num_points):
        plt.loglog(taus, allan_deviation[:, i], label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau)')
    plt.ylabel('Allan Deviation')
    plt.title('Allan Deviation for Different Spectral Points')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

# Simulate example spectral data
np.random.seed(42)
num_times = 1000
num_points = 5  # Number of spectral points
spectral_data = np.random.poisson(lam=100, size=(num_times, num_points))

# Calculate Allan variance and deviation
taus, allan_variances, allan_deviation = calculate_allan_variance(reference) # or use spectral_data instead of reference

# Visualize results
plot_allan(taus, allan_variances, allan_deviation, num_points)