import sif_parser
import matplotlib.pyplot as plt
import numpy as np

# Open the .sif file
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
sample = np.array(sample)
difference = reference - sample

'''
# Plot first recorded sample and reference spectrum
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
    weighted_allan_variances = []

    # Calculate x-axis
    spectra_per_second = 50
    time = taus / spectra_per_second

    for tau in taus:
        num_blocks = num_times // tau
        reshaped_data = data[:num_blocks * tau].reshape(num_blocks, tau, num_points)
        tau_means = np.mean(reshaped_data, axis=1)
        diff = np.diff(tau_means, axis=0)
        allan_var = 0.5 * np.mean(diff**2, axis=0)  # has shape 2048, one allan variance for each spectral point
        allan_variances.append(allan_var)
    
        # Calculate weights based on (intensity per tau)/(total intensity of spectrum)
        total_intensity = np.sum(tau_means)
        weights = np.sum(tau_means, axis=0) / total_intensity

        # Compute weighted Allan variance
        weighted_allan_variance = np.sum(weights * allan_var)
        weighted_allan_variances.append(weighted_allan_variance)

    allan_variances = np.array(allan_variances)
    allan_deviation = np.sqrt(allan_variances)
    weighted_allan_variances = np.array(weighted_allan_variances)
    weighted_allan_deviation = np.sqrt(weighted_allan_variances)

    # Conversion factor for variance to OD
    I0 = np.mean(data, axis=0)
    print(I0.shape)
    conversion_factor = (-1 / np.log(10))**2 / I0**2
    allan_variances_mOD = allan_variances * conversion_factor * (1000**2)
    print(allan_variances_mOD.shape)

    return taus, time, allan_variances, allan_deviation, weighted_allan_variances, weighted_allan_deviation, allan_variances_mOD

    # Non-vectorized variant (slow) weight calculation
    #for j in range(num_times):
    #    for i in range(num_points):
    #        weights[j][i] = (data[j][i] / np.sum(data[j]))
    #print(weights)


def plot_allan(time, allan_variances, allan_deviation, weighted_allan_variance, allan_variances_mOD,num_points):
    """
    Plot Allan variance, deviation, and weighted Allan variance for whole spectra.
    """
    plt.figure(figsize=(20, 5))

    # Plot Allan variance
    plt.subplot(1, 4, 1)
    for i in range(num_points):
        plt.loglog(time, allan_variances[:, i], label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau) [s]')
    plt.ylabel('Allan Variance '+ r'$[\frac{counts^{2}}{(0,02s)^{2}}]$')
    plt.title('Allan Variance for Different Spectral Points')
    plt.grid()
    plt.legend()

    # Plot Allan deviation
    plt.subplot(1, 4, 2)
    for i in range(num_points):
        plt.loglog(time, allan_deviation[:, i], label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau) [s]')
    plt.ylabel('Allan Deviation '+ r'$[\frac{counts}{(0,02s)}]$')
    plt.title('Allan Deviation for Different Spectral Points')
    plt.grid()
    plt.legend()

    # Plot Average Allan deviation
    plt.subplot(1, 4, 3)
    plt.loglog(taus, weighted_allan_variance)# label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau) [s]')
    plt.ylabel('Allan Variance '+ r'$[\frac{counts^{2}}{(0,02s)^{2}}]$')
    plt.title('Weighted Allan Variance')
    plt.grid()
    plt.legend()

    # Plot Allan variance in mOD
    plt.subplot(1, 4, 4)
    for i in range(num_points):
        plt.loglog(time, allan_variances_mOD[:, i], label=f'Point {i + 1}')
    plt.xlabel('Averaging time (tau) [s]')
    plt.ylabel('Allan Variance [mOD]')
    plt.title('Allan Variance for Different Spectral Points')
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
taus, time, allan_variance, allan_deviation, weighted_allan_variance, weighted_allan_deviation, allan_variances_mOD = calculate_allan_variance(difference) # or use spectral_data instead of reference

# Visualize results
plot_allan(time, allan_variance, allan_deviation, weighted_allan_variance, allan_variances_mOD, num_points)