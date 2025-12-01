import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("=== Task 1: Sensor Calibration ===")
    
    # 1. Load Data
    file_path = 'assets/SFlab1_data/calibration-data.mat'
    # TODO: Use scipy.io.loadmat to load the file
    mat = scipy.io.loadmat(file_path)
    # TODO: Extract 'tphat' variable. 
    #       Shape should be (8, N) where 8 is # of mics, N is # of pulses.
    tphat = mat['tphat']  # Assuming 'tphat' is the correct key
    print(f"loaded tphat with shape: {tphat.shape}  (8 mics, N pulses)")
    
    # 2. Compute Measurement Errors
    # Concept: In the calibration dataset, microphones are placed equidistant from the speaker.
    # Ideally, the sound should arrive at all microphones at the exact same time for a given pulse.
    # Since we don't know the absolute "true" time, we can use the mean of the 8 mics as the reference.
    
    # TODO: Initialize a list or array to store all error samples
    
    # TODO: Loop through each pulse (each column in tphat):
    #       a. Calculate the mean arrival time for this specific pulse (across the 8 mics).
    #       b. Calculate the deviation of each mic from this mean: error = measured_time - mean_time.
    #       c. Collect these deviation values.
    mean_times = np.mean(tphat, axis=0, keepdims=True)
    deviations = tphat - mean_times
    errors = deviations.flatten() #single list of all errors
    


    # 3. Statistical Analysis
    # TODO: Calculate the Standard Deviation (sigma) of all the collected errors.
    #       Note: This value (sigma) is crucial! It tells you how much to trust your sensors.
    #       You will use sigma^2 in your covariance matrix (R or sensor.pe) in the C++ code.
    
    errors_std = np.sqrt( np.sum( np.power(errors, 2) ) / ( len(errors) - 1 ) )
    print(f"Standard deviation of measurements is sigma = {errors_std} (or numpys std = {np.std(errors)})")

    # TODO: Calculate the Bias (mean of all errors). Ideally, it should be close to 0.
    bias = np.mean(errors)
    print(f"The means of all errors (bias) is = {bias}")
    
    # 4. Visualization
    # TODO: Plot a histogram of the errors using plt.hist().
    #       Does it look like a Bell curve (Gaussian)?
    plt.hist(errors, bins=50, density=True, alpha=0.6, color='g') 
    plt.title('Histogram of Measurement Errors')
    plt.xlabel('Error (seconds)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
