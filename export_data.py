import scipy.io
import numpy as np
import csv

mat = scipy.io.loadmat('assets/SFlab1_data/data.mat')

# Export Mic Locations
mic_locs = mat['mic_locations'] # 2x8
np.savetxt('assets/SFlab1_data/mic_locations.csv', mic_locs.T, delimiter=',', header='x,y', comments='')

# Export Measurements (tphat)
# Transpose to have 136 rows (time steps) and 8 columns (mics)
tphat = mat['tphat'] # 8x136
np.savetxt('assets/SFlab1_data/tphat.csv', tphat.T, delimiter=',', header='m1,m2,m3,m4,m5,m6,m7,m8', comments='')

# Export Ground Truth
# Transpose to have N rows and 3 columns (t, x, y)
pos = mat['position'] # 3xN
np.savetxt('assets/SFlab1_data/ground_truth.csv', pos.T, delimiter=',', header='t,x,y', comments='')

print("Data exported to CSV files in assets/SFlab1_data/")
