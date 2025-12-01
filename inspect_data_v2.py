import scipy.io
import numpy as np

mat = scipy.io.loadmat('assets/SFlab1_data/data.mat')

print("Mic locations (2x8):")
print(mat['mic_locations'])

print("\nPosition (3xN):")
print("First row (Time?):", mat['position'][0, :5], "...", mat['position'][0, -5:])
print("Second row (X?):", mat['position'][1, :5])
print("Third row (Y?):", mat['position'][2, :5])

print("\ntphat (8x136):")
print("First col:", mat['tphat'][:, 0])
print("Last col:", mat['tphat'][:, -1])
