import scipy.io
import numpy as np

mat = scipy.io.loadmat('assets/SFlab1_data/data.mat')

print("Keys:", mat.keys())

if 'mic_locations' in mat:
    print(f"mic_locations shape: {mat['mic_locations'].shape}")
    print("mic_locations:\n", mat['mic_locations'])

if 'tphat' in mat:
    print(f"tphat shape: {mat['tphat'].shape}")
    print("tphat first col:", mat['tphat'][:, 0])

if 'position' in mat:
    print(f"position shape: {mat['position'].shape}")
    print("position first col:", mat['position'][:, 0])

if 'fs' in mat:
    print(f"fs: {mat['fs']}")
