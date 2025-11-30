#!/usr/bin/env python3
"""
Generate sample 3D data for testing the plotting script.

Creates:
  - artifacts/sensors_3d.csv
  - artifacts/targets_3d.csv
  - artifacts/estimates_3d.csv
  - artifacts/covariances_3d.npy
"""

import numpy as np
import os

# Ensure artifacts directory exists
os.makedirs('artifacts', exist_ok=True)

# Create 4 sensors in a tetrahedral configuration
sensors = np.array([
    [0.0, 0.0, 0.0],     # Origin
    [1.0, 0.0, 0.0],     # Along X
    [0.5, 0.866, 0.0],   # Triangle in XY plane
    [0.5, 0.433, 0.816]  # Above the plane (tetrahedron)
])

# Create 2 target positions in 3D
targets = np.array([
    [2.0, 1.5, 1.0],
    [1.5, 2.0, 0.5]
])

# Create estimated positions with small offsets
np.random.seed(42)
estimates = targets + np.random.randn(2, 3) * 0.1

# Create 3x3 covariance matrices for each estimate
covariances = []
for i in range(len(estimates)):
    # Create a random positive definite covariance matrix
    A = np.random.randn(3, 3) * 0.1
    cov = A @ A.T + np.eye(3) * 0.01
    covariances.append(cov)

covariances = np.array(covariances)

# Save to CSV files
np.savetxt('artifacts/sensors_3d.csv', sensors, delimiter=',', 
           header='x,y,z', comments='', fmt='%.6f')
np.savetxt('artifacts/targets_3d.csv', targets, delimiter=',', 
           header='x,y,z', comments='', fmt='%.6f')
np.savetxt('artifacts/estimates_3d.csv', estimates, delimiter=',', 
           header='x,y,z', comments='', fmt='%.6f')
np.save('artifacts/covariances_3d.npy', covariances)

print("Generated 3D sample data in artifacts/:")
print(f"  - sensors_3d.csv ({len(sensors)} sensors)")
print(f"  - targets_3d.csv ({len(targets)} targets)")
print(f"  - estimates_3d.csv ({len(estimates)} estimates)")
print(f"  - covariances_3d.npy ({len(covariances)} 3x3 matrices)")
print("\nTest with:")
print("  python3 tools/plot_estimates.py \\")
print("    --sensors artifacts/sensors_3d.csv \\")
print("    --targets artifacts/targets_3d.csv \\")
print("    --estimates artifacts/estimates_3d.csv \\")
print("    --cov artifacts/covariances_3d.npy \\")
print("    --3d --show")
