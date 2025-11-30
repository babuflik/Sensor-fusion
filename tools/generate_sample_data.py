#!/usr/bin/env python3
"""
Generate sample CSV/Numpy files to demo `plot_estimates.py`.

Generates:
  - sensors.csv
  - targets.csv
  - estimates.csv
  - covariances.npy

Run from repository root:
  python3 tools/generate_sample_data.py
"""
import numpy as np
import csv


def write_xy(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def main():
    sensors = [ ('s0', 0.0, 0.0), ('s1', 2.0, 0.2), ('s2', 1.0, 1.8) ]
    targets = [ ('t0', 1.0, 0.8) ]
    estimates = [ ('e0', 1.1, 0.9) ]
    covs = np.array([ [[0.02, 0.005], [0.005, 0.03]] ])

    import os
    out_dir = 'artifacts'
    os.makedirs(out_dir, exist_ok=True)
    write_xy(os.path.join(out_dir, 'sensors.csv'), sensors)
    write_xy(os.path.join(out_dir, 'targets.csv'), targets)
    write_xy(os.path.join(out_dir, 'estimates.csv'), estimates)
    np.save(os.path.join(out_dir, 'covariances.npy'), covs)
    print(f'Wrote {out_dir}/sensors.csv, {out_dir}/targets.csv, {out_dir}/estimates.csv, {out_dir}/covariances.npy')


if __name__ == '__main__':
    main()
