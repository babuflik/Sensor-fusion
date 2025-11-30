#!/usr/bin/env python3
"""
Plot sensor model, target positions, estimates and confidence ellipses.

Usage:
  python3 tools/plot_estimates.py --sensors sensors.csv --targets targets.csv \
      --estimates estimates.csv --cov covariances.npy

  For 3D plotting:
  python3 tools/plot_estimates.py --sensors sensors.csv --targets targets.csv \
      --estimates estimates.csv --cov covariances.npy --3d

Inputs:
  - `sensors.csv` rows: id,x,y[,z]
  - `targets.csv` rows: id,x,y[,z]
  - `estimates.csv` rows: id,x,y[,z]
  - `covariances.npy`: numpy array of shape (N,2,2) or (N,3,3) or a single covariance matrix

The script draws 95% confidence ellipses (2D) or ellipsoids (3D) for each covariance.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import csv
import math


def read_xy_csv(path):
    """Read CSV with x,y or x,y,z coordinates (with optional id column)."""
    rows = []
    with open(path, newline='') as f:
        r = csv.reader(f)
        for line_num, line in enumerate(r):
            if not line:
                continue
            # Skip header line if it contains non-numeric text
            if line_num == 0:
                try:
                    float(line[0])
                except (ValueError, IndexError):
                    continue  # Skip header
            
            # accept id,x,y[,z] or x,y[,z]
            if len(line) >= 4:
                # id,x,y,z
                _, x, y, z = line[:4]
                rows.append((float(x), float(y), float(z)))
            elif len(line) == 3:
                # could be id,x,y or x,y,z
                try:
                    x, y, z = line[:3]
                    rows.append((float(x), float(y), float(z)))
                except ValueError:
                    # probably id,x,y
                    _, x, y = line[:3]
                    rows.append((float(x), float(y)))
            else:
                # x,y
                x, y = line[:2]
                rows.append((float(x), float(y)))
    if not rows:
        return np.array([])
    arr = np.array(rows)
    # Ensure 2D array even for single row
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def confidence_ellipse(cov, mean, n_std=2.4477, **kwargs):
    """Create 2D confidence ellipse patch."""
    # n_std = 2.4477 approximates 95% for 2D (sqrt(chi2.ppf(0.95,2)))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * n_std * np.sqrt(vals)
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


def plot_3d_ellipsoid(ax, cov, mean, n_std=2.4477, color='C0', alpha=0.2):
    """Plot 3D confidence ellipsoid."""
    # n_std = 2.4477 approximates 95% for 3D
    vals, vecs = np.linalg.eigh(cov)
    # Generate sphere points
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones_like(u), np.cos(v))
    
    # Scale by eigenvalues and rotate by eigenvectors
    for i in range(len(x_sphere)):
        for j in range(len(x_sphere[0])):
            point = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
            # Scale
            point = point * n_std * np.sqrt(vals)
            # Rotate
            point = vecs @ point
            # Translate
            point = point + mean
            x_sphere[i, j], y_sphere[i, j], z_sphere[i, j] = point
    
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color=color, alpha=alpha, edgecolor='none')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sensors', required=True)
    p.add_argument('--targets', required=False)
    p.add_argument('--estimates', required=False)
    p.add_argument('--cov', required=False)
    p.add_argument('--show', action='store_true')
    p.add_argument('--outfile', default='artifacts/plot.png')
    p.add_argument('--3d', dest='three_d', action='store_true', help='Use 3D plotting')
    p.add_argument('--elev', type=float, default=30, help='3D plot elevation angle (degrees)')
    p.add_argument('--azim', type=float, default=-60, help='3D plot azimuth angle (degrees)')
    args = p.parse_args()

    # Determine if we should use 3D based on data or flag
    use_3d = args.three_d
    
    if args.three_d:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    if args.sensors:
        s = read_xy_csv(args.sensors)
        if s.size > 0:
            if use_3d and s.shape[1] == 3:
                ax.scatter(s[:, 0], s[:, 1], s[:, 2], marker='^', color='C2', label='Sensors', s=100)
                for i, (x, y, z) in enumerate(s):
                    ax.text(x, y, z, f'  S{i}', fontsize=8)
            else:
                ax.scatter(s[:, 0], s[:, 1], marker='^', color='C2', label='Sensors')
                for i, (x, y) in enumerate(s):
                    ax.annotate(f'S{i}', (x, y), textcoords='offset points', xytext=(4, 4))

    if args.targets:
        t = read_xy_csv(args.targets)
        if t.size > 0:
            if use_3d and t.shape[1] == 3:
                ax.scatter(t[:, 0], t[:, 1], t[:, 2], marker='x', color='C1', label='Targets', s=100)
            else:
                ax.scatter(t[:, 0], t[:, 1], marker='x', color='C1', label='Targets')

    covs = None
    if args.cov:
        covs = np.load(args.cov)
        # normalize shape
        if covs.ndim == 2:
            # Single covariance matrix
            covs = np.expand_dims(covs, 0)

    if args.estimates:
        e = read_xy_csv(args.estimates)
        if e.size > 0 and use_3d and e.shape[1] == 3:
            ax.scatter(e[:, 0], e[:, 1], e[:, 2], marker='o', color='C0', label='Estimates', s=100)
            for i, (x, y, z) in enumerate(e):
                ax.text(x, y, z, f'  E{i}', fontsize=8)
            if covs is not None:
                # Plot 3D ellipsoids
                n = len(e)
                if covs.shape[0] < n:
                    covs = np.vstack([covs] + [covs[0:1]] * (n - covs.shape[0]))
                for i in range(n):
                    cov = covs[i]
                    if cov.shape == (3, 3):
                        mean = e[i]
                        plot_3d_ellipsoid(ax, cov, mean, color='C0', alpha=0.15)
        elif e.size > 0:
            ax.scatter(e[:, 0], e[:, 1], marker='o', color='C0', label='Estimates')
            for i, (x, y) in enumerate(e):
                ax.annotate(f'E{i}', (x, y), textcoords='offset points', xytext=(4, -8))
            if covs is not None:
                # Plot 2D ellipses
                n = len(e)
                if covs.shape[0] < n:
                    covs = np.vstack([covs] + [covs[0:1]] * (n - covs.shape[0]))
                for i in range(n):
                    cov = covs[i]
                    if cov.shape == (2, 2):
                        mean = (e[i, 0], e[i, 1])
                        ell = confidence_ellipse(cov, mean, edgecolor='C0', facecolor='none', lw=1.5)
                        ax.add_patch(ell)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if use_3d:
        ax.set_zlabel('Z')
        ax.view_init(elev=args.elev, azim=args.azim)
    else:
        ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    ax.grid(True)
    # ensure output directory exists
    import os
    outdir = os.path.dirname(args.outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=200)
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
