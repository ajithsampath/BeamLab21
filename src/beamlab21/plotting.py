#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: BeamLab21

"""Diagnostic plots for Zernike fits."""

import os

import matplotlib.pyplot as plt
import numpy as np


def _save(fig_dir, plotname):
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, plotname)
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.clf()
    plt.close("all")
    return path


def plot_results_cart(data, model, freq, N, x, y, plot_format, plot_directory, plot_cmap):
    """Data / fit / percentage-residual panels on a Cartesian grid."""
    residue = data - model
    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    z1 = ax1.imshow(np.log(np.clip(data, 1e-10, None)), cmap=plot_cmap, extent=extent)
    ax1.grid(False)
    plt.colorbar(z1, ax=ax1, fraction=0.047)
    ax1.set_title("Simulated CST beam")

    z2 = ax2.imshow(np.log(np.clip(model, 1e-10, None)), cmap=plot_cmap, extent=extent)
    ax2.grid(False)
    plt.colorbar(z2, ax=ax2, fraction=0.047)
    ax2.set_title(f"Fit with {N} basis functions")
    ax2.get_yaxis().set_visible(False)

    z3 = ax3.imshow((residue / np.where(model == 0, 1, model)) * 100,
                    cmap="seismic", extent=extent)
    ax3.grid(False)
    plt.colorbar(z3, ax=ax3, fraction=0.047)
    ax3.set_title("Percentage Residuals")
    ax3.get_yaxis().set_visible(False)

    plt.tight_layout()
    print("Making the plot.....\n")
    return _save(plot_directory, f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}")


def plot_results_polar(data, model, freq, N, r, theta, plot_format, plot_directory, plot_cmap):
    """Data / fit / percentage-residual panels on a polar grid.

    ``r`` and ``theta`` are the 1-D axis arrays (shape ``(nx,)`` and ``(ny,)``).
    The function builds the 2-D scatter coordinates internally.
    """
    ms = 1.5
    R, THETA = np.meshgrid(r, theta)    # shape (ny, nx) — matches data
    rho_flat = R.flatten()
    phi_flat = THETA.flatten()

    residue = data - model
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(131, projection="polar")
    ax2 = plt.subplot(132, projection="polar")
    ax3 = plt.subplot(133, projection="polar")

    z1 = ax1.scatter(phi_flat, rho_flat, c=np.log(np.clip(data.flatten(), 1e-10, None)),
                     cmap=plot_cmap, s=ms)
    ax1.grid(False)
    plt.colorbar(z1, ax=ax1, fraction=0.047)
    ax1.set_title("Beam data")

    z2 = ax2.scatter(phi_flat, rho_flat, c=np.log(np.clip(model.flatten(), 1e-10, None)),
                     cmap=plot_cmap, s=ms)
    ax2.grid(False)
    plt.colorbar(z2, ax=ax2, fraction=0.047)
    ax2.set_title(f"Fit with {N} basis functions")

    z3 = ax3.scatter(phi_flat, rho_flat,
                     c=(residue / np.where(model == 0, 1, model)).flatten() * 100,
                     cmap="seismic", s=ms)
    ax3.grid(False)
    plt.colorbar(z3, ax=ax3, fraction=0.047)
    ax3.set_title("Percentage Residual")

    plt.tight_layout()
    print("Making the plot.....\n")
    return _save(plot_directory, f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}")
