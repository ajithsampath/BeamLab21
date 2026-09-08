#Author: Ajith Sampath
#Affiliation: University of Geneva
#Project: HIRAX Beam package

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

    z1 = ax1.imshow(np.log(data), cmap=plot_cmap, extent=extent)
    ax1.grid(False)
    plt.colorbar(z1, ax=ax1, fraction=0.047)
    ax1.set_title("Simulated CST beam")

    z2 = ax2.imshow(np.log(model), cmap=plot_cmap, extent=extent)
    ax2.grid(False)
    plt.colorbar(z2, ax=ax2, fraction=0.047)
    ax2.set_title(f"Fit with {N} basis functions")
    ax2.get_yaxis().set_visible(False)

    z3 = ax3.imshow((residue / model) * 100, cmap="seismic", extent=extent)
    ax3.grid(False)
    plt.colorbar(z3, ax=ax3, fraction=0.047)
    ax3.set_title("Percentage Residuals")
    ax3.get_yaxis().set_visible(False)

    plt.tight_layout()
    print("Making the plot.....\n")
    return _save(plot_directory, f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}")


def plot_results_polar(data, model, freq, N, rho, phi, plot_format, plot_directory):
    """Data / fit / percentage-residual panels on a polar grid."""
    ms = 1.5
    residue = data - model
    plt.figure(figsize=(12, 6))
    ax1 = plt.subplot(131, projection="polar")
    ax2 = plt.subplot(132, projection="polar")
    ax3 = plt.subplot(133, projection="polar")

    z1 = ax1.scatter(phi, rho, c=np.log(data), cmap="inferno", s=ms)
    ax1.grid(False)
    plt.colorbar(z1, ax=ax1, fraction=0.047)
    ax1.set_title("Simulated CST beam")

    z2 = ax2.scatter(phi, rho, c=np.log(model), cmap="inferno", s=ms)
    ax2.grid(False)
    plt.colorbar(z2, ax=ax2, fraction=0.047)
    ax2.set_title(f"Fit with {N} basis functions")
    ax2.get_yaxis().set_visible(False)

    z3 = ax3.scatter(phi, rho, c=(residue / model) * 100, cmap="seismic", s=ms)
    ax3.grid(False)
    plt.colorbar(z3, ax=ax3, fraction=0.047)
    ax3.set_title("Percentage Residual")
    ax3.get_yaxis().set_visible(False)

    plt.tight_layout()
    print("Making the plot.....\n")
    return _save(plot_directory, f"ZernikeFitResults_{freq}MHz with N={N}{plot_format}")
