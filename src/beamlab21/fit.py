#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Fit Gaussian / Zernike models to EM-sim or drone beam data.

Returns fit parameters (Gaussian parameters or Zernike coefficients) and a model
beam, for a single frequency channel.
"""

import os
import sys

import numpy as np
import pandas as pd

from beamlab21.config import load_config
from beamlab21.fitting import GaussianFit, ZernikeFit
from beamlab21.io import save_npz
from beamlab21.paths import default_base_dir, resolve_path, resolve_under
from beamlab21.plotting import plot_results_cart
from beamlab21.zernike import NollToQuantum, reorder_coef


def run(config_path, base_dir=None):
    base_dir = base_dir if base_dir is not None else default_base_dir(config_path)
    config = load_config(config_path)

    telescope_name = config["Telescope_name"]
    datafile = resolve_under(base_dir, config["data_dir"], config["datafile"])
    freq = config["frequency"]
    fac = config["fac"]
    plot_results = config["plot_results"]
    init_gparams = np.array(config["init_gparams"])
    save_params = config["save_params"]

    print("Base directory:", base_dir)
    print("Config file path:", config_path)
    print("Fitting data from file:", datafile)
    print("Telescope:", telescope_name)
    print("Frequency channel (MHz):", freq)

    gfit = GaussianFit(str(datafile), freq, error_type=config["gaussian_error_type"],
                       normalize_data=config["normalize_data"], coord_type=config["coord_type"])

    print("Data loaded. Shape of observed data:", gfit.data.shape)
    print("Fitting a 2D Gaussian to calculate beam width...\n")
    print("Starting Gaussian fit optimization...\n")

    x, y, xo, yo, freq_arr, freq, sigx_gopt, sigy_gopt, gExpected, data, _ = gfit.optimize_Gauss(
        init_gparams, minimize_method=config["gminimize_method"],
        xtol=config["gtol"], verbose=config["gverbose"])

    print("Gaussian fit completed. Optimized sigx:", sigx_gopt, "sigy:", sigy_gopt)

    if config["save_gaussian_model"]:
        goutput_dir = resolve_path(config["goutput_dir"], base_dir)
        save_npz(goutput_dir, config["goutput_name"], x=x, y=y, xo=xo, yo=yo, model=gExpected)
    else:
        print("Fitted main lobe model is not saved! "
              "Set save_gaussian_model to True in config_fit.yaml :)\n")

    print("Generating Zernike basis...\n")
    ztfit = ZernikeFit(x, y, xo, yo, freq_arr, freq, data, config["N"],
                       error_type=config["zernike_error_type"],
                       normalize_data=config["normalize_data"], coord_type=config["coord_type"])

    init_ztparams = [sigx_gopt, sigy_gopt]

    if config["skip_minimise"]:
        sigx, sigy, coef, model_beam = ztfit.NO_optimize_ZT(init_ztparams, fac)
    else:
        print("Starting Zernike fit optimization by varying scaling parameters...\n")
        sigx, sigy, coef, model_beam, optfun = ztfit.optimize_ZT(
            init_ztparams, minimize_method=config["ztminimize_method"],
            xtol=config["zttol"], maxiter=config["ztmaxiter"])
        print("Zernike fit completed. Fit parameters:", coef)

    if save_params:
        out_coef_dir = resolve_path(config["out_coef_dir"], base_dir)
        out_sp_dir = resolve_path(config["out_sp_dir"], base_dir)

        coef_reordered = reorder_coef(coef)
        j = np.arange(len(coef_reordered))
        n_val, m_val = np.vectorize(NollToQuantum)(j)
        coef_jnm = np.column_stack((j, n_val, m_val, coef_reordered))
        coef_jnm = coef_jnm[coef_jnm[:, 3] != 0.0]

        os.makedirs(out_coef_dir, exist_ok=True)
        coef_path = os.path.join(out_coef_dir, config["out_coef_name"])
        pd.DataFrame(coef_jnm, columns=["j", "n", "m", "coef"]).to_csv(coef_path, index=False)

        os.makedirs(out_sp_dir, exist_ok=True)
        sp_path = os.path.join(out_sp_dir, config["out_sp_name"])
        sp_df = pd.DataFrame({"freq(MHz)": [freq], "sigx": [sigx], "sigy": [sigy]})
        sp_df.to_csv(sp_path, index=False)
        print(f"Scaling parameters are saved in {sp_path}!\n")

    if plot_results:
        plot_directory = resolve_path(config["plot_directory"], base_dir)
        plot_results_cart(gfit.data, model_beam, freq, config["N"], x, y,
                          config["plot_format"], str(plot_directory), config["plot_cmap"])
        print("Plotted and saved...!!!\n")
    else:
        print("The results are not plotted and hence not saved..!!!\n")

    if config["save_zernike_model"]:
        zoutput_dir = resolve_path(config["zoutput_dir"], base_dir)
        name = config["zoutput_name"] + config["zoutput_format"]
        save_npz(zoutput_dir, name, x=x, y=y, xo=xo, yo=yo, model=model_beam)
    else:
        print("Fitted model is not saved! Set save_zernike_model to True in config_fit.yaml :)\n")

    print("Decomposing/Fitting the beam for a single given frequency is done!!\n")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_fit.yaml"
    run(config_path)


if __name__ == "__main__":
    main()
