#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Fit Gaussian / Zernike models to EM-sim or drone beam data.

Returns fit parameters (Gaussian parameters or Zernike coefficients) and a model
beam, for a single frequency channel.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from beamlab21.config import load_config
from beamlab21.fitting import GaussianFit, ZernikeFit
from beamlab21.io import save_npz
from beamlab21.paths import default_base_dir, resolve_path, resolve_under
from beamlab21.plotting import plot_results_cart, plot_results_polar
from beamlab21.zernike import NollToQuantum, reorder_coef


def run(config_path, base_dir=None, coord_type=None, datafile=None, frequency=None):
    base_dir = base_dir if base_dir is not None else default_base_dir(config_path)
    # Re-render Jinja2 templates with the overridden frequency so that output
    # filenames (e.g. coefficients_{{ frequency }}.csv) pick up the right value.
    ctx = {"frequency": frequency} if frequency is not None else None
    config = load_config(config_path, context=ctx)
    if coord_type is not None:
        config["coord_type"] = coord_type
    if datafile is not None:
        # Absolute path overrides data_dir + datafile from config
        config["datafile"] = str(Path(datafile).resolve())
        config["data_dir"] = "."

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
        if gfit.coord_type == "polar":
            plot_results_polar(gfit.data, model_beam, freq, config["N"], x, y,
                               config["plot_format"], str(plot_directory), config["plot_cmap"])
        else:
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


def run_all(config_path, channels=None, base_dir=None, coord_type=None, datafile=None):
    """Fit all frequency channels in a beam cube.

    Calls :func:`run` for every channel (or a supplied subset), then stacks the
    per-channel coefficient and scale-parameter CSV files into combined outputs
    ``coefficients_all.csv`` / ``scaleparameters_all.csv`` in the same directory.

    Parameters
    ----------
    config_path : str
        Path to the fit YAML config.
    channels : list of float or None
        Frequencies in MHz to fit.  ``None`` (default) fits every channel in
        the cube.
    base_dir, coord_type, datafile :
        Same meaning as in :func:`run`.
    """
    _base = base_dir if base_dir is not None else default_base_dir(config_path)
    # Discover available channels from the cube
    _config0 = load_config(config_path)
    if datafile is not None:
        _cube_path = str(Path(datafile).resolve())
    else:
        _cube_path = str(resolve_under(_base, _config0["data_dir"], _config0["datafile"]))

    from beamlab21.io import load_beam as _lb
    _, _, freq_arr, *_ = _lb(_cube_path)

    if channels is None:
        channels = [float(f) for f in freq_arr]

    print(f"run_all: fitting {len(channels)} channel(s): "
          + ", ".join(f"{c:.1f}" for c in channels) + " MHz\n")

    for ch in channels:
        print(f"\n{'='*60}\nChannel {ch:.1f} MHz\n{'='*60}")
        run(config_path, base_dir=base_dir, coord_type=coord_type,
            datafile=datafile, frequency=ch)

    # Stack per-channel CSVs into combined files
    if not _config0.get("save_params", True):
        return

    coef_dfs, sp_dfs = [], []
    for ch in channels:
        _cfg = load_config(config_path, context={"frequency": ch})
        coef_path = os.path.join(str(resolve_path(_cfg["out_coef_dir"], _base)),
                                 _cfg["out_coef_name"])
        sp_path   = os.path.join(str(resolve_path(_cfg["out_sp_dir"], _base)),
                                 _cfg["out_sp_name"])
        if os.path.exists(coef_path):
            df = pd.read_csv(coef_path)
            df.insert(0, "freq_mhz", ch)
            coef_dfs.append(df)
        if os.path.exists(sp_path):
            df = pd.read_csv(sp_path)
            df.insert(0, "freq_mhz", ch)
            sp_dfs.append(df)

    combined_coef_path = None
    if coef_dfs:
        out_dir = str(resolve_path(_config0["out_coef_dir"], _base))
        combined_coef_path = os.path.join(out_dir, "coefficients_all.csv")
        pd.concat(coef_dfs, ignore_index=True).to_csv(combined_coef_path, index=False)
        print(f"\nStacked coefficients  → {combined_coef_path}")

    if sp_dfs:
        out_dir = str(resolve_path(_config0["out_sp_dir"], _base))
        combined_path = os.path.join(out_dir, "scaleparameters_all.csv")
        pd.concat(sp_dfs, ignore_index=True).to_csv(combined_path, index=False)
        print(f"Stacked scale params  → {combined_path}")



def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_fit.yaml"
    run(config_path)


if __name__ == "__main__":
    main()
