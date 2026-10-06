#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Compute a beam model from Zernike coefficients or Gaussian parameters."""

import sys

import numpy as np
import pandas as pd

from beamlab21.config import load_config
from beamlab21.io import save_npz
from beamlab21.models import GenZTBeam, gaussian_pointwise, twoD_Gaussian
from beamlab21.paths import default_base_dir, resolve_path, resolve_under


def run(config_path, base_dir=None, coord_type=None):
    base_dir = base_dir if base_dir is not None else default_base_dir(config_path)
    config = load_config(config_path)
    if coord_type is not None:
        config["coord_type"] = coord_type
    _coord = config.get("coord_type", "cartesian")

    freq = config["frequency"]
    c = 3e8
    wvl = c / (freq * 1e6)
    Deff = config["aperture_diameter"]

    angextent = config["pixels"] * config["angular_res"]
    if _coord == "polar":
        # Polar output grid: x = r axis [0, angextent/2], y = theta axis [0, 2π)
        x = np.linspace(0, angextent / 2, config["pixels"])
        y = np.linspace(0, 2 * np.pi, config["pixels"], endpoint=False)
    else:
        x = np.linspace(-angextent / 2, angextent / 2, config["pixels"])
        y = x
    dtype = config["dtype"]

    print("Base directory:", base_dir)
    print("Config file path:", config_path)
    print("Frequency channel (MHz):", freq)

    if config["gen_gaussian_model"]:
        print("Generating Gaussian beam model using theoretical beamwidth...")
        amp = config["gaussian_amp"]
        lambdabyDcoef = config["gaussian_lamdabyD_coef"]
        sigx = np.rad2deg(lambdabyDcoef * (wvl / Deff))
        sigy = sigx
        xo, yo = config["gaussian_offset_x"], config["gaussian_offset_y"]
        tilt = config["gaussian_rotation"]
        gparams = amp, sigx, sigy, xo, yo, tilt
        if _coord == "polar":
            R, THETA = np.meshgrid(x, y)
            gmodel = gaussian_pointwise(R * np.cos(THETA), R * np.sin(THETA), gparams)
        else:
            gmodel = twoD_Gaussian(x, y, gparams)

        if config["add_noise2gaussian"]:
            np.random.seed(config["gaussian_noise_random_seed"])
            gnoise = np.random.normal(loc=config["gaussian_noise_mean"],
                                      scale=config["gaussian_noise_sigma"], size=gmodel.shape)
        else:
            gnoise = 0.0
            print("No random noise added to the Gaussian Model...\n")

        gmodel = (gmodel + gnoise).reshape(config["pixels"], config["pixels"])
        print("Computed Gaussian model! Make sure to save it!\n")

        if config["save_gaussian_model"]:
            goutput_dir = resolve_path(config["goutput_dir"], base_dir)
            path = save_npz(goutput_dir, config["goutput_name"], x=x, y=y, data=gmodel)
            print(f"Computed Gaussian model is saved! check your model in {path}")
        else:
            print("Computed Gaussian model is not saved! "
                  "Set save_gaussian_model to True in config_compute.yaml :)\n")

    if config["gen_zernike_model"]:
        print("Generating Zernike beam model using provided coefficients and scale parameters...")
        spfile = resolve_under(base_dir, config["scaleparam_dir"], config["scaleparam_file"])
        coeffile = resolve_under(base_dir, config["coef_dir"], config["coef_file"])
        ztgen = GenZTBeam(freq, x, y, dtype, coord_type=_coord)
        ztsp_df = pd.read_csv(spfile)
        sigx, sigy = ztsp_df["sigx"].to_numpy(), ztsp_df["sigy"].to_numpy()
        ztgen.load_coef(str(coeffile))
        ztgen.basisfunc(sigx, sigy)
        ztmodel = np.dot(ztgen.Basis.T, ztgen.coef)

        if config["add_noise2zernike"]:
            np.random.seed(config["zernike_noise_random_seed"])
            znoise = np.random.normal(loc=config["zernike_noise_mean"],
                                      scale=config["zernike_noise_sigma"], size=ztmodel.shape)
        else:
            znoise = 0.0
            print("No random noise added to the Zernike Model...\n")

        ztmodel = (ztmodel + znoise).reshape(config["pixels"], config["pixels"])
        print("Computed Zernike model! Make sure to save it!\n")

        if config["save_zernike_model"]:
            zoutput_dir = resolve_path(config["zoutput_dir"], base_dir)
            name = config["zoutput_name"] + config["zoutput_format"]
            path = save_npz(zoutput_dir, name, x=x, y=y, data=ztmodel)
            print(f"Computed Zernike model is saved! check your model in {path}")
        else:
            print("Computed Zernike model is not saved! "
                  "Set save_zernike_model to True in config_compute.yaml :)\n")

    if not config["gen_gaussian_model"] and not config["gen_zernike_model"]:
        print("Set one of gen_gaussian_model / gen_zernike_model to be true!\n")
        print("Exiting without any computing ... :(\n")
        sys.exit()


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_compute.yaml"
    run(config_path)


if __name__ == "__main__":
    main()
