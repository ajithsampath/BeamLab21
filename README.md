# Beam Lab 21

Beam computation/characterization tool for 21cm arrays (inspired by HIRAX).

`beamlab21` provides a small pipeline for:
- **Generating** synthetic beam models (Gaussian or Zernike-basis) on a regular pixel grid.
- **Fitting** a Gaussian + Zernike decomposition to measured beam data (e.g. EM simulations or drone flight data), producing reusable coefficient/scale-parameter files.
- **Simulating drone-based beam mapping**: generating a flight path and evaluating a beam model at the (scattered, non-gridded) path coordinates.

---

## Installation

```bash
git clone <repo-url>
cd BeamPackage
pip install -e .
```

This installs `beamlab21` (Python package under [src/beamlab21/](src/beamlab21/)) along with its dependencies (numpy, scipy, pyyaml, matplotlib, astropy, pandas, h5py, tqdm, jinja2).

---

## Package layout

| Module | Purpose |
|---|---|
| [lib.py](src/beamlab21/lib.py) | Core helpers: config loading, Zernike/Noll index conversions, `twoD_Gaussian`, `GaussianFit` and `ZernikeFit`/`GenZTBeam` classes for fitting and generating grid-based beams. |
| [compute.py](src/beamlab21/compute.py) | CLI entry point to **generate** a Gaussian or Zernike beam model from a config file. |
| [fit.py](src/beamlab21/fit.py) | CLI entry point to **fit** a Gaussian + Zernike model to beam data from a config file. |
| [drone.py](src/beamlab21/drone.py) | Drone-based beam mapping: generate a flight path (`create_drone_path`) and evaluate a Gaussian or Zernike beam model pointwise along it (`evaluate_gaussian_on_path` / `evaluate_zernike_on_path`), complementing the grid-based functions in `lib.py`. |

---

## Usage

Follow [Tutorial.ipynb](Tutorial.ipynb) for the full generate/fit workflow, and [tests/drone.ipynb](tests/drone.ipynb) for the drone mapping workflow.

### Generating a beam model

Edit [configs/config_compute.yaml](configs/config_compute.yaml) and run:

```bash
python -m beamlab21.compute configs/config_compute.yaml
```

### Fitting a beam model

Edit [configs/config_fit.yaml](configs/config_fit.yaml) and run:

```bash
python -m beamlab21.fit configs/config_fit.yaml
```

This fits a Gaussian main lobe followed by a Zernike-basis decomposition to the input data (e.g. [data/Example_cube.npz](data/Example_cube.npz)), and saves the resulting coefficients, scale parameters, and (optionally) plots to `outputs/`.

### Drone-based beam mapping

```python
from beamlab21.drone import create_drone_path, evaluate_gaussian_on_path, evaluate_zernike_on_path

# 1. Generate a zigzag flight path
coords = create_drone_path(width=150.0, height=150.0, dx=4, dy=1, ds=0.5, jitter=0.92, direction="NS")

# 2. Evaluate a Gaussian beam along the path
gaussian_beam = evaluate_gaussian_on_path(coords[:, 0], coords[:, 1], (amp, sigx, sigy, xo, yo, tilt))

# 3. Or evaluate a Zernike beam using coefficients/scale parameters produced by fit.py
zernike_beam = evaluate_zernike_on_path(coords[:, 0], coords[:, 1], coeffile="outputs/coefficients_400.csv", spfile="outputs/scaleparameters_400.csv")
```

`create_drone_path` supports `"EW"` (East-West scan lines stepping North-South) and `"NS"` (North-South scan lines stepping East-West) sweep directions, with configurable step size (`dx`, `dy`), sample spacing along the track (`ds`), and positional `jitter`. The Zernike coefficient/scale-parameter CSV files are the same ones produced by `fit.py` (`out_coef_name` / `out_sp_name` in [config_fit.yaml](configs/config_fit.yaml)).

---

## License

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Publications

[![arXiv](https://img.shields.io/badge/arXiv-2412.09527-b31b1b.svg)](https://arxiv.org/abs/2412.09527)

This tool was used in the research article linked above.

---

## Contact

Ajith Sampath  — [ajithsampath1997@gmail.com](mailto:ajithsampath1997@gmail.com)

---
