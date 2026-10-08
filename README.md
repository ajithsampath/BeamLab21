# Beam Lab 21

Beam computation / characterization tool for 21cm arrays, inspired by and tested on HIRAX.

This tool decomposes a measured/simulated beam into a 2D Gaussian main lobe plus a
Zernike-transform (Bessel) basis, and can regenerate a beam model from saved
coefficients. See the paper linked under [Publications](#publications) for the method.

It also includes drone-based beam mapping (**under development**) — see
[Drone-based beam mapping](#drone-based-beam-mapping-under-development) below.

---

## Install

**Option A — pip (editable install):**

```bash
pip install -e ".[dev]"      # editable install + dev tools (pytest, ruff)
```

**Option B — Docker (no installation required):**

A pre-built image is published to the GitHub Container Registry on every release:

```bash
# pull the latest release
docker pull ghcr.io/ajithsampath/beamlab21:latest

# fit a beam cube (mount your data/ and outputs/ directories)
docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/configs:/app/configs \
    ghcr.io/ajithsampath/beamlab21:latest \
    fit configs/config_fit.yaml

# or with a specific version
docker pull ghcr.io/ajithsampath/beamlab21:0.1.0
```

With `docker compose` (uses the `docker-compose.yml` in the repo root):

```bash
docker compose run --rm beamlab21 fit    configs/config_fit.yaml
docker compose run --rm beamlab21 compute configs/config_compute.yaml
docker compose run --rm beamlab21 --help
```

## Input data

Fitting needs one beam cube file (`.npz` or `.h5`/`.hdf5`) — see
[**data/README.md**](data/README.md) for the exact format and how to point the
tool at your own data.

Both **Cartesian** and **polar** `(r, θ)` grids are supported. Set `coord_type:
"cartesian"` or `"polar"` in `config_fit.yaml`, or leave it as `"auto"` (the
default) to let the tool detect the coordinate system from the axis values. See
[**data/README.md**](data/README.md) for the array layout each coordinate type
expects.

The bundled example, `data/Example_cube.npz` (~39 MB), is **not** distributed with
the repo (removed for confidentiality). It can be provided on request — contact the
collaboration (see [Contact](#contact)). Once you have a copy, place it at
`data/Example_cube.npz`.

## Usage (command line)

```bash
beamlab21 fit      configs/config_fit.yaml       # fit one channel
beamlab21 fit-all  configs/config_fit.yaml       # fit every channel in the cube
beamlab21 compute  configs/config_compute.yaml   # regenerate model from saved coefficients
beamlab21 cst      data/farfield.txt --freq 400  # import a CST export and fit it
beamlab21 cst-stack f400.txt f500.txt \
          --freqs 400 500 --output data/cube.npz  # stack CST exports into one cube
```

- `beamlab21 --help` / `beamlab21 <cmd> --help` for all options.
- `--base-dir DIR` overrides where relative paths in the config resolve (defaults to
  the config's project directory, else the current directory).
- `--coord {auto,cartesian,polar}` (fit / fit-all) overrides the `coord_type` key in
  the config — useful when you want to run the same config file on both coordinate
  systems without editing it.
- `--coord {cartesian,polar}` (compute) sets the output grid coordinate system.
- `beamlab21-fit` / `beamlab21-compute` are standalone equivalents of the
  `fit` / `compute` subcommands.

### Fitting all frequency channels at once

`fit-all` loops over every channel in the beam cube, calls `fit.run` per channel,
and stacks the per-channel CSVs into `outputs/coefficients_all.csv` and
`outputs/scaleparameters_all.csv`:

```bash
# Fit all channels
beamlab21 fit-all configs/config_fit.yaml

# Fit a subset of channels (MHz)
beamlab21 fit-all configs/config_fit.yaml --channels 400 500 600

# Override the cube path without editing the config
beamlab21 fit-all configs/config_fit.yaml --datafile data/my_cube.npz
```

From Python:

```python
from beamlab21 import fit

fit.run_all("configs/config_fit.yaml")                         # all channels
fit.run_all("configs/config_fit.yaml", channels=[400, 500])   # subset
```

The stacked output `scaleparameters_all.csv` has columns `freq_mhz`, `sigx`,
`sigy`, `xo`, `yo`; `coefficients_all.csv` has columns `j`, `n`, `m`, `coef`,
`freq_mhz`. Both follow the same column layout as the per-channel files.

### Configuration

Everything is driven by the two YAML files in [`configs/`](configs/); read the
inline comments on each parameter before a run. The ones you will usually touch:

| Parameter                   | File                    | Meaning                                                           |
| --------------------------- | ----------------------- | ----------------------------------------------------------------- |
| `frequency`               | both                    | frequency channel (MHz) to work on                                |
| `N`                       | `config_fit.yaml`     | number of Zernike modes to fit                                    |
| `skip_minimise`           | `config_fit.yaml`     | skip scale-parameter optimisation (fast; recommended on a laptop) |
| `save_params`             | `config_fit.yaml`     | write `coefficients_*.csv` / `scaleparameters_*.csv`           |
| `pixels`, `angular_res` | `config_compute.yaml` | output grid size / resolution                                     |
| `coord_type`              | both                    | `"auto"` / `"cartesian"` / `"polar"` coordinate system           |

Results (models, coefficients, plots) are written to `outputs/` (git-ignored).

### From Python

```python
from beamlab21 import fit, compute
fit.run("configs/config_fit.yaml")
compute.run("configs/config_compute.yaml")

# Override the output coordinate system
compute.run("configs/config_compute.yaml", coord_type="polar")
```

## CST far-field import

BeamLab21 can read CST Studio far-field text exports (`.txt`) and fit them
directly.

```bash
beamlab21 cst data/farfield.txt --freq 400
```

This converts the CST export to a Cartesian beam cube, runs the standard
Gaussian + Zernike fit, and writes results to `outputs/` as usual.

### How the conversion works

CST exports each sample as `(theta, phi)` in degrees. The tool projects them
to Cartesian coordinates via `x = theta·cos(phi)`, `y = theta·sin(phi)`, then
interpolates (cubic) onto a regular grid. The amplitude is converted from
dB to linear (`10^(dB/20)`) before fitting.

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--freq MHz` | **required** | frequency of this export |
| `--column copol\|e` | `copol` | use `Abs(Copol)` or total `Abs(E)` |
| `--size N` | `1501` | output grid resolution (N×N pixels) |
| `--xy-max DEG` | `75.0` | half-width of Cartesian window in degrees |
| `--cube-output PATH` | *(temp)* | save the converted `.npz` cube here |
| `--config PATH` | `configs/config_fit.yaml` | fit config to use |
| `--skip-fit` | off | convert only; skip the fit pipeline |

`--skip-fit` is useful for inspecting the converted cube before fitting:

```bash
beamlab21 cst data/farfield.txt --freq 400 --cube-output data/cst_cube.npz --skip-fit
```

### Multi-frequency CST stacking

If you have a separate CST export for each frequency, `cst-stack` combines them
into a single multi-channel `.npz` beam cube which you can then pass to
`beamlab21 fit-all`:

```bash
beamlab21 cst-stack f400.txt f500.txt f600.txt \
          --freqs 400 500 600 \
          --output data/cst_cube.npz

beamlab21 fit-all configs/config_fit.yaml --datafile data/cst_cube.npz
```

From Python:

```python
from beamlab21.cst import stack

x, y, freq_arr, data = stack(
    ["f400.txt", "f500.txt", "f600.txt"],
    freqs_mhz=[400, 500, 600],
    column="copol",   # "copol" (default) or "e"
    size=1501,        # output grid resolution
    xy_max=75.0,      # half-window in degrees
)
# data.shape == (3, 1501, 1501)
```

### From Python

```python
from beamlab21 import cst

# Convert + fit in one call
cst.run("data/farfield.txt", freq_mhz=400)

# Convert only, keep the cube
x, y, freq_arr, data = cst.load_cst("data/farfield.txt", freq_mhz=400,
                                     column="copol", size=1501, xy_max=75.0)
```

---

## Beam metrics

After fitting, `beamlab21.metrics` computes standard beam characterisation
quantities from the fitted Gaussian scale parameters and the beam data:

```python
from beamlab21.metrics import hpbw, beam_solid_angle, main_lobe_efficiency, \
                               directivity, beam_summary

# Individual metrics
hpbw_x, hpbw_y = hpbw(sigx, sigy)              # half-power beam width (deg)
sa  = beam_solid_angle(data, x, y)              # ∫∫ B/B_max dx dy  (deg²)
eff = main_lobe_efficiency(data, x, y,
        xo, yo, sigx, sigy, fac=1.5)           # fraction of power² in main lobe
div = directivity(data)                          # peak / mean

# All at once
summary = beam_summary(sigx, sigy, data, x, y, xo=xo, yo=yo, fac=1.5)
# returns dict: hpbw_x, hpbw_y, beam_solid_angle, main_lobe_efficiency, directivity
```

| Quantity | Formula |
|---|---|
| HPBW | `2√(2 ln 2) · σ` ≈ 2.3548 σ |
| Beam solid angle | `∫∫ (B / B_max) dx dy` (trapezoidal rule) |
| Main-lobe efficiency | power² within an ellipse of radius `fac·σ` / total power² |
| Directivity | `B_max / ⟨B⟩` |

`sigx`, `sigy` come from the `scaleparameters_*.csv` files written by
`beamlab21 fit`, or directly from `GaussianFit.optimize_Gauss`.

### Beam chromaticity

Fit a chromatic model to beam width vs frequency. Three models are available:

```python
from beamlab21.metrics import fit_beam_chromaticity
import pandas as pd

sp = pd.read_csv("outputs/scaleparameters_all.csv")

# Power-law: σ(ν) = σ₀ · (ν/ν₀)^α
chrom = fit_beam_chromaticity(sp["freq_mhz"], sp["sigx"], sp["sigy"],
                              nu0_mhz=400, model="powerlaw")
print(chrom["alpha_x"])     # spectral index (≈ −1 for a diffraction-limited dish)

# Power-law + sinusoidal ripple: σ(ν) = σ₀·(ν/ν₀)^α · [1 + A·sin(2π(ν/ν₀)/P + φ)]
# Period P and amplitude A are seeded automatically from an FFT of the residuals.
chrom = fit_beam_chromaticity(sp["freq_mhz"], sp["sigx"], sp["sigy"],
                              model="powerlaw_ripple")
print(chrom["A_x"])         # ripple amplitude (fractional)
print(chrom["P_mhz_x"])     # ripple period in MHz  (≈ c / 2L for path-length L)

# λ/D theoretical beamwidth: σ(ν) = k · λ(ν) / D
# One free parameter k per axis; requires the dish diameter.
chrom = fit_beam_chromaticity(sp["freq_mhz"], sp["sigx"], sp["sigy"],
                              model="lambda_over_D", dish_diameter_m=6.0)
print(chrom["k_x"])         # illumination efficiency coefficient (≈ 1 for uniform)
```

| Model | Free params | Use when |
|---|---|---|
| `"powerlaw"` | σ₀, α | smooth chromatic trend |
| `"powerlaw_ripple"` | σ₀, α, A, P, φ | reflections / standing waves produce a ripple |
| `"lambda_over_D"` | k | you want the theoretical diffraction limit with a single coefficient |

### Aperture efficiency

```python
from beamlab21.metrics import aperture_efficiency

ae = aperture_efficiency(freq_mhz=400, sigx=20.0, sigy=18.0, dish_diameter_m=6.0)
print(ae["eta_ap"])       # e.g. 0.63
print(ae["A_eff_m2"])     # effective collecting area in m²
```

Uses the analytic Gaussian beam solid angle Ω_A = π·σx·σy / ln 2 (sr) and
η_ap = λ² / (4π · Ω_A · A_geom).

### FITS export

```python
from beamlab21.io import save_fits

save_fits("outputs/beam.fits", x, y, freq_arr_mhz, data)
```

Writes a 3-D FITS cube (freq, y, x) with WCS keywords for the spatial axes
(degrees) and the frequency axis (Hz).

### Airy diffraction pattern

Compute the theoretical `[2J₁(u)/u]²` diffraction pattern for a circular aperture
and optionally add a central blockage (annular aperture / secondary mirror):

```python
from beamlab21.compute import compute_airy_pattern
import numpy as np

x = np.linspace(-30, 30, 256)
y = np.linspace(-30, 30, 256)

# Unblocked aperture
airy = compute_airy_pattern(freq_mhz=400, aperture_diam_m=6.0, x=x, y=y)

# With subreflector blockage (annular aperture)
airy = compute_airy_pattern(freq_mhz=400, aperture_diam_m=6.0, x=x, y=y,
                             blockage_diam_m=1.0)
```

Can also be triggered via `config_compute.yaml` by setting
`gen_airy_pattern: True` and `airy_blockage_diameter: 1.0`. The aperture
diameter is taken from the existing `aperture_diameter` key.

---

## Package layout

```
beamlab21/
│
├── config.py          load Jinja2-templated YAML configs
├── paths.py           resolve input/output paths relative to a base directory
├── io.py              read/validate beam cubes (.npz/.h5); write .npz and FITS
├── zernike.py         Noll ⇄ quantum index bookkeeping
├── models.py          analytic 2D Gaussian, generative Zernike-transform beam
├── fitting.py         GaussianFit, ZernikeFit (grid-based)
├── plotting.py        diagnostic fit/residual plots
├── metrics.py         HPBW, solid angle, main-lobe efficiency, directivity,
│                          aperture efficiency, beam chromaticity (3 models)
├── cli.py             beamlab21 command-line entry point
│
├── fit/               Gaussian + Zernike fitting pipeline
│   ├── classes.py     GaussianFit, ZernikeFit
│   └── pipeline.py    run(), run_all()
│
├── compute/           Beam model synthesis
│   └── pipeline.py    run(), compute_airy_pattern()
│
├── cst/               CST far-field import
│   ├── load.py        load_cst(), stack()
│   └── pipeline.py    run()  (convert + fit in one call)
│
└── drone/             Drone-based beam mapping (under development)
    ├── sim_data.py    create_drone_path(), evaluate_gaussian_on_path(),
    │                      evaluate_zernike_on_path(), load_zernike_coef()
    └── fit_data.py    fit_beam_on_path(), fit_gaussian_on_path(),
                           fit_zernike_on_path(), DronePathFit
```

All public names are re-exported at the subpackage level, so both of these work:

```python
from beamlab21.fit import GaussianFit          # subpackage
from beamlab21.fitting import GaussianFit      # flat module (legacy, still supported)

from beamlab21.cst import load_cst, stack, run
from beamlab21.compute import compute_airy_pattern
from beamlab21.drone import create_drone_path
```

## Development

```bash
pytest                  # fast, data-free smoke tests
ruff check src tests
```

## Drone-based beam mapping (under development)

Simulate flying a drone through a beam: generate a flight path, then evaluate a
Gaussian or Zernike beam model at those (scattered, non-gridded) coordinates.

```python
from beamlab21.drone import create_drone_path, evaluate_gaussian_on_path, evaluate_zernike_on_path

# 1. Generate a zigzag flight path
coords = create_drone_path(width=150.0, height=150.0, dx=4, dy=1, ds=0.5, jitter=0.92, direction="NS")

# 2. Evaluate a Gaussian beam along the path
gaussian_beam = evaluate_gaussian_on_path(coords[:, 0], coords[:, 1], (amp, sigx, sigy, xo, yo, tilt))

# 3. Or evaluate a Zernike beam using coefficients/scale parameters produced by `beamlab21 fit`
zernike_beam = evaluate_zernike_on_path(coords[:, 0], coords[:, 1], coeffile="outputs/coefficients_400.csv", spfile="outputs/scaleparameters_400.csv")
```

`create_drone_path` supports `"EW"` (East-West scan lines stepping North-South) and `"NS"` (North-South scan lines stepping East-West) sweep directions, with configurable step size (`dx`, `dy`), sample spacing along the track (`ds`), and positional `jitter`. The Zernike coefficient/scale-parameter CSV files are the same ones produced by `beamlab21 fit` (`out_coef_name` / `out_sp_name` in [config_fit.yaml](configs/config_fit.yaml)). See [tests/drone.ipynb](tests/drone.ipynb) for the full worked example.

Going the other way — fitting a beam model *to* real drone-track measurements,
instead of simulating one — is handled by `beamlab21.drone.fit_data`:

```python
from beamlab21.drone import fit_beam_on_path

result = fit_beam_on_path(x, y, data)       # x, y, data: scattered track coordinates/signal
result.gaussian.params                       # amp, sigx, sigy, xo, yo, tilt_deg
result.zernike.coef                           # Zernike coefficients fit on top of the Gaussian
```

`fit_beam_on_path` runs a two-stage fit — a Gaussian main lobe
(`fit_gaussian_on_path`), then a Zernike basis on top of it
(`fit_zernike_on_path`), using the same math as `beamlab21.fitting`'s
grid-based `GaussianFit`/`ZernikeFit` (`beamlab21.models.gaussian_pointwise`,
`beamlab21.zernike.zernike_mode`) evaluated directly at the scattered
coordinates instead of on a grid. Both stages are also callable individually.
Developed and validated against real (private) flight-track data in
`tests/drone_fit_dev.ipynb` (not part of this repo's history).

---

## License

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Publications

[![ApJ](https://img.shields.io/badge/ApJ-10.3847%2F1538--4357%2Fae1b89-blue.svg)](https://iopscience.iop.org/article/10.3847/1538-4357/ae1b89)
[![arXiv](https://img.shields.io/badge/arXiv-2412.09527-b31b1b.svg)](https://arxiv.org/abs/2412.09527)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23235974.svg)](https://doi.org/10.5281/zenodo.23235974)

This tool was used in "Primary Beam Chromaticity in HIRAX. I. Characterization
from Simulations and Power Spectrum Implications," published in *The
Astrophysical Journal* (997, 1, 2026); see also the arXiv preprint above. See
[CITATION.cff](CITATION.cff) for citing the paper and/or this software.

## Contact

Ajith Sampath — [ajithsampath1997@gmail.com](mailto:ajithsampath1997@gmail.com)

Feel free to contact me for any difficulties, issues or bug reports — or raise them directly on [GitHub Issues](https://github.com/ajithsampath/BeamLab21/issues) :)

## Acknowledgements

Anthropic's Claude was used to assist with documentation and code structuring in this repository.
