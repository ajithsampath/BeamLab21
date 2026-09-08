# Beam Lab 21

Beam computation / characterization tool for 21cm arrays (inspired by HIRAX).

This tool decomposes a measured/simulated beam into a 2D Gaussian main lobe plus a
Zernike-transform (Bessel) basis, and can regenerate a beam model from saved
coefficients. See the paper linked under [Publications](#publications) for the method.

---

## Install

```bash
pip install -e ".[dev]"      # editable install + dev tools (pytest, ruff)
```

## Example data

The example beam cube `data/Example_cube.npz` (~39 MB) is **not** distributed with
the repo (removed for confidentiality). It can be provided on request — contact the
collaboration (see [Contact](#contact)).

Once you have a copy, drop it at `data/Example_cube.npz`. To download it from a
location you control, configure the URL in any of these ways (checked in this
order): a `--url` flag, the `BEAMLAB21_DATA_URL` environment variable, a
`data/DATA_URL.txt` file (one line), or the `EXAMPLE_DATA_URL` constant in
[`src/beamlab21/data.py`](src/beamlab21/data.py); then run `beamlab21 fetch-data`.

## Usage (command line)

```bash
beamlab21 fetch-data                           # obtain the example cube (see above)
beamlab21 fit      configs/config_fit.yaml     # fit Gaussian + Zernike models
beamlab21 compute  configs/config_compute.yaml # regenerate a model from saved coefficients
```

- `beamlab21 --help` / `beamlab21 <cmd> --help` for all options.
- `--base-dir DIR` overrides where relative paths in the config resolve (defaults to
  the config's project directory, else the current directory).
- `beamlab21-fit` / `beamlab21-compute` are standalone equivalents of the
  `fit` / `compute` subcommands.

### Configuration

Everything is driven by the two YAML files in [`configs/`](configs/); read the
inline comments on each parameter before a run. The ones you will usually touch:

| Parameter | File | Meaning |
|---|---|---|
| `frequency` | both | frequency channel (MHz) to work on |
| `N` | `config_fit.yaml` | number of Zernike modes to fit |
| `skip_minimise` | `config_fit.yaml` | skip scale-parameter optimisation (fast; recommended on a laptop) |
| `save_params` | `config_fit.yaml` | write `coefficients_*.csv` / `scaleparameters_*.csv` |
| `pixels`, `angular_res` | `config_compute.yaml` | output grid size / resolution |

Results (models, coefficients, plots) are written to `outputs/` (git-ignored).

### From Python

```python
from beamlab21 import fit, compute
fit.run("configs/config_fit.yaml")
compute.run("configs/config_compute.yaml")
```

## Package layout

| Module | Responsibility |
|---|---|
| `beamlab21.config`   | load Jinja2-templated YAML configs |
| `beamlab21.paths`    | resolve input/output paths relative to a base directory |
| `beamlab21.io`       | read beam cubes, write `.npz` results |
| `beamlab21.zernike`  | Noll ⇄ quantum index bookkeeping |
| `beamlab21.models`   | analytic 2D Gaussian, generative Zernike-transform beam |
| `beamlab21.fitting`  | `GaussianFit`, `ZernikeFit` |
| `beamlab21.plotting` | diagnostic fit/residual plots |
| `beamlab21.data`     | obtain the example beam cube |
| `beamlab21.fit` / `beamlab21.compute` | the two analysis workflows |
| `beamlab21.cli`      | `beamlab21` command-line entry point |

`beamlab21.lib` is a deprecated shim that re-exports the above.

## Development

```bash
pytest                  # fast, data-free smoke tests
ruff check src tests
```

---

## License

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Publications

[![arXiv](https://img.shields.io/badge/arXiv-2412.09527-b31b1b.svg)](https://arxiv.org/abs/2412.09527)

This tool was used in the research article linked above.

## Contact

Ajith Sampath — [ajithsampath1997@gmail.com](mailto:ajithsampath1997@gmail.com)
