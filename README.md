# Beam Lab 21

Beam computation / characterization tool for 21cm arrays (inspired by HIRAX).

---

## Install

```bash
pip install -e ".[dev]"      # editable install + dev tools (pytest, ruff, nbstripout)
```

## Example data

The example beam cube `data/Example_cube.npz` (~39 MB) is **not** distributed with
the repo (removed for confidentiality). It can be provided on request — contact the
collaboration (see [Contact](#contact)).

Once you have a copy, point the tool at it in any of these ways (checked in this
order): a `--url` flag, the `BEAMLAB21_DATA_URL` environment variable, a
`data/DATA_URL.txt` file (one line), or the `EXAMPLE_DATA_URL` constant in
[`src/beamlab21/data.py`](src/beamlab21/data.py). Then:

```bash
beamlab21 fetch-data                 # or: python scripts/fetch_example_data.py
```

If you already have the `.npz` locally, just drop it at `data/Example_cube.npz`
(or pass `--base-dir` / an absolute `datafile` in the config) and skip the fetch.

## Quickstart

```bash
beamlab21 fetch-data                          # download the example cube (~39 MB)
beamlab21 fit      configs/config_fit.yaml    # fit Gaussian + Zernike models
beamlab21 compute  configs/config_compute.yaml # recompute a model from saved coefficients
```

Run `beamlab21 --help` (or `beamlab21 <cmd> --help`) for options such as `--base-dir`.
The standalone `beamlab21-fit` / `beamlab21-compute` commands still work and are
equivalent to the `fit` / `compute` subcommands.

Results are written to `outputs/` (git-ignored). The same entry points are available
from Python:

```python
from beamlab21 import fit, compute
fit.run("configs/config_fit.yaml")
compute.run("configs/config_compute.yaml")
```

Paths in the configs are resolved relative to the config file's project directory
(or the current working directory), so the package runs from anywhere.

See [`Tutorial.ipynb`](Tutorial.ipynb) for a walk-through.

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
| `beamlab21.data`     | download the example beam cube |
| `beamlab21.fit` / `beamlab21.compute` | the two analysis workflows |
| `beamlab21.cli`      | `beamlab21` command-line entry point |

`beamlab21.lib` is a deprecated shim that re-exports the above.

## Development

```bash
pytest          # fast, data-free smoke tests
ruff check src tests
nbstripout Tutorial.ipynb    # strip notebook outputs before committing
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
