# Beam cube data

Fitting (`beamlab21 fit`) reads one beam cube file, pointed to by `data_dir` /
`datafile` in `configs/config_fit.yaml`. Any file matching the layout below works —
there's no fixed size, resolution, or file name requirement.

## File format

Two formats are supported, both read by
[`beamlab21.io.load_beam`](../src/beamlab21/io.py):

- **`.npz`** — a `numpy.savez` archive.
- **`.h5` / `.hdf5`** — an HDF5 file.

Either way, it must contain these four arrays/datasets, by name:

| Name | Shape | Meaning |
|---|---|---|
| `data` | `(n_freq, ny, nx)` | beam amplitude cube; `n_freq`, `ny`, `nx` can be anything |
| `x` | `(nx,)` | first coordinate axis |
| `y` | `(ny,)` | second coordinate axis |
| `freq` | `(n_freq,)` | frequency per channel, in **GHz** |

Plus one optional array:

| Name | Shape | Meaning |
|---|---|---|
| `error` | `(ny, nx)` | per-pixel measurement error (see "Error handling" below) |

### Cartesian coordinates

`x` and `y` are evenly or unevenly spaced axis arrays in degrees, arcminutes, or any
consistent angular unit. `data[f, j, i]` is the beam amplitude at `(x[i], y[j])`.
The axes may be asymmetric or not centred at zero.

Set `coord_type: "cartesian"` in `config_fit.yaml` (or leave it as `"auto"`; the
detector flags Cartesian whenever any `x` value is negative).

### Polar coordinates

`x` stores the radial axis `r` (1-D, all values `>= 0`) and `y` stores the azimuthal
axis `theta` (1-D, all values in `[0, 2π]` radians). `data[f, j, i]` is the beam
amplitude at `(r[i], theta[j])` — the same index convention as
`numpy.meshgrid(r, theta)`.

```python
import numpy as np

r     = np.linspace(0, 75, 51)                          # radial axis (nx = 51)
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)  # azimuth axis (ny = 64)

# build data on the (theta, r) grid — shape (n_freq, ny, nx)
R, THETA = np.meshgrid(r, theta)          # both (64, 51)
beam_slice = my_beam_function(R, THETA)   # (64, 51)
data = beam_slice[np.newaxis, :, :]       # (1, 64, 51)  — one frequency channel

np.savez("my_polar_beam.npz", x=r, y=theta, freq=np.array([0.4]), data=data)
```

Set `coord_type: "polar"` in `config_fit.yaml`. You can also leave it as `"auto"`:
the auto-detector identifies polar data when all `x >= 0` and all `y` are in
`[0, 2π]`; if your radial axis happens to satisfy both conditions but your data is
actually Cartesian, set `coord_type: "cartesian"` explicitly.

Internally the fitter converts the polar grid to Cartesian coordinates via
`X = R * cos(THETA)`, `Y = R * sin(THETA)` before evaluating the Gaussian and
Zernike models, so the physics is identical to a Cartesian fit.

**Error handling:** the `error` array is only used for the Gaussian fit, controlled
by `gaussian_error_type` in `config_fit.yaml`:
- `"uniform"` — `error` is ignored; a uniform error of 1 is used everywhere.
- anything else (e.g. `"proportional"`) — `error` is **required**; fitting fails
  with a clear error if it's missing from the file.

The Zernike fit always derives its own error from the data itself
(`zernike_error_type: "proportional"` or `"uniform"`) and never reads this field.

## Input validation

`load_beam` automatically validates the cube after loading and raises a descriptive
`ValueError` for common problems:

| Problem | Error message |
|---|---|
| `data` is not 3-D | `"data must be 3-D"` |
| `freq` length ≠ `data.shape[0]` | `"freq has N entries but data has M channels"` |
| `y` length ≠ `data.shape[1]` | `"y has N entries but data.shape[1] is M"` |
| `x` length ≠ `data.shape[2]` | `"x has N entries but data.shape[2] is M"` |
| all-NaN channel | `"channel K is all-NaN"` |

You can also call the validator directly:

```python
from beamlab21.io import _validate_beam_cube
_validate_beam_cube(x, y, freq_arr, data)   # raises ValueError on any problem
```

## Getting a cube

**The bundled example** — `data/Example_cube.npz`, a HIRAX example cube — is **not**
included in this repository (removed for confidentiality), but can be provided on
request; contact the collaboration (see the top-level README). Place the file
yourself at `data/Example_cube.npz`.

**Using your own cube** — put it anywhere and set `data_dir` / `datafile` in your
config to point at it (or pass a different config file to `beamlab21 fit`). Files
under `data/` matching `*.npz` or `*.npy` are git-ignored, so you can drop data
there without risk of accidentally committing it; `*.h5` / `*.hdf5` files are
git-ignored everywhere in the repo for the same reason.
