# Example data

The default configs expect `data/Example_cube.npz` — a HIRAX beam cube of shape
`(n_freq, 256, 256)` with arrays `data`, `x`, `y`, `freq` (GHz).

This file (~39 MB) is **not** tracked in git — it was removed for confidentiality.
It can be provided on request; contact the collaboration (see the top-level README).

## Using your copy

If you have the `.npz`, place it at `data/Example_cube.npz` and you're done.

To download it from a location you control:

```bash
beamlab21 fetch-data          # or: python scripts/fetch_example_data.py
```

The URL is resolved in this order:

1. `beamlab21 fetch-data --url <URL>`
2. the `BEAMLAB21_DATA_URL` environment variable
3. `data/DATA_URL.txt` (one line, the URL; git-ignored)
4. `EXAMPLE_DATA_URL` in `src/beamlab21/data.py`

`data/*.npz` and `data/*.npy` are git-ignored.
