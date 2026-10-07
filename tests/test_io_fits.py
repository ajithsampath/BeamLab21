#Author: Ajith Sampath
#Affiliation: University of Geneva

"""Tests for FITS export in beamlab21.io."""

import numpy as np
from astropy.io import fits

from beamlab21.io import save_fits


class TestSaveFits:
    def _make_cube(self, n_freq=3, ny=31, nx=31):
        x = np.linspace(-30, 30, nx)
        y = np.linspace(-30, 30, ny)
        freq = np.array([400.0, 500.0, 600.0])[:n_freq]
        Xg, Yg = np.meshgrid(x, y)
        data = np.stack([
            np.exp(-(Xg**2 + Yg**2) / (2 * (20.0 - i)**2))
            for i in range(n_freq)
        ])
        return x, y, freq, data

    def test_file_created(self, tmp_path):
        x, y, freq, data = self._make_cube()
        path = str(tmp_path / "beam.fits")
        save_fits(path, x, y, freq, data)
        assert (tmp_path / "beam.fits").exists()

    def test_shape_preserved(self, tmp_path):
        x, y, freq, data = self._make_cube(n_freq=3, ny=25, nx=31)
        path = str(tmp_path / "beam.fits")
        save_fits(path, x, y, freq, data)
        with fits.open(path) as hdul:
            assert hdul[0].data.shape == data.shape

    def test_data_values_preserved(self, tmp_path):
        x, y, freq, data = self._make_cube()
        path = str(tmp_path / "beam.fits")
        save_fits(path, x, y, freq, data)
        with fits.open(path) as hdul:
            assert np.allclose(hdul[0].data, data)

    def test_frequency_wcs(self, tmp_path):
        x, y, freq, data = self._make_cube()
        path = str(tmp_path / "beam.fits")
        save_fits(path, x, y, freq, data)
        with fits.open(path) as hdul:
            hdr = hdul[0].header
            assert hdr["CTYPE3"] == "FREQ"
            assert abs(hdr["CRVAL3"] - freq[0] * 1e6) < 1.0

    def test_overwrite(self, tmp_path):
        x, y, freq, data = self._make_cube()
        path = str(tmp_path / "beam.fits")
        save_fits(path, x, y, freq, data)
        save_fits(path, x, y, freq, data * 2, overwrite=True)  # must not raise
        with fits.open(path) as hdul:
            assert np.allclose(hdul[0].data, data * 2)

    def test_creates_parent_dir(self, tmp_path):
        x, y, freq, data = self._make_cube()
        path = str(tmp_path / "subdir" / "beam.fits")
        save_fits(path, x, y, freq, data)
        assert (tmp_path / "subdir" / "beam.fits").exists()
