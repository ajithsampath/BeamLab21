import h5py
import numpy as np

from beamlab21.io import load_beam

# A deliberately small, non-256x256 cube: load_beam must not assume any
# particular resolution or number of channels.
DATA = np.array(
    [[[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]],
     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
)  # shape (n_freq=2, ny=2, nx=3)
X = np.array([-1.0, 0.0, 1.0])
Y = np.array([-0.5, 0.5])
FREQ_GHZ = np.array([0.4, 0.405])
ERROR = np.full_like(DATA[0], 0.1)


def _assert_loaded_correctly(x, y, freq_arr, nchan, error, data):
    assert nchan == 2
    np.testing.assert_array_equal(x, X)
    np.testing.assert_array_equal(y, Y)
    np.testing.assert_allclose(freq_arr, FREQ_GHZ * 1e3)  # GHz -> MHz
    assert data.shape == (2, 2, 3)
    assert not np.isnan(data).any()  # NaNs zeroed
    assert data[0, 0, 2] == 0.0  # the NaN we injected


def test_load_beam_npz(tmp_path):
    path = tmp_path / "cube.npz"
    np.savez(path, data=DATA, x=X, y=Y, freq=FREQ_GHZ, error=ERROR)
    x, y, freq_arr, nchan, error, data = load_beam(str(path))
    _assert_loaded_correctly(x, y, freq_arr, nchan, error, data)
    np.testing.assert_array_equal(error, ERROR)


def test_load_beam_hdf5(tmp_path):
    for ext in (".h5", ".hdf5"):
        path = tmp_path / f"cube{ext}"
        with h5py.File(path, "w") as f:
            f["data"] = DATA
            f["x"] = X
            f["y"] = Y
            f["freq"] = FREQ_GHZ
            f["error"] = ERROR
        x, y, freq_arr, nchan, error, data = load_beam(str(path))
        _assert_loaded_correctly(x, y, freq_arr, nchan, error, data)
        np.testing.assert_array_equal(error, ERROR)


def test_load_beam_npz_and_hdf5_agree(tmp_path):
    npz_path = tmp_path / "cube.npz"
    np.savez(npz_path, data=DATA, x=X, y=Y, freq=FREQ_GHZ)
    h5_path = tmp_path / "cube.h5"
    with h5py.File(h5_path, "w") as f:
        f["data"] = DATA
        f["x"] = X
        f["y"] = Y
        f["freq"] = FREQ_GHZ

    *_, data_npz = load_beam(str(npz_path))
    *_, data_h5 = load_beam(str(h5_path))
    np.testing.assert_array_equal(data_npz, data_h5)


def test_load_beam_missing_error_dataset(tmp_path):
    path = tmp_path / "cube.npz"
    np.savez(path, data=DATA, x=X, y=Y, freq=FREQ_GHZ)
    *_, error, _ = load_beam(str(path))
    assert error is None


def test_load_beam_unsupported_extension(tmp_path):
    path = tmp_path / "cube.txt"
    path.write_text("not a beam cube")
    try:
        load_beam(str(path))
        raise AssertionError("expected NotImplementedError")
    except NotImplementedError as exc:
        assert ".npz" in str(exc) and ".h5" in str(exc)
