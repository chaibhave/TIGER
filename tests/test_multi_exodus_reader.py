import pytest
np = pytest.importorskip("numpy", exc_type=ImportError)
import MultiExodusReader as mer


class DummyExodusReader:
    def __init__(self, file_name):
        self.times = np.array([0.0, 1.0])
        self.x = np.array([[0.0], [1.0]])
        self.y = np.array([[0.0], [0.0]])
        self.z = np.array([[0.0], [0.0]])
        self.dim = 1

    def get_var_values(self, var_name, idx):
        return np.array([idx, idx + 1])


def test_get_data_at_time(monkeypatch):
    monkeypatch.setattr(mer.glob, "glob", lambda pattern: ["dummy.e"])
    monkeypatch.setattr(mer.ExodusReader, "ExodusReader", DummyExodusReader)
    mr = mer.MultiExodusReader("dummy.e")
    x, y, z, c = mr.get_data_at_time("c_Cr", 1.0)
    assert x.shape == (2, 1)
    assert y.shape == (2, 1)
    assert z.shape == (2, 1)
    assert np.array_equal(c, np.array([1, 2]))
