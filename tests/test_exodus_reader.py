import pytest
np = pytest.importorskip("numpy", exc_type=ImportError)
import ExodusReader as er


class DummyExodusReader:
    def __init__(self, file_name):
        if "1" in file_name:
            self.times = np.array([0.0, 1.0])
            self.nodal_var_names = ["c_Cr", "a"]
            self.elem_var_names = ["e1"]
        else:
            self.times = np.array([2.0, 3.0])
            self.nodal_var_names = ["b", "c_Cr"]
            self.elem_var_names = ["e2"]
        self.block_connect = {1: np.array([[1], [2]])}
        self.block_xyz = {
            1: (
                np.array([[0.0], [1.0]]),
                np.array([[0.0], [0.0]]),
                np.array([[0.0], [0.0]]),
            )
        }
        self.connect = self.block_connect[1]
        self.x, self.y, self.z = self.block_xyz[1]
        self.dim = 1

    def get_var_values(self, var_name, idx, block_id=None):
        return np.array([idx, idx + 1])


def test_get_data_at_time(monkeypatch):
    monkeypatch.setattr(er.glob, "glob", lambda pattern: ["f1.e", "f2.e"])
    monkeypatch.setattr(er.glob, "has_magic", lambda pattern: True)
    monkeypatch.setattr(er, "_SingleExodusReader", DummyExodusReader)
    mr = er.ExodusReader("dummy.e")
    assert mr.nodal_var_names == ["a", "b", "c_Cr"]
    assert mr.elem_var_names == ["e1", "e2"]
    x, y, z, c = mr.get_data_at_time("c_Cr", 1.0, block_id=1)
    assert x.shape == (2, 1)
    assert y.shape == (2, 1)
    assert z.shape == (2, 1)
    assert np.array_equal(c, np.array([1, 2]))
    with pytest.raises(ValueError):
        mr.get_data_at_time("missing", 1.0)


def test_missing_files_raise_error(monkeypatch):
    pattern = "missing*.e"
    monkeypatch.setattr(er.glob, "glob", lambda pat: [])
    monkeypatch.setattr(er.glob, "has_magic", lambda pat: True)
    with pytest.raises(FileNotFoundError, match=pattern):
        er.ExodusReader(pattern)


def test_get_data_at_time_single_file(monkeypatch):
    monkeypatch.setattr(er.glob, "glob", lambda pattern: ["f1.e"])
    monkeypatch.setattr(er.glob, "has_magic", lambda pattern: True)
    monkeypatch.setattr(er, "_SingleExodusReader", DummyExodusReader)
    mr = er.ExodusReader("dummy.e")
    x, y, z, c = mr.get_data_at_time("c_Cr", 1.0, block_id=1)
    assert x.shape == (2, 1)
    assert y.shape == (2, 1)
    assert z.shape == (2, 1)
    assert c.shape == (2,)
