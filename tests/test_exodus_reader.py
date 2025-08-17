import pytest
np = pytest.importorskip("numpy", exc_type=ImportError)
import ExodusReader as er


class DummyExodusReader:
    def __init__(self, file_name):
        self.times = np.array([0.0, 1.0])
        self.x = np.array([[0.0], [1.0]])
        self.y = np.array([[0.0], [0.0]])
        self.z = np.array([[0.0], [0.0]])
        self.dim = 1
        if "1" in file_name:
            self.nodal_var_names = ["c_Cr", "a"]
            self.elem_var_names = ["e1"]
        else:
            self.nodal_var_names = ["b", "c_Cr"]
            self.elem_var_names = ["e2"]

    def get_var_values(self, var_name, idx):
        return np.array([idx, idx + 1])


def test_get_data_at_time(monkeypatch):
    monkeypatch.setattr(er.glob, "glob", lambda pattern: ["f1.e", "f2.e"])
    monkeypatch.setattr(er.glob, "has_magic", lambda pattern: True)
    monkeypatch.setattr(er, "_SingleExodusReader", DummyExodusReader)
    mr = er.ExodusReader("dummy.e")
    assert mr.nodal_var_names == ["a", "b", "c_Cr"]
    assert mr.elem_var_names == ["e1", "e2"]
    x, y, z, c = mr.get_data_at_time("c_Cr", 1.0)
    assert x.shape == (2, 1)
    assert y.shape == (2, 1)
    assert z.shape == (2, 1)
    assert np.array_equal(c, np.array([1, 2]))
    with pytest.raises(ValueError):
        mr.get_data_at_time("missing", 1.0)
