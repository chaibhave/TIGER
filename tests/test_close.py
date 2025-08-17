import importlib
import pathlib
import sys
import types
import pytest


@pytest.fixture
def er_module():
    """Import ExodusReader with minimal stubs for optional dependencies."""
    # Ensure a fresh import each time to avoid cross-test contamination
    sys.modules.pop("ExodusReader", None)
    if "numpy" not in sys.modules:
        sys.modules["numpy"] = types.ModuleType("numpy")
    if "netCDF4" not in sys.modules:
        nc = types.ModuleType("netCDF4")
        nc.Dataset = object  # type: ignore[attr-defined]
        sys.modules["netCDF4"] = nc
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
    return importlib.import_module("ExodusReader")


class DummyDataset:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_single_reader_close_releases_resources(er_module):
    ds = DummyDataset()
    sr = er_module._SingleExodusReader.__new__(er_module._SingleExodusReader)
    sr.mesh = ds
    sr.close()
    assert ds.closed
    assert sr.mesh is None


def test_single_reader_context_manager_closes(er_module):
    ds = DummyDataset()
    sr = er_module._SingleExodusReader.__new__(er_module._SingleExodusReader)
    sr.mesh = ds
    with sr:
        pass
    assert ds.closed


class DummyChild:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_multi_reader_close_releases_children(er_module):
    r1 = DummyChild()
    r2 = DummyChild()
    mr = er_module._MultiExodusReader.__new__(er_module._MultiExodusReader)
    mr.exodus_readers = [r1, r2]
    mr.close()
    assert r1.closed and r2.closed


def test_multi_reader_context_manager_closes_children(er_module):
    r1 = DummyChild()
    r2 = DummyChild()
    mr = er_module._MultiExodusReader.__new__(er_module._MultiExodusReader)
    mr.exodus_readers = [r1, r2]
    with mr:
        pass
    assert r1.closed and r2.closed
