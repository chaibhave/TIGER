"""Convenience readers for Exodus and Nemesis files.

This module exposes :class:`ExodusReader`, which dispatches to single-file or
multi-file readers as appropriate.  The multi-file reader aggregates nodal and
elemental variable names from all children and exposes them via the
``nodal_var_names`` and ``elem_var_names`` attributes.  These lists can be used
to discover variables that may be requested with :meth:`get_data_at_time`.
"""

from __future__ import annotations

import glob
import os
import re
from typing import List, Tuple

import numpy as np
from netCDF4 import Dataset

try:  # pragma: no cover - mpi4py is optional
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - mpi4py is optional
    MPI = None  # type: ignore


class BaseExodusReader:
    """Shared logic for loading a single Exodus file."""

    def __init__(self, file_name: str) -> None:
        if not os.path.exists(file_name):
            raise FileNotFoundError(
                "File path does not exist. Please check if file path and name are correct."
            )

        self.file_name = file_name
        if MPI is not None:
            try:
                self.mesh = Dataset(
                    self.file_name,
                    "r",
                    parallel=True,
                    comm=MPI.COMM_WORLD,
                    info=MPI.Info(),
                )
            except ValueError:
                # Fall back to serial mode if netCDF4 lacks parallel support
                self.mesh = Dataset(self.file_name, "r")
        else:
            self.mesh = Dataset(self.file_name, "r")

        self.get_times()
        self.get_xyz()
        try:
            self.get_nodal_names()
        except Exception:
            self.nodal_var_names = []  # pragma: no cover - optional data
        try:
            self.get_elem_names()
        except Exception:
            self.elem_var_names = []  # pragma: no cover - optional data

    def get_times(self) -> np.ndarray:
        self.times = self.mesh.variables["time_whole"][:]
        return self.times

    def get_xyz(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.dim = 0
        try:
            x = self.mesh.variables["coordx"][:]
            self.dim += 1
        except Exception:
            raise ValueError("X dimension empty. Mesh must have at least one non-empty dimension")
        try:
            y = self.mesh.variables["coordy"][:]
            self.dim += 1
        except Exception:
            y = np.zeros(x.shape)
        try:
            z = self.mesh.variables["coordz"][:]
            self.dim += 1
        except Exception:
            z = np.zeros(x.shape)

        connect_re = re.compile("connect[0-9]+$")
        select_keys = [key for key in self.mesh.variables.keys() if connect_re.match(key)]
        connect = np.vstack([self.mesh.variables[key][:] for key in select_keys])
        X = x[connect[:] - 1]
        Y = y[connect[:] - 1]
        Z = z[connect[:] - 1]

        self.x = np.asarray(X)
        self.y = np.asarray(Y)
        self.z = np.asarray(Z)
        self.connect = connect
        return self.x, self.y, self.z

    def get_nodal_names(self) -> List[str]:
        names = self.mesh.variables["name_nod_var"]
        names.set_auto_mask(False)
        self.nodal_var_names = [b"".join(c).decode("latin1") for c in names[:]]
        return self.nodal_var_names

    def get_elem_names(self) -> List[str]:
        names = self.mesh.variables["name_elem_var"]
        names.set_auto_mask(False)
        elem_var_names: List[str] = []
        for n in names[:]:
            temp = [i.decode("latin1") for i in n]
            idx = temp.index("")
            elem_var_names += ["".join(temp[:idx])]
        self.elem_var_names = elem_var_names
        return self.elem_var_names

    def get_var_values(self, var_name: str, timestep: int) -> np.ndarray:
        if var_name in getattr(self, "nodal_var_names", []):
            idx = self.nodal_var_names.index(var_name)
            var_name_exodus = "vals_nod_var" + str(idx + 1)
            var_vals_nodal = self.mesh.variables[var_name_exodus]
            if timestep == -1:
                var_vals = np.average(
                    [var_vals_nodal[:, (self.connect[:, i] - 1)] for i in range(self.connect.shape[1])],
                    0,
                )
            else:
                var_vals = np.average(
                    [var_vals_nodal[timestep, (self.connect[:, i] - 1)] for i in range(self.connect.shape[1])],
                    0,
                )
        elif var_name in getattr(self, "elem_var_names", []):
            idx = self.elem_var_names.index(var_name)
            var_name_exodus = "vals_elem_var" + str(idx + 1) + "eb1"
            var_vals = np.asarray(self.mesh.variables[var_name_exodus][timestep])
        else:
            raise ValueError("Value not in nodal or elemental variables. Check variable name.")
        return var_vals


class _SingleExodusReader(BaseExodusReader):
    """Internal helper for reading a single file."""

    pass


class _MultiExodusReader:
    """Internal helper for reading multiple Exodus files."""

    def __init__(self, file_names: str) -> None:
        self.file_names = glob.glob(file_names)
        if not self.file_names:
            raise FileNotFoundError(file_names)
        global_times = set()
        file_times = []
        exodus_readers = []
        nodal_names: List[str] = []
        elem_names: List[str] = []
        for file_name in self.file_names:
            er = _SingleExodusReader(file_name)
            times = er.times
            global_times.update(times[:])
            exodus_readers.append(er)
            file_times.append([min(times), max(times)])
            nodal_names.extend(getattr(er, "nodal_var_names", []))
            elem_names.extend(getattr(er, "elem_var_names", []))
        self.dim = exodus_readers[0].dim
        global_times = list(global_times)
        global_times.sort()
        self.global_times = global_times
        self.exodus_readers = exodus_readers
        self.file_times = np.asarray(file_times)
        self.nodal_var_names = sorted(set(nodal_names))
        self.elem_var_names = sorted(set(elem_names))

    def _validate_var_name(self, var_name: str) -> None:
        if var_name not in self.nodal_var_names and var_name not in self.elem_var_names:
            raise ValueError(
                "Value not in nodal or elemental variables. Check variable name."
            )

    def get_data_from_file_idx(
        self, var_name: str, read_time: float, i: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._validate_var_name(var_name)
        er = self.exodus_readers[i]
        x = er.x
        y = er.y
        z = er.z
        idx_arr = np.where(np.isclose(er.times, read_time))[0]
        if idx_arr.size == 0:
            raise ValueError(f"Time {read_time} not found in file {er.file_name}")
        idx = idx_arr[0]
        c = er.get_var_values(var_name, idx)
        return x, y, z, c

    def get_data_at_time(
        self, var_name: str, read_time: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._validate_var_name(var_name)
        X = []
        Y = []
        Z = []
        C = []
        for i, file_time in enumerate(self.file_times):
            if file_time[0] <= read_time and file_time[1] >= read_time:
                x, y, z, c = self.get_data_from_file_idx(var_name, read_time, i)
                try:
                    X.append(x)
                    Y.append(y)
                    Z.append(z)
                    C.append(c)
                except Exception:  # pragma: no cover - fallback path
                    X = x
                    Y = y
                    Z = z
                    C = c
        X = np.vstack(X)
        Y = np.vstack(Y)
        Z = np.vstack(Z)
        C = np.hstack(C)
        return X, Y, Z, C


class ExodusReader:
    """Dispatcher that selects between single-file and multi-file readers."""

    def __new__(cls, path: str):  # type: ignore[override]
        files = glob.glob(path)
        if glob.has_magic(path) or len(files) > 1:
            return _MultiExodusReader(path)
        return _SingleExodusReader(path)
