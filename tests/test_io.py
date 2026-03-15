import os
import pickle
from datetime import datetime

import pytest

from qten.utils import io


@pytest.fixture
def io_root(tmp_path):
    io._io_dir = None
    io._all_env = None
    io._current_env = None
    root = io.iodir(tmp_path)
    yield root
    io._io_dir = None
    io._all_env = None
    io._current_env = None


@pytest.fixture
def io_env(io_root):
    name = "test_env"
    io.env(name)
    return name


def test_env_requires_set(io_root):
    with pytest.raises(RuntimeError):
        io.env()


def test_env_creates_directory(io_root):
    name = io.env("alpha")
    assert name == "alpha"
    assert io.env() == "alpha"
    assert os.path.isdir(os.path.join(io_root, "alpha"))


def test_save_and_load_versions(io_root, io_env):
    obj = {"a": 1}
    v1 = io.save(obj, "sample")
    v2 = io.save({"a": 2}, "sample")

    assert v1 == 1
    assert v2 == 2
    assert io.load("sample", 1) == obj
    assert io.load("sample", -1) == {"a": 2}


def test_load_missing_name_raises(io_root, io_env):
    with pytest.raises(FileNotFoundError):
        io.load("missing")


def test_load_missing_version_raises(io_root, io_env):
    io.save({"a": 1}, "sample")
    with pytest.raises(FileNotFoundError):
        io.load("sample", 2)


def test_list_saved(io_root, io_env):
    io.save({"a": 1}, "sample")
    io.save({"a": 2}, "sample")

    name_dir = os.path.join(io_root, io_env, "sample")
    with open(os.path.join(name_dir, "notes.txt"), "w", encoding="utf-8") as handle:
        handle.write("ignore")
    with open(
        os.path.join(name_dir, "version_bad.pkl"), "w", encoding="utf-8"
    ) as handle:
        handle.write("ignore")

    rows = io.list_saved("sample")
    assert [row["version"] for row in rows] == [1, 2]
    for row in rows:
        assert isinstance(row["created"], str)
        datetime.fromisoformat(row["created"])
        assert isinstance(row["size_mib"], float)


def test_list_saved_missing_name_raises(io_root, io_env):
    with pytest.raises(FileNotFoundError):
        io.list_saved("missing")


def test_multi_environment_isolation(io_root):
    io.env("env_a")
    io.save({"env": "a"}, "sample")

    io.env("env_b")
    io.save({"env": "b"}, "sample")

    io.env("env_a")
    assert io.load("sample", -1) == {"env": "a"}

    io.env("env_b")
    assert io.load("sample", -1) == {"env": "b"}


def test_save_unpicklable_raises_pickling_error(io_root, io_env):
    def _unpicklable():
        return 1

    with pytest.raises(pickle.PicklingError):
        io.save(_unpicklable, "sample")
