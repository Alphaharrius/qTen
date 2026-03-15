import os
import pickle
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from .loggings import get_logger


_logger = get_logger(__name__, show_datetime=True)


_io_dir: Optional[str] = None


def iodir(path: Optional[Union[str, os.PathLike[str]]] = None) -> str:
    """
    Get or set the base directory for IO storage.

    If a path is provided, it becomes the active IO directory. The directory
    is created if needed. If no path is provided, the current IO directory
    is returned; when unset, defaults to ".data".

    Parameters
    ----------
    `path`
        Optional filesystem path to use as the IO root.

    Returns
    -------
    `str`
        The resolved IO directory path.
    """
    global _io_dir
    if path is not None:
        _io_dir = os.path.abspath(os.fspath(path))
        _logger.debug("IO directory set to: %s", _io_dir)
    dir_path = _io_dir or ".data"
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


_all_env: Optional[Set[str]] = None
_current_env: Optional[str] = None


def env(name: Optional[str] = None) -> str:
    """
    Get or set the active environment name under the IO directory.

    When called without a name, returns the currently active environment if
    one was set during this process, otherwise raises a RuntimeError.
    When a name is provided, ensures a subdirectory exists under the IO
    directory and sets it as the active environment.

    Parameters
    ----------
    `name`
        Environment name to activate, or `None` to query the current one.

    Returns
    -------
    `str`
        The current or newly-set environment name.

    Raises
    ------
    `RuntimeError`
        If no environment is set and `name` is `None`.
    """
    global _all_env
    global _current_env
    if name is None:
        if _current_env is not None:
            return _current_env
        raise RuntimeError("No environment is currently set.")

    if _all_env is None:
        root = iodir()
        _all_env = {
            entry
            for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        }

    if name not in _all_env:
        os.makedirs(os.path.join(iodir(), name), exist_ok=True)
        _all_env.add(name)
        _logger.debug("Environment created: %s", name)

    _current_env = name
    _logger.debug("Environment set to: %s", _current_env)
    return _current_env


def _scan_versions(path: str) -> List[int]:
    versions: List[int] = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                name = entry.name
                if not (name.startswith("version_") and name.endswith(".pkl")):
                    continue
                ver_str = name[len("version_") : -len(".pkl")]
                try:
                    versions.append(int(ver_str))
                except ValueError:
                    continue
    except FileNotFoundError:
        return []
    return versions


def save(obj: Any, name: str) -> int:
    """
    Save an object to disk as a pickle with automatic versioning.

    Parameters
    ----------
    `obj`
        Object to serialize.
    `name`
        Logical name for grouping versions.

    Returns
    -------
    `int`
        The assigned version number.

    Raises
    ------
    `pickle.PicklingError`
        If the object cannot be pickled.
    """
    root = os.path.join(iodir(), env())
    name_dir = os.path.join(root, name)
    os.makedirs(name_dir, exist_ok=True)

    versions = _scan_versions(name_dir)
    version = max(versions) + 1 if versions else 1
    path = os.path.join(name_dir, f"version_{version}.pkl")
    try:
        with open(path, "wb") as file:
            pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise pickle.PicklingError("Object is not picklable.") from exc
    _logger.debug("Saved %s version %s to: %s", name, version, path)
    return version


def load(name: str, version: int = -1) -> Any:
    """
    Load a previously saved object by name and version.

    Parameters
    ----------
    `name`
        Logical name for grouping versions.
    `version`
        Version to load; use `-1` for the latest version.
    Returns
    -------
    `Any`
        The deserialized object.

    Raises
    ------
    `FileNotFoundError`
        If the name or version does not exist.
    `pickle.UnpicklingError`
        If the pickle data is corrupted.

    Notes
    -----
    Only load data you trust; pickle is not secure against malicious data.
    """
    root = os.path.join(iodir(), env())
    name_dir = os.path.join(root, name)
    versions = _scan_versions(name_dir)
    if not versions:
        raise FileNotFoundError(f"No saved versions for name: {name}")

    if version == -1:
        version = max(versions)
    elif version not in versions:
        raise FileNotFoundError(f"Version {version} not found for name: {name}")

    path = os.path.join(name_dir, f"version_{version}.pkl")
    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def list_saved(name: str) -> List[Dict[str, Any]]:
    """
    List saved versions for a name.

    Parameters
    ----------
    `name`
        Logical name for grouping versions.
    Returns
    -------
    `list[dict[str, Any]]`
        Rows with `version`, `created`, and `size_mib`.

    Raises
    ------
    `FileNotFoundError`
        If the name does not exist.
    """
    root = os.path.join(iodir(), env())
    name_dir = os.path.join(root, name)
    if not os.path.isdir(name_dir):
        raise FileNotFoundError(f"No saved versions for name: {name}")

    rows: List[Dict[str, Any]] = []
    with os.scandir(name_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            fname = entry.name
            if not (fname.startswith("version_") and fname.endswith(".pkl")):
                continue
            ver_str = fname[len("version_") : -len(".pkl")]
            try:
                version = int(ver_str)
            except ValueError:
                continue
            stat = entry.stat()
            created = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            size_mib = stat.st_size / (1024 * 1024)
            rows.append(
                {
                    "version": version,
                    "created": created,
                    "size_mib": size_mib,
                }
            )

    rows.sort(key=lambda row: row["version"])
    return rows
