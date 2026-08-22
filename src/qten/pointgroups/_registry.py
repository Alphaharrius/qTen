"""
Runtime loader for packaged crystallographic point-group data.

This module reads the qten-owned JSON character/generator tables and exposes
[`named_pointgroup()`][qten.pointgroups._registry.named_pointgroup] for building
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] instances from
Hermann-Mauguin symbols or Schoenflies aliases.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from importlib import resources
from typing import Any, Sequence

import numpy as np
import sympy as sy

from ..phys.spin import SU2_SECTION_CONVENTION, principal_su2_from_rows
from .finite import FinitePointGroup


_STANDARD_AXIS_NAMES = ("x", "y", "z")
_HEXAGONAL_CARTESIAN_BASIS = sy.ImmutableDenseMatrix(
    [
        [1, -sy.Rational(1, 2), 0],
        [0, sy.sqrt(3) / 2, 0],
        [0, 0, 1],
    ]
)


@lru_cache
def _point_group_data() -> dict[str, Any]:
    data_path = resources.files("qten.pointgroups.data").joinpath(
        "point_group_data.json"
    )
    return json.loads(data_path.read_text(encoding="utf-8"))


@lru_cache
def _double_point_group_data() -> dict[str, Any]:
    data_path = resources.files("qten.pointgroups.data").joinpath(
        "double_point_group_data.json"
    )
    data = json.loads(data_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported double-point-group data schema.")
    if data.get("section_convention") != SU2_SECTION_CONVENTION:
        raise ValueError("Double-point-group data uses an incompatible SU(2) section.")
    return data


@lru_cache
def _records_by_symbol() -> dict[str, dict[str, Any]]:
    data = _point_group_data()
    return {record["symbol"]: record for record in data["point_groups"]}


@lru_cache
def _double_records_by_symbol() -> dict[str, dict[str, Any]]:
    data = _double_point_group_data()
    return {record["symbol"]: record for record in data["point_groups"]}


@lru_cache
def _alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for record in _point_group_data()["point_groups"]:
        for alias in record["aliases"]:
            aliases[alias] = record["symbol"]
    return aliases


def known_point_group_symbols() -> tuple[str, ...]:
    """Return the canonical Hermann-Mauguin symbols available in the registry."""

    return tuple(sorted(_records_by_symbol()))


def _axis_symbols(axis_names: str) -> tuple[sy.Symbol, ...]:
    return tuple(sy.Symbol(name) for name in axis_names)


def _canonical_symbol(symbol: str) -> str:
    cleaned = symbol.replace(" ", "")
    records = _records_by_symbol()
    aliases = _alias_map()
    if cleaned in records:
        return cleaned
    if cleaned in aliases:
        return aliases[cleaned]

    lower_aliases = {key.lower(): value for key, value in aliases.items()}
    lowered = cleaned.lower()
    if lowered in lower_aliases:
        return lower_aliases[lowered]

    raise ValueError(f"Unknown crystallographic point-group symbol '{symbol}'.")


def _split_named_query(query: str) -> tuple[str, str]:
    cleaned = query.strip().replace(" ", "")
    if not cleaned:
        raise ValueError("Point-group query must be non-empty.")

    symbol = cleaned
    axes = "xyz"
    if "-" in cleaned:
        maybe_symbol, maybe_axes = cleaned.rsplit("-", 1)
        if maybe_symbol and set(maybe_axes).issubset(set(_STANDARD_AXIS_NAMES)):
            if len(set(maybe_axes)) != len(maybe_axes):
                raise ValueError(
                    f"Named point-group axes must be unique, got '{maybe_axes}'."
                )
            symbol = maybe_symbol
            axes = maybe_axes

    return _canonical_symbol(symbol), axes


def _project_generator(
    matrix: sy.ImmutableDenseMatrix, axis_names: str
) -> sy.ImmutableDenseMatrix:
    if any(name not in _STANDARD_AXIS_NAMES for name in axis_names):
        raise ValueError(
            "Packaged point-group generators can only be projected onto x/y/z axes."
        )

    selected = tuple(_STANDARD_AXIS_NAMES.index(name) for name in axis_names)
    omitted = tuple(i for i in range(3) if i not in selected)
    for row in selected:
        for col in omitted:
            if sy.simplify(matrix[row, col]) != 0:
                raise ValueError(
                    f"Generator does not preserve the selected axes '{axis_names}'."
                )
    for row in omitted:
        for col in selected:
            if sy.simplify(matrix[row, col]) != 0:
                raise ValueError(
                    f"Generator does not preserve the selected axes '{axis_names}'."
                )

    return sy.ImmutableDenseMatrix(
        [[matrix[row, col] for col in selected] for row in selected]
    )


def _cartesianize_generator(
    matrix: sy.ImmutableDenseMatrix, crystal_system: str
) -> sy.ImmutableDenseMatrix:
    """Convert packaged crystallographic coordinates to Cartesian x/y/z."""
    if crystal_system not in {"trigonal", "hexagonal"}:
        return matrix
    basis = _HEXAGONAL_CARTESIAN_BASIS
    return sy.ImmutableDenseMatrix(sy.simplify(basis @ matrix @ basis.inv()))


def _numpy_o3(matrix: sy.Matrix) -> np.ndarray:
    """Real 3x3 matrix as float64, for numeric products and lifts."""
    return np.array(
        [[float(sy.N(matrix[row, col])) for col in range(3)] for row in range(3)],
        dtype=np.float64,
    )


def _o3_key(rotation: np.ndarray, *, digits: int = 6) -> tuple[float, ...]:
    return tuple(round(float(entry), digits) for entry in rotation.reshape(-1))


def _su2_numpy_from_cartesian(matrix: sy.ImmutableDenseMatrix) -> np.ndarray:
    """Numeric principal-branch SU(2) lift of a Cartesian O(3) matrix."""
    return np.array(
        principal_su2_from_rows(_numpy_o3(matrix).tolist()), dtype=np.complex128
    )


def _check_numeric_factor_system(
    rotations: Sequence[np.ndarray],
    lifts: np.ndarray,
    factor_system: Sequence[Sequence[int]],
    *,
    symbol: str,
    integer: bool,
) -> None:
    r"""Check \(U(g)U(h)=\omega(g,h)U(gh)\) with numeric 2x2 arithmetic."""
    order = len(rotations)
    if lifts.shape != (order, 2, 2):
        raise ValueError(f"Spinor lift stack has the wrong shape for {symbol}.")
    if len(factor_system) != order or any(len(row) != order for row in factor_system):
        raise ValueError(f"Spinor factor system has the wrong shape for {symbol}.")

    if integer:
        index = {
            tuple(int(entry) for entry in rotation.reshape(-1)): i
            for i, rotation in enumerate(rotations)
        }
    else:
        index = {_o3_key(rotation): i for i, rotation in enumerate(rotations)}
    if len(index) != order:
        raise ValueError(f"Spinor table contains duplicate operations for {symbol}.")

    omega = np.asarray(factor_system, dtype=np.int8)
    if set(int(value) for value in omega.reshape(-1)) - {-1, 1}:
        raise ValueError(f"Spinor factor is not ±1 for {symbol}.")

    for i, left in enumerate(rotations):
        for j, right in enumerate(rotations):
            product = left @ right
            key = (
                tuple(int(entry) for entry in product.reshape(-1))
                if integer
                else _o3_key(product)
            )
            if key not in index:
                raise ValueError(
                    f"Spinor factor system product is missing for {symbol}."
                )
            expected = int(omega[i, j]) * lifts[index[key]]
            if not np.allclose(lifts[i] @ lifts[j], expected, rtol=0.0, atol=1e-8):
                raise ValueError(
                    "Spinor factor system does not match the current SU(2) lift "
                    f"for {symbol}."
                )


def verify_spinor_factor_system(group: FinitePointGroup) -> None:
    r"""
    Check that a group's spinor table matches the current SU(2) lift.

    Packaged tables are verified from crystallographic integer products, so the
    pair loop never builds symbolic \(gh\). Custom tables use numeric Cartesian
    keys. In both cases \(U(g)\) is the numeric principal-branch lift.
    """
    table = group.spinor_irreps
    if table is None:
        raise ValueError(
            f"No spinor character-table data is available for {group.symbol}."
        )

    operations = tuple(
        sy.ImmutableDenseMatrix(operation) for operation in table["operations"]
    )
    symbol = group.symbol or "<anonymous>"
    cartesian = [_numpy_o3(matrix) for matrix in operations]
    operation_keys = {_o3_key(rotation) for rotation in cartesian}
    if len(operation_keys) != len(operations):
        raise ValueError(f"Spinor table contains duplicate operations for {symbol}.")
    element_keys = {_o3_key(_numpy_o3(element.irrep)) for element in group.elements()}
    if element_keys != operation_keys:
        raise ValueError(
            "Spinor table operations do not match generated point-group "
            f"elements for {symbol}."
        )

    lifts = np.stack([_su2_numpy_from_cartesian(matrix) for matrix in operations])
    source = (
        _double_records_by_symbol().get(group.symbol)
        if group.symbol is not None
        else None
    )
    if (
        source is not None
        and len(source.get("operations", ())) == len(operations)
        and source.get("factor_system") == table.get("factor_system")
    ):
        cryst = [np.asarray(operation, dtype=int) for operation in source["operations"]]
        _check_numeric_factor_system(
            cryst,
            lifts,
            source["factor_system"],
            symbol=symbol,
            integer=True,
        )
        return

    _check_numeric_factor_system(
        cartesian,
        lifts,
        table["factor_system"],
        symbol=symbol,
        integer=False,
    )


@lru_cache
def _verified_spinor_table(symbol: str) -> dict[str, Any] | None:
    """Cartesian spinor table for `symbol`, checked once against the current lift."""
    record = _records_by_symbol()[symbol]
    table = _spinor_table(record, "xyz")
    if table is None:
        return None
    source = _double_records_by_symbol()[symbol]
    cryst = [np.asarray(operation, dtype=int) for operation in source["operations"]]
    lifts = np.stack(
        [_su2_numpy_from_cartesian(operation) for operation in table["operations"]]
    )
    _check_numeric_factor_system(
        cryst,
        lifts,
        source["factor_system"],
        symbol=symbol,
        integer=True,
    )
    return table


def _spinor_table(record: dict[str, Any], axis_names: str) -> dict[str, Any] | None:
    """Return a verified spinor table in QTen's Cartesian matrix convention."""
    if axis_names != "xyz":
        return None
    table = _double_records_by_symbol().get(record["symbol"])
    if table is None:
        return None

    generator_bytes = json.dumps(
        record["generators"], separators=(",", ":"), sort_keys=True
    ).encode()
    fingerprint = hashlib.sha256(generator_bytes).hexdigest()
    if table.get("generator_sha256") != fingerprint:
        raise ValueError(
            f"Spinor table generator fingerprint is stale for {record['symbol']}."
        )

    normalized = dict(table)
    normalized["operations"] = tuple(
        _cartesianize_generator(
            sy.ImmutableDenseMatrix(operation), record["crystal_system"]
        )
        for operation in table["operations"]
    )
    return normalized


@lru_cache
def named_pointgroup(query: str) -> FinitePointGroup:
    """
    Build a finite point group from packaged generator data.

    The query accepts canonical Hermann-Mauguin symbols such as ``"4mm"`` and
    Schoenflies aliases such as ``"C4v"``. A trailing axis suffix may be used to
    project standard 3D matrices onto invariant coordinate subspaces, e.g.
    ``"C4v-xy"``.

    Parameters
    ----------
    query : str
        Named crystallographic point-group query, optionally with an axis
        suffix such as ``"C4v-xy"``.

    Returns
    -------
    FinitePointGroup
        Finite group generated by the packaged matrices projected onto the
        selected axes, including packaged character-table data when available.

    Raises
    ------
    ValueError
        If the symbol is unknown, the axis suffix is invalid, or a generator
        does not preserve the selected axes.
    """

    symbol, axis_names = _split_named_query(query)
    record = _records_by_symbol()[symbol]

    matrices = tuple(
        _project_generator(
            _cartesianize_generator(
                sy.ImmutableDenseMatrix(generator), record["crystal_system"]
            ),
            axis_names,
        )
        for generator in record["generators"]
    )
    spinor_irreps = _verified_spinor_table(symbol) if axis_names == "xyz" else None
    return FinitePointGroup.from_matrices(
        matrices=matrices,
        axes=_axis_symbols(axis_names),
        symbol=symbol,
        irreps=record.get("irreps"),
        spinor_irreps=spinor_irreps,
    )
