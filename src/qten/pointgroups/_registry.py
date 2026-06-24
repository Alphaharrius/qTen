"""Runtime loader for packaged crystallographic point-group data."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import sympy as sy

from .finite import FinitePointGroup


_STANDARD_AXIS_NAMES = ("x", "y", "z")


@lru_cache
def _point_group_data() -> dict[str, Any]:
    data_path = resources.files("qten.pointgroups.data").joinpath(
        "point_group_data.json"
    )
    return json.loads(data_path.read_text(encoding="utf-8"))


@lru_cache
def _records_by_symbol() -> dict[str, dict[str, Any]]:
    data = _point_group_data()
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


def named_pointgroup(query: str) -> FinitePointGroup:
    """Build a finite point group from packaged generator data.

    The query accepts canonical Hermann-Mauguin symbols such as ``"4mm"`` and
    Schoenflies aliases such as ``"C4v"``. A trailing axis suffix may be used to
    project standard 3D matrices onto invariant coordinate subspaces, e.g.
    ``"C4v-xy"``.
    """

    symbol, axis_names = _split_named_query(query)
    record = _records_by_symbol()[symbol]

    matrices = tuple(
        _project_generator(
            sy.ImmutableDenseMatrix(generator),
            axis_names,
        )
        for generator in record["generators"]
    )
    return FinitePointGroup.from_matrices(
        matrices=matrices,
        axes=_axis_symbols(axis_names),
        symbol=symbol,
        irreps=record.get("irreps"),
    )
