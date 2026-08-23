"""
Runtime loader for packaged crystallographic point-group data.

This module reads the qten-owned JSON character/generator tables and exposes
[`named_pointgroup()`][qten.pointgroups._registry.named_pointgroup] for building
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] instances from
Hermann-Mauguin symbols or Schoenflies aliases.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import sympy as sy

from .finite import FinitePointGroup, _matrix_key


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
def _records_by_symbol() -> dict[str, dict[str, Any]]:
    data = _point_group_data()
    return {
        record["symbol"]: record
        for record in data["point_groups"]
        if record.get("dim", 3) == 3 and record.get("frame", "xyz") == "xyz"
    }


@lru_cache
def _alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for record in _point_group_data()["point_groups"]:
        if record.get("dim", 3) != 3 or record.get("frame", "xyz") != "xyz":
            continue
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


def _embedded_record(symbol: str, dim: int, frame: str) -> dict[str, Any] | None:
    for record in _point_group_data()["point_groups"]:
        if (
            record["symbol"] == symbol
            and record.get("dim") == dim
            and record.get("frame") == frame
        ):
            return record
    return None


def _as_matrix(value: Any) -> sy.ImmutableDenseMatrix:
    rows = [
        [sy.sympify(entry) if isinstance(entry, str) else entry for entry in row]
        for row in value
    ]
    return sy.ImmutableDenseMatrix(rows)


def _group_from_record(record: dict[str, Any], spin: str) -> FinitePointGroup:
    generators = record["generators"]
    dim = int(record.get("dim", 3))
    frame = str(record.get("frame", "xyz"))
    if isinstance(generators, dict):
        spatial = tuple(_as_matrix(matrix) for matrix in generators["spatial"])
        rotation3s = tuple(_as_matrix(matrix) for matrix in generators["rotation3"])
        if len(spatial) != len(rotation3s):
            raise ValueError(
                f"spatial and rotation3 generators are out of order for {record['symbol']}."
            )
        group = FinitePointGroup.from_matrices(
            matrices=spatial,
            axes=_axis_symbols(frame),
            symbol=record["symbol"],
            irreps=record.get("irreps"),
            rotation3s=rotation3s,
            spin=spin,
        )
    else:
        crystal = record.get("crystal_system", "")
        cartesian = tuple(
            _cartesianize_generator(_as_matrix(matrix), crystal)
            for matrix in generators
        )
        group = FinitePointGroup.from_matrices(
            matrices=cartesian,
            axes=_axis_symbols("xyz") if dim == 3 else _axis_symbols(frame),
            symbol=record["symbol"],
            irreps=record.get("irreps"),
            spin=spin,
        )
    packaged = record.get("spinor_irreps") if spin == "electron" else None
    return _attach_spinor_table(group, packaged=packaged)


def verify_spinor_factor_system(group: FinitePointGroup) -> None:
    r"""
    Check that the group's SU(2) section is consistent and its spinor table
    is a complete set of projective irreps of \(G\).
    """
    from ._characters import factor_system_and_lifts

    factor_system_and_lifts(group)
    table = group.spinor_table()
    irreps = table["irreps"]
    order = group.order
    if sum(int(row["dim"]) ** 2 for row in irreps.values()) != order:
        raise ValueError(
            f"Spinor irrep dimensions do not satisfy sum(dim^2)=|G| for {group.symbol}."
        )
    for irrep_name in irreps:
        characters = group.spinor_irrep_characters_by_element(irrep_name)
        if len(characters) != order:
            raise ValueError(
                f"Spinor characters for '{irrep_name}' do not match the group order."
            )


def _rotation_mapping_z_to(axis: tuple[float, ...]) -> sy.ImmutableDenseMatrix:
    """Return a proper rotation that maps the Cartesian z-axis to `axis`."""
    target = sy.ImmutableDenseMatrix(
        [sy.nsimplify(sy.sympify(component), tolerance=1e-12) for component in axis]
    )
    norm = sy.sqrt(sum(component**2 for component in target))
    if norm == 0:
        raise ValueError("axis must be a nonzero vector.")
    target = sy.ImmutableDenseMatrix(sy.simplify(target / norm))
    z_axis = sy.ImmutableDenseMatrix([0, 0, 1])
    cosine = sy.simplify(z_axis.dot(target))
    if cosine == 1:
        return sy.ImmutableDenseMatrix.eye(3)
    if cosine == -1:
        return sy.ImmutableDenseMatrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    helper = (
        sy.ImmutableDenseMatrix([1, 0, 0])
        if abs(complex(sy.N(target[0]))) < 0.9
        else sy.ImmutableDenseMatrix([0, 1, 0])
    )
    first = helper.cross(target)
    first = sy.ImmutableDenseMatrix(sy.simplify(first / first.norm()))
    second = sy.ImmutableDenseMatrix(sy.simplify(target.cross(first)))
    return sy.ImmutableDenseMatrix(sy.simplify(first.row_join(second).row_join(target)))


def _plane_onb(
    normal: tuple[float, ...],
) -> tuple[sy.ImmutableDenseMatrix, sy.ImmutableDenseMatrix]:
    nvec = sy.ImmutableDenseMatrix(
        [sy.nsimplify(sy.sympify(component), tolerance=1e-12) for component in normal]
    )
    nvec = sy.ImmutableDenseMatrix(
        sy.simplify(nvec / sy.sqrt(sum(nvec[i] ** 2 for i in range(3))))
    )
    helper = (
        sy.ImmutableDenseMatrix([1, 0, 0])
        if abs(complex(sy.N(nvec[0]))) < 0.9
        else sy.ImmutableDenseMatrix([0, 1, 0])
    )
    first = nvec.cross(helper)
    first = sy.ImmutableDenseMatrix(sy.simplify(first / first.norm()))
    second = nvec.cross(first)
    second = sy.ImmutableDenseMatrix(sy.simplify(second / second.norm()))
    return first, second


def _project_matrix_to_plane(
    matrix: sy.ImmutableDenseMatrix,
    first: sy.ImmutableDenseMatrix,
    second: sy.ImmutableDenseMatrix,
) -> sy.ImmutableDenseMatrix:
    basis = first.row_join(second)
    return sy.ImmutableDenseMatrix(sy.simplify(basis.T @ matrix @ basis))


def _attach_spinor_table(
    group: FinitePointGroup,
    packaged: dict[str, Any] | None = None,
) -> FinitePointGroup:
    if group.spin != "electron":
        return group
    table = packaged if packaged is not None else group.spinor_irreps
    if table is None:
        from ._characters import compute_spinor_irreps

        table = compute_spinor_irreps(group)
    if table is group.spinor_irreps:
        return group
    return FinitePointGroup(
        generators=group.generators,
        axes=group.axes,
        symbol=group.symbol,
        irreps=group.irreps,
        spinor_irreps=table,
        spin=group.spin,
        class_indices=group.class_indices,
    )


def _require_faithful_restriction(
    group: FinitePointGroup,
    project: Any,
    *,
    symbol: str,
    plane_desc: str,
) -> dict[tuple[float, ...], int]:
    """Reject a plane cut that identifies distinct 3D operations.

    Returns a map from projected-matrix key to Bilbao class index when the
    3D group carries packaged ordinary irreps.
    """
    class_by_element = group.element_class_indices() if group.irreps else None
    seen_3d: dict[tuple[float, ...], tuple[float, ...]] = {}
    class_by_2d: dict[tuple[float, ...], int] = {}
    for index, element in enumerate(group.elements()):
        key2 = _matrix_key(project(element.irrep))
        key3 = _matrix_key(element.irrep)
        previous = seen_3d.get(key2)
        if previous is not None and previous != key3:
            raise ValueError(
                f"Point group {symbol} does not act faithfully on {plane_desc}: "
                "distinct 3D operations become the same spatial matrix. "
                "Use the 3D group (axis=) or a faithful subgroup "
                "(for example 4mm instead of 4/m)."
            )
        seen_3d[key2] = key3
        if class_by_element is not None:
            class_by_2d[key2] = class_by_element[index]
    return class_by_2d


def _group_with_projected_classes(
    group: FinitePointGroup,
    class_by_2d: dict[tuple[float, ...], int],
) -> FinitePointGroup:
    if not class_by_2d:
        return group
    try:
        indices = tuple(
            class_by_2d[_matrix_key(element.irrep)] for element in group.elements()
        )
    except KeyError as exc:
        raise ValueError(
            f"Could not transport class labels onto the {group.symbol} plane cut."
        ) from exc
    return FinitePointGroup(
        generators=group.generators,
        axes=group.axes,
        symbol=group.symbol,
        irreps=group.irreps,
        spinor_irreps=group.spinor_irreps,
        spin=group.spin,
        class_indices=indices,
    )


def _hashable_axis(value: object) -> str | tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, tuple):
        raise TypeError(f"plane/axis must be a string or tuple, got {type(value)!r}.")
    return tuple(float(sy.N(component)) for component in value)


@lru_cache
def named_pointgroup(
    query: str,
    plane: str | tuple[float, ...] | None = None,
    axis: tuple[float, ...] | None = None,
    spin: str = "electron",
) -> FinitePointGroup:
    """
    Build a finite point group from packaged generator data.

    The query accepts canonical Hermann-Mauguin symbols such as ``"4mm"`` and
    Schoenflies aliases such as ``"C4v"``. A trailing axis suffix such as
    ``"C4v-xy"`` is an alias for ``plane="xy"``: spatial matrices become 2D
    while the 3D rotations used for spin are kept. A named group that is not
    faithful on the requested plane raises ``ValueError``.
    """

    symbol, axis_names = _split_named_query(query)
    if isinstance(plane, str):
        if axis_names != "xyz" and axis_names != plane:
            raise ValueError(
                f"Query '{query}' already selects axes {axis_names!r}, "
                f"which conflicts with plane={plane!r}."
            )
        axis_names = plane
    record = _records_by_symbol()[symbol]
    packaged = record.get("spinor_irreps") if spin == "electron" else None
    cartesian = tuple(
        _cartesianize_generator(
            sy.ImmutableDenseMatrix(generator), record["crystal_system"]
        )
        for generator in record["generators"]
    )
    xyz_axes = _axis_symbols("xyz")

    if axis is not None:
        if axis_names != "xyz" or plane is not None:
            raise ValueError("axis= applies to a 3D named group without a plane cut.")
        group = FinitePointGroup.from_matrices(
            matrices=cartesian,
            axes=xyz_axes,
            symbol=symbol,
            irreps=record.get("irreps"),
            spin=spin,
        )
        return group.reoriented_by(_rotation_mapping_z_to(axis))

    if isinstance(plane, tuple):
        group = FinitePointGroup.from_matrices(
            matrices=cartesian,
            axes=xyz_axes,
            symbol=symbol,
            irreps=record.get("irreps"),
            spin=spin,
        )
        group = group.reoriented_by(_rotation_mapping_z_to(plane))
        first, second = _plane_onb(plane)
        spatial = []
        rotations = []
        normal = sy.ImmutableDenseMatrix(list(plane))
        normal = sy.ImmutableDenseMatrix(
            sy.simplify(normal / sy.sqrt(sum(normal[i] ** 2 for i in range(3))))
        )
        for generator in group.generators:
            image = sy.ImmutableDenseMatrix(sy.simplify(generator.irrep @ normal))
            parallel = sy.simplify(image.cross(normal).norm()) == 0
            if not parallel:
                raise ValueError(
                    f"Generator of {symbol} does not preserve the plane with "
                    f"normal {plane}."
                )
            spatial.append(_project_matrix_to_plane(generator.irrep, first, second))
            rotations.append(generator.irrep)
        class_by_2d = _require_faithful_restriction(
            group,
            lambda matrix: _project_matrix_to_plane(matrix, first, second),
            symbol=symbol,
            plane_desc=f"the plane with normal {plane}",
        )
        cut = FinitePointGroup.from_matrices(
            matrices=spatial,
            axes=_axis_symbols("xy"),
            symbol=symbol,
            irreps=record.get("irreps"),
            rotation3s=rotations,
            spin=spin,
        )
        return _group_with_projected_classes(cut, class_by_2d)

    if axis_names == "xyz":
        group = FinitePointGroup.from_matrices(
            matrices=cartesian,
            axes=xyz_axes,
            symbol=symbol,
            irreps=record.get("irreps"),
            spin=spin,
        )
        return _attach_spinor_table(group, packaged=packaged)

    embedded = _embedded_record(symbol, len(axis_names), axis_names)
    if embedded is not None:
        return _group_from_record(embedded, spin)

    group3 = FinitePointGroup.from_matrices(
        matrices=cartesian,
        axes=xyz_axes,
        symbol=symbol,
        irreps=record.get("irreps"),
        spin=spin,
    )
    class_by_2d = _require_faithful_restriction(
        group3,
        lambda matrix: _project_generator(matrix, axis_names),
        symbol=symbol,
        plane_desc=f"the {axis_names} plane",
    )
    projected = tuple(_project_generator(matrix, axis_names) for matrix in cartesian)
    group = FinitePointGroup.from_matrices(
        matrices=projected,
        axes=_axis_symbols(axis_names),
        symbol=symbol,
        irreps=record.get("irreps"),
        rotation3s=cartesian,
        spin=spin,
    )
    group = _group_with_projected_classes(group, class_by_2d)
    return _attach_spinor_table(group, packaged=packaged)
