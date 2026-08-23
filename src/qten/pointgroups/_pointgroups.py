"""
Compact point-group constructors.

This module provides the user-facing
[`pointgroup()`][qten.pointgroups._pointgroups.pointgroup] factory. Compact
affine queries such as `c4-xy:xy` return a single
[`PointGroupElement`][qten.pointgroups.elements.PointGroupElement]. Named
crystallographic queries such as `C4v` or `4mm` return a
[`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] built from
packaged generator data.

Repository usage
----------------
Use [`pointgroup()`][qten.pointgroups._pointgroups.pointgroup] for interactive
construction. Use [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement]
or [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup] directly when
a custom symbolic representation is needed.
"""

import re
from typing import Literal

import sympy as sy

from .elements import PointGroupElement
from .finite import FinitePointGroup
from ._registry import _hashable_axis, named_pointgroup
from ..phys.spin import _embed_in_cartesian_xyz


_AFFINE_QUERY_RE = re.compile(
    r"^(?P<group>c\d+|m)-(?P<ambient>[xyz]+):(?P<target>[xyz]+)$"
)


def _parse_affine_query(query: str):
    match = _AFFINE_QUERY_RE.fullmatch(query.strip())
    if match is None:
        raise ValueError(
            "Invalid query format. Expected '<group>-<ambient>:<target>', "
            "for example 'c3-xy:xy'."
        )

    group = match.group("group")
    ambient = match.group("ambient")
    target = match.group("target")

    if len(set(ambient)) != len(ambient):
        raise ValueError(f"Ambient axes must be unique, got '{ambient}'.")
    if len(set(target)) != len(target):
        raise ValueError(f"Target axes must be unique, got '{target}'.")
    if not set(target).issubset(set(ambient)):
        raise ValueError(
            f"Target axes '{target}' must be a subset of ambient axes '{ambient}'."
        )

    return group, ambient, target


def _is_affine_query(query: str) -> bool:
    return _AFFINE_QUERY_RE.fullmatch(query.strip()) is not None


def _build_cyclic_irrep(n: int, ambient: str, target: str) -> sy.ImmutableDenseMatrix:
    if n < 2:
        raise ValueError(f"Cyclic group order must be at least 2, got c{n}.")
    if len(ambient) < 2:
        raise ValueError(
            "Cyclic rotation requires at least 2D ambient space. "
            f"Got ambient axes '{ambient}'."
        )
    if len(target) != 2:
        raise ValueError(
            "Cyclic rotation must act on a 2D plane, so target axes must have length 2."
        )
    if len(ambient) == 2 and set(target) != set(ambient):
        raise ValueError(
            "For 2D ambient space, cyclic target plane must match the ambient plane. "
            f"Got ambient '{ambient}' and target '{target}'."
        )

    dim = len(ambient)
    irrep = sy.eye(dim)
    if not isinstance(irrep, sy.ImmutableDenseMatrix):
        irrep = sy.ImmutableDenseMatrix(irrep)

    i = ambient.index(target[0])
    j = ambient.index(target[1])
    sign = 1 if i < j else -1
    p, q = sorted((i, j))
    theta = sign * 2 * sy.pi / n
    cos_t = sy.cos(theta)
    sin_t = sy.sin(theta)

    mutable = sy.Matrix(irrep)
    # Place the 2D rotation block on the plane, while target order chooses orientation.
    mutable[p, p] = cos_t
    mutable[p, q] = -sin_t
    mutable[q, p] = sin_t
    mutable[q, q] = cos_t
    return sy.ImmutableDenseMatrix(mutable)


def _build_mirror_irrep(ambient: str, target: str) -> sy.ImmutableDenseMatrix:
    dim = len(ambient)
    if dim not in (1, 2, 3):
        raise ValueError(
            "Mirror currently supports only 1D/2D/3D ambient space, "
            f"got {dim}D with axes '{ambient}'."
        )

    if dim == 1:
        if len(target) != 1 or target != ambient:
            raise ValueError(
                "In 1D, mirror target must match ambient axis (e.g. 'm-x:x')."
            )
        return sy.ImmutableDenseMatrix([[-1]])

    if dim == 2:
        if len(target) != 1:
            raise ValueError(
                "In 2D, mirror target must be a single axis (the fixed axis)."
            )
        fixed = ambient.index(target[0])
        mutable = sy.eye(dim)
        for idx in range(dim):
            if idx != fixed:
                mutable[idx, idx] = -1
        return sy.ImmutableDenseMatrix(mutable)

    if len(target) != 2:
        raise ValueError("In 3D, mirror target must be a 2-axis plane (e.g. 'yz').")

    fixed_plane = {ambient.index(target[0]), ambient.index(target[1])}
    mutable = sy.eye(dim)
    for idx in range(dim):
        if idx not in fixed_plane:
            mutable[idx, idx] = -1
    return sy.ImmutableDenseMatrix(mutable)


def pointgroup(
    query: str,
    *,
    plane: str | tuple[float, ...] | None = None,
    axis: tuple[float, ...] | None = None,
    spin: Literal["trivial", "electron"] = "electron",
) -> PointGroupElement | FinitePointGroup:
    r"""
    Build a point-group object from a compact query string.

    This is a user-facing constructor for common point operations and named
    crystallographic point groups in Cartesian axes (`x`, `y`, `z`). Compact
    affine queries such as `c4-xy:xy` return a
    [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] with no
    character table. Named crystallographic queries such as `C4v` or `-43m`
    return a [`FinitePointGroup`][qten.pointgroups.finite.FinitePointGroup]
    with packaged ordinary characters and spinor characters computed from
    QTen's SU(2) lift. Use `plane=` / `axis=` at construction to fix the
    spatial frame; `spin="trivial"` is a define-time exception that sets
    `u(g)=I`. Hilbert spaces that already contain `Spin` use the spinor table.

    Query grammar
    -------------
    The accepted format is `"<group>-<ambient>:<target>"`.

    Group tokens
    ------------
    Use `c{n}` for a cyclic rotation of order `n`, such as `c2`, `c3`, or
    `c6`. Use `m` for a mirror reflection. Named crystallographic point groups
    can also be queried by Hermann-Mauguin symbol (`4mm`, `mmm`) or Schoenflies
    alias (`C4v`, `D2h`). Named groups use packaged generator data and may
    include a trailing axis suffix such as `C4v-xy`.

    Axis tokens
    -----------
    `<ambient>` is an ordered ambient axis string using `x`, `y`, and `z`
    without repeats. It defines the space dimension and basis-axis order in the
    returned transform. `<target>` is an axis subset selecting where the group
    action lives.

    Group semantics
    ---------------
    Cyclic groups are interpreted as 2D rotation blocks with angle
    \(\theta = 2\pi/n\).
    For cyclic groups, `<target>` must have exactly two axes and defines the
    rotation plane. In 2D ambient spaces, the cyclic target plane must use the
    same two axes as the ambient space. Cyclic target order controls
    orientation: `c3-xy:xy` and `c3-xy:yx` act on the same plane with inverse
    orientation. In 3D cyclic rotations, the remaining axis is unchanged.

    The active plane receives the block
    \(R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\
    \sin\theta & \cos\theta\end{pmatrix}\), where \(\theta = 2\pi/n\).

    In code, this block is inserted into the returned `irrep` matrix; target
    axis order chooses the sign of `theta`.

    In 1D mirrors, `<target>` must match the ambient axis and the action is a
    sign flip. In 2D mirrors, `<target>` has one axis and denotes the fixed
    axis. In 3D mirrors, `<target>` has two axes and denotes the fixed plane.

    Validation rules
    ----------------
    `ambient` and `target` cannot contain repeated axis letters. `target` must
    be a subset of `ambient`. Invalid dimensional or group combinations raise
    `ValueError`.

    Parameters
    ----------
    query : str
        Compact point-group query of the form `"<group>-<ambient>:<target>"`,
        or a named Hermann-Mauguin / Schoenflies symbol.
    plane : str | tuple[float, ...] | None, optional
        Construction-time plane. A string such as `"xy"` reduces spatial
        matrices while keeping the 3D rotations for spin. A vector is the
        plane normal.
    axis : tuple[float, ...] | None, optional
        Reorient a 3D named group so its standard z-axis maps to this vector.
    spin : Literal["trivial", "electron"], default "electron"
        Define-time spin policy. `"electron"` lifts `rotation3`; `"trivial"`
        uses `u(g)=I`.

    Returns
    -------
    PointGroupElement | FinitePointGroup
        Compact affine queries return a single abelian operation. Named
        crystallographic queries return a finite point group generated by one
        or more exact matrix operations.

    Raises
    ------
    ValueError
        If the query format, group token, axis token, or dimensional
        combination is unsupported.

    Examples
    --------
    ```python
    from qten.pointgroups import pointgroup

    rotation = pointgroup("C6", plane="xy")         # sixfold in xy
    inverse = pointgroup("C6", plane=(0, 0, -1))    # opposite orientation
    mirror = pointgroup("Cs", plane="x")            # 1D spatial, σ_yz spin
    td = pointgroup("Td")                      # tetrahedral
    c4v = pointgroup("C4v", plane="xy")        # 2D spatial, 3D spin
    c3v = pointgroup("C3v", axis=(1, 1, 1))    # C3 about [111]
    flavor = pointgroup("C4v", spin="trivial") # u(g)=I
    ```
    """
    if not _is_affine_query(query):
        return named_pointgroup(
            query,
            plane=_hashable_axis(plane),
            axis=_hashable_axis(axis),
            spin=spin,
        )

    if plane is not None or axis is not None:
        raise ValueError(
            "plane= and axis= apply to named point groups, not affine queries."
        )

    group, ambient, target = _parse_affine_query(query)

    axes_symbols = {
        "x": sy.Symbol("x"),
        "y": sy.Symbol("y"),
        "z": sy.Symbol("z"),
    }
    axes = tuple(axes_symbols[c] for c in ambient)
    if group.startswith("c"):
        n = int(group[1:])
        irrep = _build_cyclic_irrep(n=n, ambient=ambient, target=target)
    elif group == "m":
        irrep = _build_mirror_irrep(ambient=ambient, target=target)
    else:
        raise ValueError(
            f"Unsupported group '{group}'. Supported groups are cyclic and mirror."
        )

    if len(ambient) < 3:
        rotation3 = _embed_in_cartesian_xyz(irrep, tuple(ambient))
    else:
        rotation3 = irrep
    return PointGroupElement(irrep=irrep, axes=axes, rotation3=rotation3, spin=spin)
