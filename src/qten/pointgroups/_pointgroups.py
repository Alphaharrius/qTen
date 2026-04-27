"""
Compact point-group constructors.

This module provides the user-facing [`pointgroup()`][qten.pointgroups._pointgroups.pointgroup]
factory for constructing common [`AbelianGroup`][qten.pointgroups.abelian.AbelianGroup]
instances from short query strings. The query language covers simple cyclic
rotations and mirror reflections on Cartesian axes and returns the symbolic
point-group representation used by the rest of QTen.

Repository usage
----------------
Use [`pointgroup()`][qten.pointgroups._pointgroups.pointgroup] for convenient
interactive construction. Use [`AbelianGroup`][qten.pointgroups.abelian.AbelianGroup]
directly when a custom symbolic representation matrix is needed.
"""

import re

import sympy as sy

from .abelian import AbelianGroup


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


def pointgroup(query: str) -> AbelianGroup:
    r"""
    Build an [`AbelianGroup`][qten.pointgroups.abelian.AbelianGroup] from a compact query string.

    This is a user-facing constructor for common point operations in Cartesian
    axes (`x`, `y`, `z`), currently supporting cyclic rotations and mirrors.
    Only these two group families are implemented at present; other group types
    are not yet supported and will raise `ValueError`.

    Query grammar
    -------------
    The accepted format is `"<group>-<ambient>:<target>"`.

    Group tokens
    ------------
    Use `c{n}` for a cyclic rotation of order `n`, such as `c2`, `c3`, or
    `c6`. Use `m` for a mirror reflection.

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

    \[
    R(\theta) =
    \begin{pmatrix}
    \cos\theta & -\sin\theta \\
    \sin\theta & \cos\theta
    \end{pmatrix},
    \qquad
    \theta = \frac{2\pi}{n}.
    \]

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
        Compact point-group query of the form `"<group>-<ambient>:<target>"`.

    Returns
    -------
    AbelianGroup
        Group with `irrep` set to the linear matrix representation from query
        semantics and `axes` set to SymPy symbols in ambient order.

    Raises
    ------
    ValueError
        If the query format, group token, axis token, or dimensional
        combination is unsupported.

    Examples
    --------
    ```python
    from qten.pointgroups import pointgroup

    rotation = pointgroup("c6-xy:xy")     # 60-degree rotation in xy
    inverse = pointgroup("c6-xy:yx")      # inverse orientation
    mirror = pointgroup("m-xyz:yz")       # mirror about the yz-plane
    ```
    """
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

    return AbelianGroup(irrep=irrep, axes=axes)
