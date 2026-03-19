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
    """
    Build an `AbelianGroup` from a compact query string.

    This is a user-facing constructor for common point operations in Cartesian
    axes (`x`, `y`, `z`), currently supporting cyclic rotations and mirrors.
    Only these two group families are implemented at present; other group types
    are not yet supported and will raise `ValueError`.

    Query grammar
    -------------
    The accepted format is:

    `"<group>-<ambient>:<target>"`

    where:
    - `<group>` is:
      - `c{n}` for cyclic rotation of order `n` (e.g. `c2`, `c3`, `c6`, ...),
      - `m` for mirror reflection.
    - `<ambient>` is the ordered ambient axis string (`x`, `y`, `z` without repeats),
      defining the space dimension and basis-axis order in the returned transform.
      Examples: `x`, `xy`, `xyz`, `yzx`.
    - `<target>` chooses where the group action lives (must be subset of ambient).

    Group semantics
    ---------------
    Cyclic (`c{n}`)
    - Always interpreted as a 2D rotation block with angle `2*pi/n`.
    - `<target>` must have exactly 2 axes, defining the rotation plane.
    - In 2D ambient, `<target>` must use the same two axes as ambient.
      Example: `c3-xy:xy` is valid, `c3-xy:xz` is invalid.
    - Target order controls orientation:
      - `c3-xy:xy` and `c3-xy:yx` act on the same plane,
      - the second is the inverse orientation of the first.
    - In 3D ambient, the remaining axis is unchanged.
      Example: `c6-xyz:yz` rotates in the `yz` plane, keeps `x` fixed.

    Mirror (`m`)
    - 1D (`ambient` length 1):
      - `<target>` must match ambient axis,
      - action is sign flip in 1D (`x -> -x`).
    - 2D (`ambient` length 2):
      - `<target>` must have exactly 1 axis and denotes the fixed axis.
      - Example: `m-xy:y` mirrors about the y-axis (`x -> -x`, `y -> y`).
    - 3D (`ambient` length 3):
      - `<target>` must have exactly 2 axes and denotes the fixed plane.
      - Example: `m-xyz:yz` mirrors about the yz-plane (`x -> -x`).

    Validation rules
    ----------------
    - `ambient` and `target` cannot contain repeated axis letters.
    - `target` must be a subset of `ambient`.
    - Invalid dimensional/group combinations raise `ValueError`.

    Return value
    ------------
    Returns an `AbelianGroup` with:
    - `irrep`: the linear matrix representation from query semantics,
    - `axes`: symbols in ambient order.

    Examples
    --------
    Cyclic, 2D:
    - `pointgroup("c6-xy:xy")`:
      60-degree rotation in `xy`.
    - `pointgroup("c6-xy:yx")`:
      inverse orientation of the same `c6`.

    Cyclic, 3D:
    - `pointgroup("c6-xyz:yz")`:
      rotate `yz` by 60 degrees, keep `x` fixed.

    Mirror, 1D:
    - `pointgroup("m-x:x")`:
      reflection in 1D (`x -> -x`).

    Mirror, 2D:
    - `pointgroup("m-xy:y")`:
      mirror about y-axis (`x -> -x`, `y -> y`).
    - `pointgroup("m-xy:x")`:
      mirror about x-axis (`x -> x`, `y -> -y`).

    Mirror, 3D:
    - `pointgroup("m-xyz:yz")`:
      mirror about yz-plane (`x -> -x`).
    - `pointgroup("m-xyz:xz")`:
      mirror about xz-plane (`y -> -y`).
    - `pointgroup("m-xyz:xy")`:
      mirror about xy-plane (`z -> -z`).
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
