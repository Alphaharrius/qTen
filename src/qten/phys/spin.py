r"""
Spin-1/2 irrep labels and SU(2) lifts of spatial point-group rotations.

[`Spin`][qten.phys.spin.Spin] is a typed U(1) irrep for use inside
[`U1Basis`][qten.symbolics.hilbert_space.U1Basis], replacing ad-hoc
`"up"` / `"down"` strings. Crystal rotations act on spin through the
spin-1/2 cover \(u(g)\\in SU(2)\) of the proper part of \(R(g)\\in O(3)\).

Because a generic \(SU(2)\) matrix maps one spin state to a superposition,
the single-outcome operator contract `PointGroupOpr @ Spin -> Spin` cannot
express the full action. Use
[`expand_spin`][qten.phys.spin.expand_spin] together with the spinful Hilbert
representation in [`qten.pointgroups.ops`][qten.pointgroups.ops].
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple

import sympy as sy

from ..abstracts import Operable

if TYPE_CHECKING:
    from ..pointgroups.elements import PointGroupElement, PointGroupOpr
    from ..symbolics import HilbertSpace

_HALF = sy.Rational(1, 2)
_SPIN_MS = (_HALF, -_HALF)


@dataclass(frozen=True)
class Spin(Operable):
    r"""
    Spin-1/2 projection label \(m_s = \pm 1/2\).

    Use as a typed irrep inside [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]:

    ```python
    psi = U1Basis.new(site, Spin.up)
    ```

    Attributes
    ----------
    ms : sy.Rational
        Magnetic quantum number. Must be `+1/2` or `-1/2`.
    """

    ms: sy.Rational

    def __post_init__(self) -> None:
        ms = sy.Rational(self.ms)
        if ms not in _SPIN_MS:
            raise ValueError(f"Spin.ms must be ±1/2, got {self.ms}")
        object.__setattr__(self, "ms", ms)

    @property
    def is_up(self) -> bool:
        return self.ms == _HALF

    @property
    def is_down(self) -> bool:
        return self.ms == -_HALF

    def __str__(self) -> str:
        return "up" if self.is_up else "down"

    def __repr__(self) -> str:
        return f"Spin.{'up' if self.is_up else 'down'}"


Spin.up = Spin(_HALF)  # type: ignore[attr-defined]
Spin.down = Spin(-_HALF)  # type: ignore[attr-defined]


@Operable.__lt__.register
def _(a: Spin, b: Spin) -> bool:
    return a.ms > b.ms  # up (+1/2) before down (-1/2)


@Operable.__gt__.register
def _(a: Spin, b: Spin) -> bool:
    return a.ms < b.ms


def _pauli() -> tuple[
    sy.ImmutableDenseMatrix, sy.ImmutableDenseMatrix, sy.ImmutableDenseMatrix
]:
    sx = sy.ImmutableDenseMatrix([[0, 1], [1, 0]])
    sy_ = sy.ImmutableDenseMatrix([[0, -sy.I], [sy.I, 0]])
    sz = sy.ImmutableDenseMatrix([[1, 0], [0, -1]])
    return sx, sy_, sz


def proper_rotation_matrix(R: sy.Matrix) -> sy.ImmutableDenseMatrix:
    r"""
    Return the proper \(SO(3)\) factor used for the spinor lift.

    For \(\det R = +1\), returns `R`. For improper isometries
    (\(\det R = -1\)), returns `-R` (det \(+1\)), since spatial inversion
    does not act on spin-1/2 and any improper \(g\) is inversion composed
    with a proper rotation.
    """
    M = sy.ImmutableDenseMatrix(R)
    det = sy.simplify(M.det())
    if det == 1:
        return M
    if det == -1:
        return sy.ImmutableDenseMatrix(-M)
    raise ValueError(f"Expected det(R)=±1, got {det}")


def _matrix_cache_key(M: sy.Matrix) -> tuple:
    """Hashable exact key for a SymPy matrix (used to cache SU(2) lifts)."""
    mat = sy.ImmutableDenseMatrix(M)
    return (mat.rows, mat.cols, tuple(sy.simplify(entry) for entry in mat))


@lru_cache(maxsize=256)
def _su2_from_so3_cached(key: tuple) -> sy.ImmutableDenseMatrix:
    rows, cols, entries = key
    M = sy.ImmutableDenseMatrix(rows, cols, list(entries))
    if sy.simplify(M.det()) != 1:
        raise ValueError("su2_from_so3 expects a proper rotation (det=+1)")

    c = sy.simplify((M.trace() - 1) / 2)
    c = sy.Min(sy.Max(c, -1), 1)
    theta = sy.acos(c)

    if sy.simplify(theta) == 0:
        return sy.ImmutableDenseMatrix.eye(2)

    sx, sy_, sz = _pauli()
    skew = sy.simplify(M - M.T)
    nx = sy.simplify(skew[2, 1] / 2)
    ny = sy.simplify(skew[0, 2] / 2)
    nz = sy.simplify(skew[1, 0] / 2)
    nnorm = sy.simplify(sy.sqrt(nx**2 + ny**2 + nz**2))

    if nnorm == 0:
        # θ = π: R = R^T. Axis from R + I = 2 n n^T.
        RpI = sy.simplify(M + sy.eye(3))
        axis = None
        for col in range(3):
            col_vec = RpI[:, col]
            if sy.simplify(col_vec.dot(col_vec)) != 0:
                axis = col_vec
                break
        if axis is None:
            raise ValueError(f"Could not extract axis from 180-degree rotation {M}")
        nvec = sy.ImmutableDenseMatrix(
            sy.simplify(axis / sy.sqrt(axis.dot(axis)))
        )
        nx, ny, nz = (sy.simplify(nvec[i]) for i in range(3))
        u = -sy.I * (nx * sx + ny * sy_ + nz * sz)
        return sy.ImmutableDenseMatrix(sy.simplify(u))

    nvec = sy.ImmutableDenseMatrix(
        sy.simplify(sy.ImmutableDenseMatrix([nx, ny, nz]) / nnorm)
    )
    nx, ny, nz = (sy.simplify(nvec[i]) for i in range(3))
    half = sy.simplify(theta / 2)
    u = sy.cos(half) * sy.eye(2) - sy.I * sy.sin(half) * (
        nx * sx + ny * sy_ + nz * sz
    )
    return sy.ImmutableDenseMatrix(sy.simplify(u))


def su2_from_so3(R: sy.Matrix) -> sy.ImmutableDenseMatrix:
    r"""
    Lift an \(SO(3)\) matrix to one \(SU(2)\) factor \(u(R)\).

    Uses the axis-angle form
    \(u = \cos(\theta/2)\,I - i\sin(\theta/2)\,\hat n\cdot\boldsymbol\sigma\),
    with a continuous branch at the identity.

    Parameters
    ----------
    R : sy.Matrix
        Proper \(3\times 3\) rotation matrix (\(\det = +1\)).

    Returns
    -------
    sy.ImmutableDenseMatrix
        \(2\times 2\) unitary matrix with determinant \(1\).
    """
    return _su2_from_so3_cached(_matrix_cache_key(R))


def su2_of_point_group(
    g: "PointGroupElement | PointGroupOpr",
) -> sy.ImmutableDenseMatrix:
    """Return the SU(2) spin factor for a point-group element / affine wrapper."""
    from ..pointgroups.elements import PointGroupOpr

    element = g.g if isinstance(g, PointGroupOpr) else g
    return su2_from_so3(proper_rotation_matrix(element.irrep))


def su2_numeric(
    g: "PointGroupElement | PointGroupOpr",
) -> list[list[complex]]:
    r"""Complex \(2\times 2\) SU(2) factor for fast Hilbert-space assembly."""
    u = su2_of_point_group(g)
    return [[complex(sy.N(u[i, j])) for j in range(2)] for i in range(2)]



def expand_spin(
    g: "PointGroupElement | PointGroupOpr", spin: Spin
) -> Tuple[Tuple[sy.Expr, Spin], ...]:
    r"""
    Expand \(u(g)|s\rangle\) in the \(\{|\uparrow\rangle,|\downarrow\rangle\}\) basis.

    Returns
    -------
    tuple[tuple[sy.Expr, Spin], ...]
        Nonzero `(amplitude, Spin)` pairs.
    """
    u = su2_of_point_group(g)
    col = 0 if spin.is_up else 1
    out: list[tuple[sy.Expr, Spin]] = []
    for row, target in enumerate((Spin.up, Spin.down)):
        amp = sy.simplify(u[row, col])
        if amp != 0:
            out.append((amp, target))
    if not out:
        raise RuntimeError(f"SU(2) image of {spin} under {g} vanished")
    return tuple(out)


def contains_spin(space: "HilbertSpace") -> bool:
    """Return True if any basis state carries a [`Spin`][qten.phys.spin.Spin] irrep."""
    return any(type(rep) is Spin for psi in space.elements() for rep in psi.base)
