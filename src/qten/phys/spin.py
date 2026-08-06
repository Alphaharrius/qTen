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
_CARTESIAN_AXES = ("x", "y", "z")
_ROTATION_TOL = 1e-10


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


def _expr_is_zero(value: sy.Expr, *, tol: float = _ROTATION_TOL) -> bool:
    """Return whether an exact or numerical SymPy expression is effectively zero."""
    value = sy.simplify(value)
    if value == 0:
        return True
    if value.free_symbols:
        equals = value.equals(0)
        return bool(equals)
    try:
        return abs(complex(sy.N(value))) <= tol
    except (TypeError, ValueError):
        return False


def _matrix_is_zero(matrix: sy.Matrix, *, tol: float = _ROTATION_TOL) -> bool:
    return all(_expr_is_zero(entry, tol=tol) for entry in matrix)


def _validated_o3_matrix(
    R: sy.Matrix, *, require_proper: bool
) -> tuple[sy.ImmutableDenseMatrix, int]:
    """Validate a real orthogonal 3D matrix and classify its determinant."""
    M = sy.ImmutableDenseMatrix(R)
    if M.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got shape={M.shape}")
    if any(not _expr_is_zero(sy.im(entry)) for entry in M):
        raise ValueError("Rotation matrix entries must be real")
    orthogonality_error = sy.ImmutableDenseMatrix(sy.simplify(M.T @ M - sy.eye(3)))
    if not _matrix_is_zero(orthogonality_error):
        raise ValueError("Expected an orthogonal rotation matrix with R.T @ R = I")

    det = sy.simplify(M.det())
    if _expr_is_zero(det - 1):
        det_sign = 1
    elif _expr_is_zero(det + 1):
        det_sign = -1
    else:
        raise ValueError(f"Expected det(R)=±1, got {det}")
    if require_proper and det_sign != 1:
        raise ValueError("su2_from_so3 expects a proper rotation (det=+1)")
    return M, det_sign


def proper_rotation_matrix(R: sy.Matrix) -> sy.ImmutableDenseMatrix:
    r"""
    Return the proper \(SO(3)\) factor used for the spinor lift.

    For \(\det R = +1\), returns `R`. For improper isometries
    (\(\det R = -1\)), returns `-R` (det \(+1\)), since spatial inversion
    does not act on spin-1/2 and any improper \(g\) is inversion composed
    with a proper rotation.
    """
    M, det_sign = _validated_o3_matrix(R, require_proper=False)
    if det_sign == 1:
        return M
    return sy.ImmutableDenseMatrix(-M)


def _matrix_cache_key(M: sy.Matrix) -> tuple:
    """Hashable exact key for a SymPy matrix (used to cache SU(2) lifts)."""
    mat = sy.ImmutableDenseMatrix(M)
    return (mat.rows, mat.cols, tuple(sy.simplify(entry) for entry in mat))


def _numeric_su2_from_so3(M: sy.Matrix) -> sy.ImmutableDenseMatrix:
    """Robust quaternion lift for an inexact numerical rotation matrix."""
    r = [[float(sy.N(M[i, j])) for j in range(3)] for i in range(3)]
    trace = r[0][0] + r[1][1] + r[2][2]

    if trace > 0.0:
        scale = 2.0 * max(0.0, trace + 1.0) ** 0.5
        w = 0.25 * scale
        x = (r[2][1] - r[1][2]) / scale
        y = (r[0][2] - r[2][0]) / scale
        z = (r[1][0] - r[0][1]) / scale
    elif r[0][0] >= r[1][1] and r[0][0] >= r[2][2]:
        scale = 2.0 * max(0.0, 1.0 + r[0][0] - r[1][1] - r[2][2]) ** 0.5
        x = 0.25 * scale
        w = (r[2][1] - r[1][2]) / scale
        y = (r[0][1] + r[1][0]) / scale
        z = (r[0][2] + r[2][0]) / scale
    elif r[1][1] >= r[2][2]:
        scale = 2.0 * max(0.0, 1.0 + r[1][1] - r[0][0] - r[2][2]) ** 0.5
        y = 0.25 * scale
        w = (r[0][2] - r[2][0]) / scale
        x = (r[0][1] + r[1][0]) / scale
        z = (r[1][2] + r[2][1]) / scale
    else:
        scale = 2.0 * max(0.0, 1.0 + r[2][2] - r[0][0] - r[1][1]) ** 0.5
        z = 0.25 * scale
        w = (r[1][0] - r[0][1]) / scale
        x = (r[0][2] + r[2][0]) / scale
        y = (r[1][2] + r[2][1]) / scale

    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ValueError(f"Could not extract a quaternion from rotation {M}")
    w, x, y, z = (component / norm for component in (w, x, y, z))

    # Choose the principal SO(3) branch. Both signs are valid lifts, but this
    # convention preserves continuity from the identity for angles below π.
    if w < 0.0:
        w, x, y, z = (-component for component in (w, x, y, z))

    return sy.ImmutableDenseMatrix(
        [
            [sy.Float(w) - sy.I * sy.Float(z), -sy.Float(y) - sy.I * sy.Float(x)],
            [sy.Float(y) - sy.I * sy.Float(x), sy.Float(w) + sy.I * sy.Float(z)],
        ]
    )


@lru_cache(maxsize=256)
def _su2_from_so3_cached(key: tuple) -> sy.ImmutableDenseMatrix:
    rows, cols, entries = key
    M = sy.ImmutableDenseMatrix(rows, cols, list(entries))

    c = sy.simplify((M.trace() - 1) / 2)
    c = sy.Min(sy.Max(c, -1), 1)
    theta = sy.acos(c)

    if _expr_is_zero(theta):
        return sy.ImmutableDenseMatrix.eye(2)

    sx, sy_, sz = _pauli()
    skew = sy.simplify(M - M.T)
    nx = sy.simplify(skew[2, 1] / 2)
    ny = sy.simplify(skew[0, 2] / 2)
    nz = sy.simplify(skew[1, 0] / 2)
    nnorm = sy.simplify(sy.sqrt(nx**2 + ny**2 + nz**2))

    if _expr_is_zero(nnorm):
        # At or numerically near θ = π, the skew part is ill-conditioned.
        # Use R + I ≈ 2 n n^T and select its largest-norm column so roundoff
        # residue in a nominally zero column cannot determine the axis.
        RpI = sy.simplify(M + sy.eye(3))
        axis_candidates: list[tuple[float | None, sy.Matrix]] = []
        for col in range(3):
            col_vec = RpI[:, col]
            norm_sq = sy.simplify(col_vec.dot(col_vec))
            if _expr_is_zero(norm_sq):
                continue
            try:
                numeric_norm_sq: float | None = abs(complex(sy.N(norm_sq)))
            except (TypeError, ValueError):
                numeric_norm_sq = None
            axis_candidates.append((numeric_norm_sq, col_vec))
        if not axis_candidates:
            raise ValueError(f"Could not extract axis from 180-degree rotation {M}")
        numeric_candidates = [
            candidate for candidate in axis_candidates if candidate[0] is not None
        ]
        axis = (
            max(
                numeric_candidates,
                key=lambda candidate: (
                    candidate[0] if candidate[0] is not None else -1.0
                ),
            )[1]
            if numeric_candidates
            else axis_candidates[0][1]
        )

        # For a near-π (rather than exact π) rotation, orient the axis using
        # the small but sign-carrying skew vector. At exact π either sign is
        # an equally valid SU(2) lift.
        orientation = sy.simplify(axis.dot(sy.ImmutableDenseMatrix([nx, ny, nz])))
        if orientation != 0 and not orientation.free_symbols:
            try:
                numeric_orientation = complex(sy.N(orientation))
                if (
                    abs(numeric_orientation.imag) <= _ROTATION_TOL
                    and numeric_orientation.real < 0
                ):
                    axis = -axis
            except (TypeError, ValueError):
                pass
        nvec = sy.ImmutableDenseMatrix(sy.simplify(axis / sy.sqrt(axis.dot(axis))))
        nx, ny, nz = (sy.simplify(nvec[i]) for i in range(3))
        half = sy.simplify(theta / 2)
        u = sy.cos(half) * sy.eye(2) - sy.I * sy.sin(half) * (
            nx * sx + ny * sy_ + nz * sz
        )
        return sy.ImmutableDenseMatrix(sy.simplify(u))

    nvec = sy.ImmutableDenseMatrix(
        sy.simplify(sy.ImmutableDenseMatrix([nx, ny, nz]) / nnorm)
    )
    nx, ny, nz = (sy.simplify(nvec[i]) for i in range(3))
    half = sy.simplify(theta / 2)
    u = sy.cos(half) * sy.eye(2) - sy.I * sy.sin(half) * (nx * sx + ny * sy_ + nz * sz)
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
    M, _ = _validated_o3_matrix(R, require_proper=True)
    if any(entry.free_symbols for entry in M):
        raise ValueError(
            "Parameterized symbolic rotations are not supported; substitute "
            "numerical or exact constant parameter values before lifting to SU(2)."
        )
    if any(entry.atoms(sy.Float) for entry in M):
        return _numeric_su2_from_so3(M)
    return _su2_from_so3_cached(_matrix_cache_key(M))


def _canonical_cartesian_rotation(
    g: "PointGroupElement | PointGroupOpr",
) -> sy.ImmutableDenseMatrix:
    """Return a point operation in QTen's canonical Cartesian spin frame."""
    from ..pointgroups.elements import PointGroupOpr

    element = g.g if isinstance(g, PointGroupOpr) else g
    axis_names = tuple(getattr(axis, "name", str(axis)) for axis in element.axes)
    if axis_names != _CARTESIAN_AXES:
        raise ValueError(
            "Spin-1/2 lifts require canonical Cartesian axes (x, y, z), "
            f"got {axis_names}"
        )

    matrix = sy.ImmutableDenseMatrix(element.irrep)
    if isinstance(g, PointGroupOpr):
        basis = sy.ImmutableDenseMatrix(g.base().basis)
        if basis.shape != (3, 3):
            raise ValueError(
                "Spin-1/2 lifts require a three-dimensional affine-space basis"
            )
        matrix = sy.ImmutableDenseMatrix(sy.simplify(basis @ matrix @ basis.inv()))
    return matrix


def su2_of_point_group(
    g: "PointGroupElement | PointGroupOpr",
) -> sy.ImmutableDenseMatrix:
    """Return the SU(2) factor in the canonical Cartesian spin frame."""
    cartesian = _canonical_cartesian_rotation(g)
    return su2_from_so3(proper_rotation_matrix(cartesian))


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
