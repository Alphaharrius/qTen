from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar, Union, cast, Mapping, Generic, Sequence
from typing_extensions import override
from abc import ABC, abstractmethod
from multipledispatch import dispatch  # type: ignore[import-untyped]
from itertools import product
from functools import lru_cache
import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix, sympify
from sympy.matrices.normalforms import smith_normal_form  # type: ignore[import-untyped]

from ..utils.collections_ext import FrozenDict
from ..abstracts import Operable, HasDual, HasBase, Convertible
from ..plottings.base import Plottable
from .boundary import BoundaryCondition, PeriodicBoundary
from ..validations import need_validation
from ..validations.symbolics import check_invertibility, check_numerical
from ..precision import get_precision_config


@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC):
    """
    Abstract base class for spatial objects with a well-defined dimension.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the spatial object."""
        raise NotImplementedError()


@need_validation(check_numerical("basis"), check_invertibility("basis"))
@dataclass(frozen=True)
class AffineSpace(Spatial):
    """
    Affine coordinate system described by a basis matrix.

    Attributes
    ----------
    `basis` : `ImmutableDenseMatrix`
        Basis matrix whose columns span the affine coordinate system.
    """

    basis: ImmutableDenseMatrix

    @property
    def dim(self) -> int:
        """Return the dimension induced by the basis matrix."""
        return self.basis.rows

    def __str__(self):
        data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"AffineSpace(basis={data})"

    def __repr__(self):
        return str(self)


_O = TypeVar("_O", bound="Offset")


@dataclass(frozen=True)
class AbstractLattice(Generic[_O], AffineSpace, HasDual):
    """
    Common interface for lattices and reciprocal lattices.
    """

    @property
    def affine(self) -> AffineSpace:
        """Return the underlying affine space of this lattice."""
        return AffineSpace(basis=self.basis)

    @abstractmethod
    def cartes(self) -> Tuple[_O, ...]:
        """Enumerate the canonical coordinates of the lattice."""
        raise NotImplementedError()


@dataclass(frozen=True)
class Lattice(AbstractLattice["Offset"]):
    """
    Periodic real-space lattice with an optional multi-site unit cell.

    Attributes
    ----------
    `basis` : `ImmutableDenseMatrix`
        Real-space basis matrix of the lattice.
    `boundaries` : `BoundaryCondition`
        Boundary condition defining the finite periodic region.
    """

    boundaries: BoundaryCondition
    _unit_cell_fractional: FrozenDict = field(init=False, repr=False, compare=True)

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        """Return the lattice extents along each primitive direction."""
        S = smith_normal_form(self.boundaries.basis, domain=sy.ZZ)
        return tuple(S.diagonal())

    def __init__(
        self,
        basis: ImmutableDenseMatrix,
        boundaries: BoundaryCondition | None = None,
        unit_cell: Mapping[str, ImmutableDenseMatrix] | None = None,
        shape: Sequence[int] | None = None,
    ):
        """
        Construct a lattice from a basis, boundary condition, and unit cell.

        Parameters
        ----------
        `basis` : `ImmutableDenseMatrix`
            Real-space basis matrix.
        `boundaries` : `BoundaryCondition | None`
            Boundary condition defining the periodic region. If omitted,
            `shape` is used to build a `PeriodicBoundary`.
        `unit_cell` : `Mapping[str, ImmutableDenseMatrix] | None`
            Mapping from site labels to site positions in Cartesian coordinates.
            Positions are converted and stored internally in fractional
            coordinates.
        `shape` : `Sequence[int] | None`
            Legacy shorthand for a diagonal periodic boundary.
        """
        object.__setattr__(self, "basis", basis)

        if boundaries is None:
            if shape is None:
                raise TypeError(
                    "Lattice requires either `boundaries` or legacy `shape`."
                )
            if len(shape) != self.dim:
                raise ValueError(
                    f"shape must have length {self.dim}, got {len(shape)}."
                )
            if any(int(n) <= 0 for n in shape):
                raise ValueError(f"shape entries must be positive, got {tuple(shape)}.")
            boundaries = PeriodicBoundary(ImmutableDenseMatrix.diag(*map(int, shape)))
        elif shape is not None and tuple(map(int, shape)) != tuple(
            int(n) for n in smith_normal_form(boundaries.basis, domain=sy.ZZ).diagonal()
        ):
            raise ValueError(
                "`shape` and `boundaries` are inconsistent; provide only one "
                "or ensure they represent the same periodic extents."
            )
        object.__setattr__(self, "boundaries", boundaries)

        if unit_cell is None:
            unit_cell = {"r": ImmutableDenseMatrix([0] * self.dim)}

        if len(unit_cell) == 0:
            raise ValueError(
                "unit_cell is empty; define at least one site in unit_cell."
            )

        processed_cell: dict[str, ImmutableDenseMatrix] = {}
        basis_inverse = self.basis.inv()
        for site, offset in unit_cell.items():
            if not isinstance(offset, ImmutableDenseMatrix):
                raise TypeError(
                    f"unit_cell['{site}'] must be ImmutableDenseMatrix, got {type(offset).__name__}."
                )
            if offset.shape != (self.dim, 1):
                try:
                    offset = offset.reshape(self.dim, 1)
                except Exception as e:
                    raise ValueError(
                        f"unit_cell['{site}'] has shape {offset.shape}; expected ({self.dim}, 1)."
                    ) from e
            fractional_offset = basis_inverse @ offset
            fractional_offset = ImmutableDenseMatrix(fractional_offset)
            processed_cell[site] = fractional_offset
        object.__setattr__(self, "_unit_cell_fractional", FrozenDict(processed_cell))

    @property
    @lru_cache
    def unit_cell(self) -> FrozenDict:
        """Return unit-cell sites as `Offset` objects in this lattice."""
        return FrozenDict(
            {
                site: Offset(rep=offset, space=self)
                for site, offset in self._unit_cell_fractional.items()
            }
        )

    @property
    @lru_cache
    def dual(self) -> "ReciprocalLattice":
        """Return the reciprocal lattice dual to this real-space lattice."""
        reciprocal_basis = 2 * sy.pi * self.basis.inv().T
        return ReciprocalLattice(basis=reciprocal_basis, lattice=self)

    @lru_cache
    @override
    def cartes(self) -> Tuple["Offset", ...]:
        """Enumerate all canonical lattice offsets in the finite region."""
        elements = product(*tuple(range(n) for n in self.shape))
        return tuple(
            Offset(rep=ImmutableDenseMatrix(el), space=self) for el in elements
        )

    def coords(
        self,
    ) -> torch.Tensor:
        """
        Compute Cartesian coordinates for every site in the finite lattice.

        Returns
        -------
        `torch.Tensor`
            Tensor of shape `(n_sites, dim)` containing site coordinates in
            Cartesian form.
        """
        precision = get_precision_config()

        def _as_numeric_row(mat: ImmutableDenseMatrix) -> np.ndarray:
            return np.array(mat.evalf(), dtype=precision.np_float).reshape(-1)

        cell_reps = self.boundaries.representatives()
        if not cell_reps:
            return torch.empty((0, self.dim), dtype=precision.torch_float)

        lat_reps = np.stack(
            [_as_numeric_row(rep) for rep in cell_reps]
        )  # (N_cells, dim)

        sorted_unit_cell = sorted(self.unit_cell.items(), key=lambda x: str(x[0]))
        basis_reps = np.stack(
            [_as_numeric_row(site_offset.rep) for _, site_offset in sorted_unit_cell]
        )  # (N_basis, dim)

        total_fractional = lat_reps[:, np.newaxis, :] + basis_reps[np.newaxis, :, :]
        total_fractional_flat = total_fractional.reshape(-1, self.dim)

        basis_mat = np.array(self.basis.evalf(), dtype=precision.np_float)

        coords_np = total_fractional_flat @ basis_mat.T
        return torch.tensor(coords_np, dtype=precision.torch_float)


@dataclass(frozen=True)
class ReciprocalLattice(AbstractLattice["Momentum"]):
    """
    Reciprocal-space lattice dual to a real-space `Lattice`.

    Attributes
    ----------
    `basis` : `ImmutableDenseMatrix`
        Reciprocal basis matrix including the conventional `2π` factor.
    `lattice` : `Lattice`
        Real-space lattice from which this reciprocal lattice is derived.
    """

    lattice: Lattice

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        """Return the discrete reciprocal-grid shape."""
        return self.lattice.shape

    @property
    @lru_cache
    def size(self) -> int:
        """Return the total number of reciprocal points."""
        return int(np.prod(self.shape))

    @property
    @lru_cache
    def dual(self) -> Lattice:
        """Return the real-space lattice dual to this reciprocal lattice."""
        return self.lattice

    @lru_cache
    @override
    def cartes(self) -> Tuple["Momentum", ...]:
        """Enumerate canonical momentum points in fractional coordinates."""
        element_indices = product(*(range(n) for n in self.shape))
        sizes = ImmutableDenseMatrix(tuple(sy.Rational(1, n) for n in self.shape))
        scaled_elements = (
            ImmutableDenseMatrix(el).multiply_elementwise(sizes)
            for el in element_indices
        )
        return tuple(
            Momentum(rep=ImmutableDenseMatrix(el), space=self) for el in scaled_elements
        )


_VecType = TypeVar("_VecType", bound=Union[np.ndarray, ImmutableDenseMatrix])
"""Type variable for vector types that can be returned by `Offset.to_vec()`."""


def _check_offset_matches_space(r: "Offset") -> None:
    if r.rep.shape != (r.space.dim, 1):
        raise ValueError(
            f"Invalid Shape: Offset.rep must have shape {(r.space.dim, 1)} to match its affine space, "
            f"got {r.rep.shape}."
        )


S = TypeVar("S", bound=AffineSpace)
"""Generic type for the `AffineSpace`."""


@need_validation(_check_offset_matches_space, check_numerical("rep"))
@dataclass(frozen=True)
class Offset(Generic[S], Spatial, HasBase[S]):
    """
    Offset vector in an affine basis.

    Let :math:`x = (r_x, S_x)` and :math:`y = (r_y, S_y)`, where
    :math:`r_x, r_y \\in \\mathbb{R}^{d \\times 1}` are coordinate columns and
    :math:`S_x, S_y` are affine spaces with basis matrices :math:`B_x, B_y`.

    Algebra
    -------
    :math:`-x = (-r_x, S_x)`.

    :math:`x + y = (r_x + \\tilde r_y, S_x)`, where
    :math:`\\tilde r_y = B_x^{-1} B_y r_y` if :math:`S_x \\neq S_y`
    (equivalently, rebase :math:`y` into :math:`S_x` first).

    :math:`x - y = x + (-y)`.

    Equality
    --------
    :math:`x = y \\iff (r_x = r_y) \\land (S_x = S_y)`.
    This is exact structural equality; no implicit rebasing is applied.

    Order
    -----
    For :math:`x < y` and :math:`x > y`:
    compare :math:`d_x` and :math:`d_y` first.
    If :math:`d_x = d_y`, compare Cartesian tuples
    :math:`\\mathrm{tuple}(B_x r_x)` and :math:`\\mathrm{tuple}(B_y r_y)`
    lexicographically.

    Unsupported operators
    ---------------------
    :math:`\\le, \\ge, \\times, @, /, //, ^, \\land, \\lor`
    are not defined for `Offset` and raise `NotImplementedError`.

    Attributes
    ----------
    `rep` : `ImmutableDenseMatrix`
        Column vector of coordinates expressed in `space`.
    `space` : `AffineSpace`
        Affine space that defines the coordinate basis for `rep`.
    """

    rep: ImmutableDenseMatrix
    space: S

    def __post_init__(self):
        """Normalize lattice offsets into the canonical wrapped representative."""
        if isinstance(self.space, Lattice):
            wrapped = self.space.boundaries.wrap(self.rep)
            object.__setattr__(self, "rep", ImmutableDenseMatrix(wrapped))

    @property
    def dim(self) -> int:
        """Return the number of coordinates in this offset."""
        return self.rep.rows

    def fractional(self) -> "Offset":
        """
        Return the fractional coordinates of this Offset within its affine space.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Offset(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)  # Prevent mypy type checking issues

    def base(self) -> S:
        """Get the `AffineSpace` this `Offset` is expressed in."""
        return self.space

    def rebase(self, space: S) -> "Offset[S]":
        """
        Re-express this Offset in a different AffineSpace.

        Parameters
        ----------
        `space` : `AffineSpace`
            The new affine space to express this Offset in.

        Returns
        -------
        `Offset`
            New Offset expressed in the given affine space.
        """
        rebase_transform_mat = space.basis.inv() @ self.space.basis
        new_rep = rebase_transform_mat @ self.rep
        return Offset(rep=ImmutableDenseMatrix(new_rep), space=space)

    def to_vec(self, T: Type[_VecType]) -> _VecType:
        """Convert this Offset to a vector in Cartesian coordinates by applying
        the basis transformation of its affine space.

        Returns
        -------
        `ImmutableDenseMatrix`
            The Cartesian coordinate vector in column format corresponding to this Offset.
        """
        vec = self.space.basis @ self.rep
        if T == ImmutableDenseMatrix:
            return vec
        elif T == np.ndarray:
            precision = get_precision_config()
            return cast(
                _VecType, np.array(vec.evalf(), dtype=precision.np_float).reshape(-1)
            )
        else:
            raise TypeError(
                f"Unsupported type {T} for to_vec. Supported types: np.ndarray, ImmutableDenseMatrix"
            )

    def __str__(self):
        # If it's a column vector, flatten to 1D python list
        if self.rep.shape[1] == 1:
            vec = [str(sympify(v)) for v in list(self.rep)]
        else:
            vec = [[str(sympify(x)) for x in row] for row in self.rep.tolist()]
        basis = [[str(sympify(x)) for x in row] for row in self.space.basis.tolist()]
        return f"Offset({vec} ∈ {basis})"

    def __repr__(self):
        return str(self)


@dispatch(Offset, Offset)  # type: ignore[no-redef]
def operator_lt(a: Offset, b: Offset) -> bool:
    if a.dim != b.dim:
        return a.dim < b.dim
    va = a.to_vec(np.ndarray)
    vb = b.to_vec(np.ndarray)
    return tuple(va) < tuple(vb)


@dispatch(Offset, Offset)  # type: ignore[no-redef]
def operator_gt(a: Offset, b: Offset) -> bool:
    if a.dim != b.dim:
        return a.dim > b.dim
    va = a.to_vec(np.ndarray)
    vb = b.to_vec(np.ndarray)
    return tuple(va) > tuple(vb)


@dataclass(frozen=True)
class Momentum(Offset[ReciprocalLattice], Convertible):
    """
    Reciprocal-space coordinate expressed in a `ReciprocalLattice`.

    Attributes
    ----------
    `rep` : `ImmutableDenseMatrix`
        Column vector of reciprocal coordinates in fractional form.
    `space` : `ReciprocalLattice`
        Reciprocal lattice that defines the basis for `rep`.
    """

    @override
    def fractional(self) -> "Momentum":
        """
        Return the fractional coordinates of this Momentum within its lattice space.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Momentum(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)  # Prevent mypy type checking issues

    def base(self) -> ReciprocalLattice:  # type: ignore[override]
        """Get the `ReciprocalLattice` this `Momentum` is expressed in."""
        assert isinstance(self.space, ReciprocalLattice), (
            "Momentum.space must be a ReciprocalLattice"
        )
        return self.space

    def rebase(self, space: ReciprocalLattice) -> "Momentum":  # type: ignore[override]
        """
        Re-express this Momentum in a different ReciprocalLattice.

        Parameters
        ----------
        `space` : `AffineSpace`
            The new affine space (must be a ReciprocalLattice) to express this Momentum in.

        Returns
        -------
        `Momentum`
            New Momentum expressed in the given reciprocal lattice.
        """

        rebase_transform_mat = space.basis.inv() @ self.space.basis
        new_rep = rebase_transform_mat @ self.rep
        return Momentum(rep=ImmutableDenseMatrix(new_rep), space=space)


@dispatch(Offset, Offset)  # type: ignore[no-redef]
def operator_add(a: Offset, b: Offset) -> Offset:
    if a.space != b.space:
        b = b.rebase(a.space)
    new_rep = a.rep + b.rep
    return Offset(rep=ImmutableDenseMatrix(new_rep), space=a.space)


@dispatch(Momentum, Momentum)  # type: ignore[no-redef]
def operator_add(a: Momentum, b: Momentum) -> Momentum:
    if a.space != b.space:
        b = b.rebase(a.space)
    new_rep = a.rep + b.rep
    return Momentum(rep=ImmutableDenseMatrix(new_rep), space=a.space)


@dispatch(Offset)  # type: ignore[no-redef]
def operator_neg(r: Offset) -> Offset:
    return Offset(rep=-r.rep, space=r.space)


@dispatch(Momentum)  # type: ignore[no-redef]
def operator_neg(r: Momentum) -> Momentum:
    return Momentum(rep=-r.rep, space=r.space)


@dispatch(Offset, Offset)  # type: ignore[no-redef]
def operator_sub(a: Offset, b: Offset) -> Offset:
    return a + (-b)


@dispatch(Momentum, Momentum)  # type: ignore[no-redef]
def operator_sub(a: Momentum, b: Momentum) -> Momentum:
    return a + (-b)
