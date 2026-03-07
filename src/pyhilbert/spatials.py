from dataclasses import dataclass, field
from typing import Tuple, Type, TypeVar, Union, cast, Mapping
from typing_extensions import override
from abc import ABC, abstractmethod
from multipledispatch import dispatch  # type: ignore[import-untyped]
from itertools import product
from functools import lru_cache
import sympy as sy
import numpy as np
import torch
from .precision import get_precision_config
from sympy import ImmutableDenseMatrix, sympify
from sympy.matrices.normalforms import smith_normal_form  # type: ignore[import-untyped]
from .utils import FrozenDict
from .abstracts import Operable, HasDual, HasBase, Plottable
from .boundary import BoundaryCondition


@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError()


@dataclass(frozen=True)
class AffineSpace(Spatial):
    basis: ImmutableDenseMatrix

    # TODO __post_init__ to validate basis is rational / int

    @property
    def dim(self) -> int:
        return self.basis.rows

    def __str__(self):
        data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"AffineSpace(basis={data})"

    def __repr__(self):
        return str(self)


@dataclass(frozen=True)
class AbstractLattice(AffineSpace, HasDual):
    @property
    def affine(self) -> AffineSpace:
        return AffineSpace(basis=self.basis)


@dataclass(frozen=True)
class Lattice(AbstractLattice):
    boundaries: BoundaryCondition
    _unit_cell_fractional: FrozenDict = field(init=False, repr=False, compare=True)

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        S = smith_normal_form(self.boundaries.basis, domain=sy.ZZ)
        return tuple(S.diagonal())

    def __init__(
        self,
        basis: ImmutableDenseMatrix,
        boundaries: BoundaryCondition,
        unit_cell: Mapping[str, ImmutableDenseMatrix],
    ):
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "boundaries", boundaries)

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
        return FrozenDict(
            {
                site: Offset(rep=offset, space=self)
                for site, offset in self._unit_cell_fractional.items()
            }
        )

    @property
    @lru_cache
    def dual(self) -> "ReciprocalLattice":
        reciprocal_basis = 2 * sy.pi * self.basis.inv().T
        return ReciprocalLattice(basis=reciprocal_basis, lattice=self)

    def coords(
        self,
    ) -> torch.Tensor:
        """
        Vectorized calculation of all site coordinates.
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
class ReciprocalLattice(AbstractLattice):
    lattice: Lattice

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        return self.lattice.shape

    @property
    @lru_cache
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    @lru_cache
    def dual(self) -> Lattice:
        return self.lattice


_VecType = TypeVar("_VecType", bound=Union[np.ndarray, ImmutableDenseMatrix])
"""Type variable for vector types that can be returned by `Offset.to_vec()`."""


@dataclass(frozen=True)
class Offset(Spatial, HasBase[Lattice]):
    rep: ImmutableDenseMatrix
    space: Lattice

    def __post_init__(self):
        if self.rep.shape != (self.space.dim, 1):
            raise ValueError("Invalid Shape")
        if isinstance(self.space, Lattice):
            wrapped = self.space.boundaries.wrap(self.rep)
            object.__setattr__(self, "rep", ImmutableDenseMatrix(wrapped))

    @property
    def dim(self) -> int:
        return self.rep.rows

    def fractional(self) -> "Offset":
        """
        Return the fractional coordinates of this Offset within its lattice space.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Offset(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)  # Prevent mypy type checking issues

    def base(self) -> Lattice:
        """Get the `Lattice` this `Offset` is expressed in."""
        return self.space

    def rebase(self, space: Lattice) -> "Offset":
        """
        Re-express this Offset in a different Lattice.

        Parameters
        ----------
        `space` : `Lattice`
            The new lattice to express this Offset in.

        Returns
        -------
        `Offset`
            New Offset expressed in the given lattice.
        """
        rebase_transform_mat = space.basis.inv() @ self.space.basis
        new_rep = rebase_transform_mat @ self.rep
        return Offset(rep=ImmutableDenseMatrix(new_rep), space=space)

    def to_vec(self, T: Type[_VecType]) -> _VecType:
        """Convert this Offset to a vector in Cartesian coordinates by applying
        the basis transformation of its lattice.

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


@dataclass(frozen=True)
class Momentum(Offset, HasBase[ReciprocalLattice]):
    space: ReciprocalLattice  # type: ignore[assignment]

    @override
    def fractional(self) -> "Momentum":
        """
        Return the fractional coordinates of this Offset within its lattice space.
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


@dispatch(Lattice)  # type: ignore[no-redef]
@lru_cache
def cartes(lattice: Lattice) -> Tuple[Offset, ...]:
    elements = product(*tuple(range(n) for n in lattice.shape))
    return tuple(Offset(rep=ImmutableDenseMatrix(el), space=lattice) for el in elements)


@dispatch(ReciprocalLattice)  # type: ignore[no-redef]
def cartes(lattice: ReciprocalLattice) -> Tuple[Momentum, ...]:
    elements = product(*tuple(range(n) for n in lattice.shape))
    sizes = ImmutableDenseMatrix(tuple(sy.Rational(1, n) for n in lattice.shape))
    elements = (ImmutableDenseMatrix(el).multiply_elementwise(sizes) for el in elements)
    return tuple(
        Momentum(rep=ImmutableDenseMatrix(el), space=lattice) for el in elements
    )
