from abc import ABC, abstractmethod
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
import sympy as sy
from sympy.matrices.normalforms import smith_normal_decomp  # type: ignore[import-untyped]
from functools import lru_cache
from itertools import product
from typing import List, Tuple, cast

from ..abstracts import Functional
from ..utils.collections_ext import FrozenDict
from ..validations import need_validation
from ..validations.symbolics import check_proper_transformation, check_numerical
from . import (
    AffineSpace,
    Lattice,
    Momentum,
    Offset,
    PeriodicBoundary,
    ReciprocalLattice,
)


@need_validation(check_proper_transformation("M"), check_numerical("M"))
@dataclass(frozen=True)
class AbstractBasisTransform(Functional, ABC):
    M: ImmutableDenseMatrix

    @abstractmethod
    def inv(self) -> "AbstractBasisTransform":
        pass


class BasisTransform(AbstractBasisTransform):
    def inv(self) -> "InverseBasisTransform":
        """Return the inverse-view transform associated with this matrix."""
        return InverseBasisTransform(self.M)


class InverseBasisTransform(AbstractBasisTransform):
    def inv(self) -> BasisTransform:
        """Return the forward-view transform associated with this matrix."""
        return BasisTransform(self.M)


@lru_cache
def _supercell_shifts(
    dim: int, M: ImmutableDenseMatrix
) -> Tuple[ImmutableDenseMatrix, ...]:
    """
    Generate the integer shifts within the supercell defined by M.
    """
    S, U, V = smith_normal_decomp(M, domain=sy.ZZ)
    U_inv = U.inv()
    ranges = [range(int(S[i, i])) for i in range(dim)]
    shifts = [U_inv @ ImmutableDenseMatrix(dim, 1, n) for n in product(*ranges)]
    return tuple(shifts)


def _primitive_label_from_folded_label(label: str) -> str:
    return label.rsplit("_", 1)[0] if "_" in label else label


def _insert_primitive_unit_cell_site(
    unit_cell: dict[str, ImmutableDenseMatrix],
    label: str,
    offset: ImmutableDenseMatrix,
) -> None:
    if label not in unit_cell:
        unit_cell[label] = offset
        return
    if unit_cell[label] == offset:
        return
    suffix = 1
    collision_label = f"{label}_{suffix}"
    while collision_label in unit_cell:
        suffix += 1
        collision_label = f"{label}_{suffix}"
    unit_cell[collision_label] = offset


@BasisTransform.register(AffineSpace)
def affine_space_transform(t: BasisTransform, space: AffineSpace) -> AffineSpace:
    """
    Transform an AffineSpace by the basis transformation M.
    """
    new_basis = space.basis @ t.M
    return AffineSpace(basis=new_basis)


@BasisTransform.register(Lattice)
@lru_cache(maxsize=None)
def lattice_transform(t: BasisTransform, lat: Lattice) -> Lattice:
    """
    Generates a Supercell based on the scaling matrix M.
    Automatically populates the new unit cell with original atoms
    to preserve physical density.
    """
    # 1. Validate M
    shifts = _supercell_shifts(lat.dim, t.M)

    # 4. Transform Atoms
    M_inv = t.M.inv()
    new_unit_cell = {}

    # Iterate over existing atoms (or implicit origin)
    if lat.unit_cell:
        items = cast(List[Tuple[str, Offset]], list(lat.unit_cell.items()))
    else:
        default_offset = Offset(rep=ImmutableDenseMatrix([0] * lat.dim), space=lat)
        items = [("0", default_offset)]

    new_basis = lat.basis @ t.M

    for label, atom_offset in items:
        atom_vec = atom_offset.rep
        for i, k in enumerate(shifts):
            new_frac = M_inv @ (atom_vec + k)
            new_frac = new_frac.applyfunc(lambda x: x - sy.floor(x))
            new_label = f"{label}_{i}" if len(shifts) > 1 else label
            new_unit_cell[new_label] = ImmutableDenseMatrix(new_frac)

    if not isinstance(lat.boundaries, PeriodicBoundary):
        raise NotImplementedError(
            f"BasisTransform currently supports PeriodicBoundary only, got {type(lat.boundaries).__name__}."
        )

    new_boundary_basis = M_inv @ lat.boundaries.basis
    if any(not sy.cancel(x).is_integer for x in new_boundary_basis):
        raise ValueError(
            "Transformed boundary basis must remain integral for PeriodicBoundary."
        )
    new_boundaries = PeriodicBoundary(
        ImmutableDenseMatrix(new_boundary_basis.applyfunc(int))
    )
    return Lattice(
        basis=new_basis,
        boundaries=new_boundaries,
        unit_cell=FrozenDict(new_unit_cell),
    )


@AbstractBasisTransform.register(ReciprocalLattice)
def reciprocal_lattice_transform(
    t: AbstractBasisTransform, lat: ReciprocalLattice
) -> ReciprocalLattice:
    """
    Generate the reciprocal lattice corresponding to the transformed direct lattice.
    """
    dual_lat = lat.dual
    transformed_dual_lat = t(dual_lat)
    return transformed_dual_lat.dual


@AbstractBasisTransform.register(Offset)
def offset_transform(t: AbstractBasisTransform, r: Offset) -> Offset:
    """

    Transform an Offset by the basis transformation M.
    """
    new_space = t(r.space)
    return r.rebase(new_space)


@AbstractBasisTransform.register(Momentum)
def momentum_transform(t: AbstractBasisTransform, momentum: Momentum) -> Momentum:
    """
    Docstring for momentum_transform

    Parameters
    ----------
    """
    new_space = t(momentum.space)
    return momentum.rebase(new_space)


@InverseBasisTransform.register(AffineSpace)
def inverse_affine_space_transform(
    t: InverseBasisTransform, space: AffineSpace
) -> AffineSpace:
    """Transform an AffineSpace by the inverse basis transformation."""
    new_basis = space.basis @ t.M.inv()
    return AffineSpace(basis=new_basis)


@InverseBasisTransform.register(Lattice)
@lru_cache(maxsize=None)
def inverse_lattice_transform(t: InverseBasisTransform, lat: Lattice) -> Lattice:
    """
    Invert a supercell lattice back to a primitive-cell basis defined by M.
    """
    if not isinstance(lat.boundaries, PeriodicBoundary):
        raise NotImplementedError(
            f"InverseBasisTransform currently supports PeriodicBoundary only, got {type(lat.boundaries).__name__}."
        )

    primitive_basis = lat.basis @ t.M.inv()
    primitive_boundary_basis = t.M @ lat.boundaries.basis
    if any(not sy.cancel(x).is_integer for x in primitive_boundary_basis):
        raise ValueError(
            "Inverse-transformed boundary basis must remain integral for PeriodicBoundary."
        )
    primitive_boundaries = PeriodicBoundary(
        ImmutableDenseMatrix(primitive_boundary_basis.applyfunc(int))
    )

    primitive_seed = Lattice(
        basis=ImmutableDenseMatrix(primitive_basis),
        boundaries=primitive_boundaries,
    )
    primitive_unit_cell: dict[str, ImmutableDenseMatrix] = {}
    for folded_label, folded_offset in lat.unit_cell.items():
        primitive_offset = folded_offset.rebase(primitive_seed).fractional()
        primitive_label = _primitive_label_from_folded_label(folded_label)
        _insert_primitive_unit_cell_site(
            primitive_unit_cell,
            primitive_label,
            ImmutableDenseMatrix(primitive_offset.rep),
        )

    return Lattice(
        basis=ImmutableDenseMatrix(primitive_basis),
        boundaries=primitive_boundaries,
        unit_cell=FrozenDict(primitive_unit_cell),
    )
