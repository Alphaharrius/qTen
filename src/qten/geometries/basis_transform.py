r"""
Basis-change operators for direct and reciprocal lattice geometry.

This module defines the linear basis-transform machinery used to re-express
geometric objects under a change of lattice basis without moving the
underlying physical points in space. The central idea is that a matrix `M`
changes the coordinate description attached to an
[`AffineSpace`][qten.geometries.spatials.AffineSpace],
[`Lattice`][qten.geometries.spatials.Lattice], or
[`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice], and the
corresponding [`Offset`][qten.geometries.spatials.Offset] and
[`Momentum`][qten.geometries.spatials.Momentum] coordinates are then rebased
into the transformed spaces.

Two complementary conventions are implemented:

- [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
  is the forward-view transform, acting on a direct-lattice basis as
  \(A \mapsto A M\). In repository usage, this is the supercell-building
  convention. In code, the corresponding matrix product is `basis @ M`.
- [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
  is the paired inverse-view transform, acting as \(A \mapsto A M^{-1}\). In
  repository usage, this is the primitive-cell recovery or unfolding
  convention. In code, this uses products with `M.inv()`.

The registrations in this module keep direct and reciprocal lattices
consistent with one another. When a direct lattice basis changes, the
reciprocal basis transforms contragrediently, offsets are rebased into the new
direct space, and momenta are rebased into the new reciprocal space. For
bounded lattices, the boundary-identification matrix is transformed together
with the lattice basis so that the same finite torus is described in the new
coordinate system.

Repository usage
----------------
This module underpins two major workflows in QTen:

- Geometry-level supercell construction and recovery, where a lattice and its
  unit-cell offsets are rewritten in a new basis.
- Band-structure folding and unfolding via
  [`bandfold`][qten.bands.bandfold] and
  [`bandunfold`][qten.bands.bandunfold], which rely on these transforms to map
  between primitive and transformed Brillouin-zone descriptions.

Notes
-----
The concrete lattice registrations in this module currently support
[`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary] explicitly.
For transformed periodic systems, the transformed boundary basis must remain
integral so that the finite quotient lattice is still represented exactly.
"""

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
    """
    Abstract linear basis-change operator parameterized by a matrix `M`.

    [`AbstractBasisTransform`][qten.geometries.basis_transform.AbstractBasisTransform]
    represents a change of coordinates or basis on geometric objects such as
    affine spaces, lattices, offsets, and momenta. Concrete subclasses decide
    whether `M` should be interpreted as the forward transform or as its
    inverse-view companion.

    Parameters
    ----------
    M : ImmutableDenseMatrix
        Square transformation matrix describing the basis change.

    Attributes
    ----------
    M : ImmutableDenseMatrix
        Square transformation matrix describing the basis change.

    Notes
    -----
    Concrete actions are provided through [`Functional`][qten.abstracts.Functional]
    registrations on supported geometric object types.
    """

    M: ImmutableDenseMatrix
    """
    Square transformation matrix describing the basis change. Concrete
    subclasses decide whether this matrix should be read as the forward basis
    map or the inverse-view companion used for the same geometric transform.
    """

    @abstractmethod
    def inv(self) -> "AbstractBasisTransform":
        """
        Return the opposite-view basis transform.

        Returns
        -------
        AbstractBasisTransform
            Transform that undoes this view convention, without inverting the
            stored matrix eagerly unless required by the concrete subclass.
        """
        pass


class BasisTransform(AbstractBasisTransform):
    r"""
    Forward-view basis transform acting with matrix `M`.

    This is the "build a supercell / change to a coarser
    [`Lattice`][qten.geometries.spatials.Lattice] basis" convention used
    throughout the geometry and band-folding code. For a direct-lattice basis
    matrix `A`, the transformed basis is

    \(A' = A M\).

    The same physical point is then re-expressed in the new basis rather than
    moved in space. In other words, `BasisTransform` changes coordinates by
    changing the basis objects attached to the geometry.

    Supported Actions
    -----------------
    `BasisTransform` is registered on the following object types:

    - [`AffineSpace`][qten.geometries.spatials.AffineSpace]
    - [`Lattice`][qten.geometries.spatials.Lattice]
    - [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
    - [`Offset`][qten.geometries.spatials.Offset]
    - [`Momentum`][qten.geometries.spatials.Momentum]

    The registrations are coordinated so that
    [`Lattice`][qten.geometries.spatials.Lattice] and
    [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice] objects
    remain dual to each other.

    Mathematical Action
    -------------------
    Let `A` denote a
    [`Lattice`][qten.geometries.spatials.Lattice]-basis matrix and
    \(B = 2\pi A^{-\mathsf{T}}\) the corresponding
    [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]-basis
    matrix.

    On [`AffineSpace`][qten.geometries.spatials.AffineSpace]
    : \(A' = A M\).

    On [`Lattice`][qten.geometries.spatials.Lattice]
    : the lattice basis becomes \(A M\), while periodic boundary generators are
      re-expressed as \(M^{-1}G\) so that the same physical torus is described
      in the transformed basis. For integer \(M\) with positive determinant,
      this typically produces a supercell with \(\lvert\det M\rvert\) sites in
      the transformed unit cell.

    On [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
    : because reciprocal bases transform contragrediently,
      \(B' = B M^{-\mathsf{T}}\).

    On [`Offset`][qten.geometries.spatials.Offset]
    : an offset with
      [`Lattice`][qten.geometries.spatials.Lattice]-fractional coordinates \(r\)
      is rebased into the transformed space, giving
      \(r' = M^{-1} r\).

    On [`Momentum`][qten.geometries.spatials.Momentum]
    : a momentum with
      [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]-
      fractional coordinates \(\kappa\) are rebased into the transformed
      reciprocal space, giving
      \(\kappa' = M^{\mathsf{T}}\kappa\).

    In implementation terms, these rules correspond to matrix products such as
    `A @ M`, `B @ M.inv().T`, `M.inv() @ r`, and `M.T @ kappa`.

    Repository Usage
    ----------------
    In this repository, `BasisTransform` is used in two main ways:

    - Geometry-level supercell construction via `BasisTransform(lat)`, which
      enlarges the [`Lattice`][qten.geometries.spatials.Lattice] unit cell and
      propagates the corresponding boundary and
      [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
      changes.
    - Band folding via [`bandfold`][qten.bands.bandfold], where the transform
      maps a primitive Brillouin zone onto the Brillouin zone of the transformed
      lattice and enlarges the Hilbert-space legs to match the folded unit
      cell.

    Notes
    -----
    This class does not store \(M^{-1}\). It stores `M` and applies the forward
    convention consistently across dispatches. The opposite convention is
    represented by [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform].
    """

    def inv(self) -> "InverseBasisTransform":
        r"""
        Return the inverse-view transform associated with the same matrix `M`.

        This switches from the forward convention
        \(A \mapsto A M\) to the inverse convention
        \(A \mapsto A M^{-1}\) without changing the stored parameter `M`.
        The returned object therefore represents the basis-change view that
        reverses the action of this [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
        on supported geometry objects.

        Returns
        -------
        InverseBasisTransform
            [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
            carrying the same matrix `M`.

        Notes
        -----
        This method does not replace `M` by \(M^{-1}\) in the dataclass field.
        Instead, it returns the companion transform class whose dispatch rules
        interpret the same matrix using the opposite basis-change convention.

        For example, if this transform maps a
        [`Lattice`][qten.geometries.spatials.Lattice] basis as
        \(A \mapsto A M\), then the returned transform maps it as
        \(A \mapsto A M^{-1}\). Likewise,
        applying `self.inv().inv()` returns a new
        [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
        with the original matrix `M`.
        """
        return InverseBasisTransform(self.M)


class InverseBasisTransform(AbstractBasisTransform):
    r"""
    Inverse-view basis transform paired with [`BasisTransform`][qten.geometries.basis_transform.BasisTransform].

    `InverseBasisTransform(M)` uses the same stored matrix as
    [`BasisTransform`][qten.geometries.basis_transform.BasisTransform], but it
    interprets that matrix as the inverse change of basis. For a
    [`Lattice`][qten.geometries.spatials.Lattice]-basis matrix `A`, the
    transformed basis is

    \(A' = A M^{-1}\).

    This is the "undo the supercell / recover the primitive description"
    convention used when unfolding folded lattices, offsets, and band
    structures back into their primitive representation.

    Supported Actions
    -----------------
    `InverseBasisTransform` directly registers specialized implementations for:

    - [`AffineSpace`][qten.geometries.spatials.AffineSpace]
    - [`Lattice`][qten.geometries.spatials.Lattice]

    Through the shared [`AbstractBasisTransform`][qten.geometries.basis_transform.AbstractBasisTransform]
    registrations, it also acts on:

    - [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
    - [`Offset`][qten.geometries.spatials.Offset]
    - [`Momentum`][qten.geometries.spatials.Momentum]

    Mathematical Action
    -------------------
    Let `A` denote a
    [`Lattice`][qten.geometries.spatials.Lattice]-basis matrix and
    \(B = 2\pi A^{-\mathsf{T}}\) the corresponding
    [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]-basis
    matrix.

    On [`AffineSpace`][qten.geometries.spatials.AffineSpace]
    : \(A' = A M^{-1}\).

    On [`Lattice`][qten.geometries.spatials.Lattice]
    : the lattice basis becomes \(A M^{-1}\), while periodic boundary
      generators are mapped as \(G \mapsto M G\). For lattices produced by
      [`BasisTransform`][qten.geometries.basis_transform.BasisTransform], this
      reconstructs the primitive-cell description and merges folded unit-cell
      labels back onto primitive labels when possible.

    On [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]
    : duality gives
      \(B' = B M^{\mathsf{T}}\).

    On [`Offset`][qten.geometries.spatials.Offset]
    : [`Lattice`][qten.geometries.spatials.Lattice]-fractional coordinates are
      rebased by
      \(r' = M r\).

    On [`Momentum`][qten.geometries.spatials.Momentum]
    : [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]-
      fractional coordinates are rebased by
      \(\kappa' = M^{-\mathsf{T}}\kappa\).

    In implementation terms, these rules correspond to matrix products such as
    `A @ M.inv()`, `B @ M.T`, `M @ r`, and `M.inv().T @ kappa`.

    Repository Usage
    ----------------
    In this repository, `InverseBasisTransform` is primarily used for:

    - Recovering primitive lattices from supercells created by
      [`BasisTransform`][qten.geometries.basis_transform.BasisTransform].
    - Band unfolding via [`bandunfold`][qten.bands.bandunfold], which requires
      an `InverseBasisTransform` explicitly so the direction of the operation is
      unambiguous at runtime.

    Notes
    -----
    `InverseBasisTransform(M).inv()` returns `BasisTransform(M)`. The two
    classes therefore share the same matrix parameter while differing only in
    which side of the basis-change convention they represent.
    """

    def inv(self) -> BasisTransform:
        r"""
        Return the forward-view transform associated with the same matrix `M`.

        This switches from the inverse convention
        \(A \mapsto A M^{-1}\) back to the forward convention
        \(A \mapsto A M\) without changing the stored parameter `M`.
        The returned object is the companion
        [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
        used for supercell construction and band folding.

        Returns
        -------
        BasisTransform
            [`BasisTransform`][qten.geometries.basis_transform.BasisTransform]
            carrying the same matrix `M`.

        Notes
        -----
        This method does not explicitly invert the stored matrix field.
        Instead, it returns the paired transform class whose registrations
        interpret `M` in the forward basis-change convention.

        For example, if this transform maps a
        [`Lattice`][qten.geometries.spatials.Lattice] basis as
        \(A \mapsto A M^{-1}\), then the returned transform maps it as
        \(A \mapsto A M\). Likewise,
        applying `self.inv().inv()` returns a new
        [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
        with the original matrix `M`.
        """
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
def _(t: BasisTransform, space: AffineSpace) -> AffineSpace:
    """
    Transform an AffineSpace by the basis transformation M.
    """
    new_basis = space.basis @ t.M
    return AffineSpace(basis=new_basis)


@BasisTransform.register(Lattice)
@lru_cache(maxsize=None)
def _(t: BasisTransform, lat: Lattice) -> Lattice:
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
def _(t: AbstractBasisTransform, lat: ReciprocalLattice) -> ReciprocalLattice:
    """
    Generate the reciprocal lattice corresponding to the transformed direct lattice.
    """
    dual_lat = lat.dual
    transformed_dual_lat = t(dual_lat)
    return transformed_dual_lat.dual


@AbstractBasisTransform.register(Offset)
def _(t: AbstractBasisTransform, r: Offset) -> Offset:
    """

    Transform an Offset by the basis transformation M.
    """
    new_space = t(r.space)
    return r.rebase(new_space)


@AbstractBasisTransform.register(Momentum)
def _(t: AbstractBasisTransform, momentum: Momentum) -> Momentum:
    """Transform a momentum by rebasing it into the transformed reciprocal space."""
    new_space = t(momentum.space)
    return momentum.rebase(new_space)


@InverseBasisTransform.register(AffineSpace)
def _(t: InverseBasisTransform, space: AffineSpace) -> AffineSpace:
    """Transform an AffineSpace by the inverse basis transformation."""
    new_basis = space.basis @ t.M.inv()
    return AffineSpace(basis=new_basis)


@InverseBasisTransform.register(Lattice)
@lru_cache(maxsize=None)
def _(t: InverseBasisTransform, lat: Lattice) -> Lattice:
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
