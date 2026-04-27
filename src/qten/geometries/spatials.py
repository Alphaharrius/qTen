r"""
Geometry primitives for real-space and reciprocal-space coordinates.

This module defines the coordinate objects that the rest of QTen uses to talk
about positions on a finite lattice and momenta on the corresponding sampled
reciprocal grid.

The central convention is:

- An [`AffineSpace`][qten.geometries.spatials.AffineSpace] stores a basis
  matrix whose columns are the primitive vectors of some coordinate frame.
- An [`Offset`][qten.geometries.spatials.Offset] stores coordinates `rep`
  relative to `basis`; in code this is `basis @ rep`, mathematically \(A r\).
- A [`Lattice`][qten.geometries.spatials.Lattice] is an affine space together
  with a periodic identification of cells and an optional multi-site unit cell.
- A [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice] is the
  dual momentum-space lattice with basis \(2\pi(A^{-1})^{\mathsf{T}}\), so
  plane-wave phases can be evaluated directly as
  \(\exp(-\mathrm{i}\, k\cdot r)\) in Cartesian coordinates.
  In code, this basis is built from the direct lattice as
  `2 * sy.pi * basis.inv().T`.

With direct basis matrix \(A\), fractional coordinates \(r\), and reciprocal
basis \(G = 2\pi(A^{-1})^{\mathsf{T}}\), the core coordinate convention is
\(x = A r\), with plane-wave phases written as
\(\exp(-\mathrm{i}\, k \cdot x)\) for Cartesian positions \(x\) and Cartesian
reciprocal vectors \(k\).
The corresponding code expression for the coordinate map is `basis @ rep`.

Throughout the module, "fractional coordinates" means coefficients in the
primitive basis, not Cartesian coordinates. Integer parts label unit-cell
translations, while fractional parts label positions inside a unit cell or
inside the first reciprocal cell.
"""

from dataclasses import dataclass, field
from numbers import Number
from typing import (
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    Mapping,
    Generic,
    Sequence,
    Optional,
)
from typing_extensions import override
from abc import ABC, abstractmethod
from functools import lru_cache
import sympy as sy
import numpy as np
import torch
from sympy import ImmutableDenseMatrix, sympify
from sympy.matrices.normalforms import smith_normal_form  # type: ignore[import-untyped]

from ..utils.collections_ext import FrozenDict
from ..utils.devices import Device
from multimethod import parametric

from ..abstracts import Operable, HasDual, HasBase, Convertible
from ..plottings import Plottable
from .boundary import BoundaryCondition, PeriodicBoundary
from ..validations import need_validation
from ..validations.symbolics import check_invertibility, check_numerical
from ..precision import get_precision_config


@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC):
    """
    Abstract base class for geometry objects with a well-defined spatial dimension.

    Physically, subclasses represent either coordinate systems
    ([`AffineSpace`][qten.geometries.spatials.AffineSpace],
    [`Lattice`][qten.geometries.spatials.Lattice],
    [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice]) or
    vectors/points expressed in those systems
    ([`Offset`][qten.geometries.spatials.Offset],
    [`Momentum`][qten.geometries.spatials.Momentum]).
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimension of the spatial object."""
        raise NotImplementedError()


@need_validation(check_numerical("basis"), check_invertibility("basis"))
@dataclass(frozen=True)
class AffineSpace(Spatial):
    r"""
    Affine coordinate system described by a basis matrix.

    Mathematically, if the basis matrix is \(A = [a_1,\ldots,a_d]\), then a
    column of coordinates \(r\) represents the Cartesian vector

    \(x = A r = \sum_j r_j a_j\).

    This class does not by itself impose periodicity or discreteness. It is the
    ambient continuous coordinate frame in which lattice vectors and unit-cell
    positions are expressed.

    String representations
    ----------------------
    - `str(space)` returns `AffineSpace(basis=...)`, where `basis` is shown as
      a nested Python list of stringified SymPy entries.
    - `repr(space)` is identical to `str(space)`.

    The output is intended to expose the basis matrix directly and does not add
    any extra constructor metadata beyond that basis.

    Attributes
    ----------
    basis : ImmutableDenseMatrix
        Basis matrix whose columns span the affine coordinate system.
    """

    basis: ImmutableDenseMatrix
    r"""
    Basis matrix whose columns span the affine coordinate system. Coordinate
    columns \(r\) in this space represent Cartesian vectors through `basis @ r`
    in code, mathematically \(A r\).
    """

    @property
    def dim(self) -> int:
        """
        Return the geometric dimension of the affine space.

        This is the number of primitive basis vectors, equivalently the number
        of rows of `basis` and the number of coordinates needed to specify a
        point/vector in this frame.
        """
        return self.basis.rows

    def origin(self) -> "Offset":
        """
        Return the zero vector of this affine space.

        Physically this is the chosen coordinate origin. In fractional
        coordinates it is the column of all zeros, and in Cartesian coordinates
        it maps to the zero displacement.
        """
        return Offset(rep=ImmutableDenseMatrix([0] * self.dim), space=self)

    def __str__(self):
        """Return `AffineSpace(basis=...)` with the basis shown entry-by-entry."""
        data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"AffineSpace(basis={data})"

    def __repr__(self):
        """Return the same display string as [`__str__()`][qten.geometries.spatials.AffineSpace.__str__]."""
        return str(self)


_O = TypeVar("_O", bound="Offset")


@lru_cache(maxsize=None)
def _rebase_transform_matrix(
    src: AffineSpace, dest: AffineSpace
) -> ImmutableDenseMatrix:
    return ImmutableDenseMatrix(dest.basis.inv() @ src.basis)


@dataclass(frozen=True)
class AbstractLattice(Generic[_O], AffineSpace, HasDual):
    """
    Common interface for direct and reciprocal lattices.

    Both share the same structure: an affine basis plus a finite set of
    canonical representatives obtained from periodic identifications. The type
    parameter distinguishes whether those representatives are
    [`Offset`][qten.geometries.spatials.Offset] objects in real space or
    [`Momentum`][qten.geometries.spatials.Momentum] objects in reciprocal
    space.
    """

    @property
    def affine(self) -> AffineSpace:
        """
        Return the underlying continuous affine space.

        This forgets the discrete sampled set and keeps only the basis. It is
        useful when you want to talk about arbitrary vectors in the same frame,
        not only allowed lattice sites or sampled momentum points.
        """
        return AffineSpace(basis=self.basis)

    @abstractmethod
    def cartes(
        self,
        T: Type[Union[_O, torch.Tensor, np.ndarray]] | None = None,
        *,
        device: Optional[Device] = None,
    ) -> Union[Tuple[_O, ...], torch.Tensor, np.ndarray]:
        """
        Enumerate the canonical representatives of the finite lattice.

        For a direct lattice this means one representative for every site in
        the finite periodic supercell. For a reciprocal lattice it means one
        representative for every sampled momentum point in the discrete
        Brillouin-zone grid.
        """
        raise NotImplementedError()


@dataclass(frozen=True)
class Lattice(AbstractLattice["Offset"]):
    r"""
    Periodic real-space lattice with an optional multi-site unit cell.

    A `Lattice` combines three ingredients:

    - `basis`, whose columns are the primitive real-space lattice vectors.
    - `boundaries`, which identify lattice translations related by the finite
      periodic supercell.
    - `unit_cell`, which places one or more orbitals/sites inside each
      primitive cell using fractional coordinates.

    If the primitive basis is \(A\) and a site has fractional coordinates
    \(r = n + \tau\), with integer cell index \(n\) and intra-cell offset
    \(\tau\), then its physical Cartesian position is
    \(x = A(n + \tau)\). In code, this is the same `basis @ rep` convention.

    Registered operations
    ---------------------
    The inherited operator dunders from [`Operable`][qten.abstracts.Operable]
    are hidden from the generated API page. For `Lattice`, the concrete
    multimethod behavior defined in this module is:

    - `offset in lattice`: membership test for
      [`Offset`][qten.geometries.spatials.Offset] values. The queried offset is
      first rebased into this lattice, then its fractional part is compared
      against the set of unit-cell site positions. Equivalently, this asks
      whether the point is a lattice site modulo lattice translations.

    No arithmetic operators are registered directly on `Lattice`.

    String representations
    ----------------------
    - `str(lattice)` returns `Lattice(basis=..., boundaries=...)`.
    - The `basis` part is shown as a nested list of stringified SymPy entries.
    - The `boundaries` part uses the boundary object's own `str(...)`
      representation.
    - `repr(lattice)` is identical to `str(lattice)`.

    This representation is meant to show the real-space primitive vectors and
    the finite periodic boundary data, but it does not expand the unit cell.

    Attributes
    ----------
    basis : ImmutableDenseMatrix
        Real-space basis matrix of the lattice.
    boundaries : BoundaryCondition
        Boundary condition defining the finite periodic region.
    _unit_cell_fractional : FrozenDict
        Mapping from unit-cell [`Offset`][qten.geometries.spatials.Offset]
        representatives to their canonical wrapped fractional coordinates.
    """

    boundaries: BoundaryCondition
    """
    Boundary condition defining the finite periodic region and canonical
    representative choice for lattice coordinates.
    """
    _unit_cell_fractional: FrozenDict = field(init=False, repr=False, compare=True)
    """
    Mapping from unit-cell [`Offset`][qten.geometries.spatials.Offset]
    representatives to their canonical wrapped fractional coordinates. This
    cached lookup is used for membership tests and unit-cell normalization.
    """

    def __str__(self):
        """Return `Lattice(basis=..., boundaries=...)` using readable symbolic entries."""
        basis_data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"Lattice(basis={basis_data}, boundaries={self.boundaries})"

    def __repr__(self):
        """Return the same display string as [`__str__()`][qten.geometries.spatials.Lattice.__str__]."""
        return str(self)

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        """
        Return the finite lattice periods along independent primitive directions.

        This is extracted from the Smith normal form of the boundary matrix, so
        it describes the invariant factors of the quotient group of lattice
        translations. For diagonal boundaries it reduces to the familiar system
        size `(L_1, ..., L_d)`.
        """
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
        basis : ImmutableDenseMatrix
            Real-space basis matrix.
        boundaries : BoundaryCondition | None
            Boundary condition defining the periodic region. If omitted,
            `shape` is used to build a [`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary].
        unit_cell : Mapping[str, ImmutableDenseMatrix] | None
            Mapping from site labels to site positions in fractional coordinates.
        shape : Sequence[int] | None
            Legacy shorthand for a diagonal periodic boundary.

        Notes
        -----
        The `unit_cell` positions are stored in fractional lattice
        coordinates. An entry such as `(1/2, 0)` means "halfway along the first
        primitive vector inside the cell", not Cartesian coordinates.
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
            processed_cell[site] = offset
        object.__setattr__(self, "_unit_cell_fractional", FrozenDict(processed_cell))

    @property
    @lru_cache
    def unit_cell(self) -> FrozenDict:
        """
        Return the basis sites/orbitals of one primitive cell.

        Each value is an [`Offset`][qten.geometries.spatials.Offset] whose
        fractional part specifies the site position `τ` inside the unit cell.
        Physically, these are the inequivalent basis positions that are
        repeated by all lattice translations.
        """
        return FrozenDict(
            {
                site: Offset(rep=offset, space=self)
                for site, offset in self._unit_cell_fractional.items()
            }
        )

    @property
    @lru_cache
    def dual(self) -> "ReciprocalLattice":
        r"""
        Return the reciprocal lattice dual to this real-space lattice.

        If the direct basis is `A`, the reciprocal basis is
        \(G = 2\pi (A^{-1})^{\mathsf{T}}\). This convention ensures

        \(\exp(\mathrm{i}\, G_j \cdot A_k) = 1\) for primitive
        direct/reciprocal basis pairs and lets Fourier phases be written
        directly as \(\exp(-\mathrm{i}\, k \cdot r)\).
        """
        reciprocal_basis = 2 * sy.pi * self.basis.inv().T
        return ReciprocalLattice(basis=reciprocal_basis, lattice=self)

    @lru_cache
    @override
    def cartes(
        self,
        T: Type[Union["Offset", torch.Tensor, np.ndarray]] | None = None,
        *,
        device: Optional[Device] = None,
    ) -> Union[Tuple["Offset", ...], torch.Tensor, np.ndarray]:
        r"""
        Enumerate every site in the finite lattice.

        Parameters
        ----------
        T : Type[Union[Offset, torch.Tensor, np.ndarray]]
            Requested return type. [`Offset`][qten.geometries.spatials.Offset] returns lattice-site objects,
            while `torch.Tensor` and `np.ndarray` return Cartesian coordinates
            with shape `(n_sites, dim)`.

        Notes
        -----
        The enumeration consists of one wrapped representative for every
        periodic cell in `boundaries`, combined with every site in `unit_cell`.
        In tensor/array form, the output is already converted to Cartesian
        coordinates with `basis @ rep`, mathematically \(A r\).
        """
        if T == torch.Tensor:
            return _lattice_coords(self, device=device)
        if T == np.ndarray:
            return cast(np.ndarray, _lattice_coords(self).detach().cpu().numpy())
        if T not in (None, Offset):
            raise TypeError(
                f"Unsupported type {T} for cartes. Supported types: Offset, torch.Tensor, np.ndarray"
            )

        elements = self.boundaries.representatives()
        unit_cell_sites = tuple(self.unit_cell.values())
        return tuple(
            Offset(rep=ImmutableDenseMatrix(element + site.rep), space=self)
            for element in elements
            for site in unit_cell_sites
        )

    @lru_cache
    def basis_vectors(self) -> Tuple["Offset", ...]:
        """
        Return the primitive basis vectors as spatial [`Offset`][qten.geometries.spatials.Offset]s.

        If a primitive vector coincides with a valid lattice site modulo unit-cell
        offsets, it is returned in `self`. Otherwise it is returned in
        `self.affine`, since the translation vector is still a valid spatial
        vector even when it is not itself a site of the lattice.

        Physically, these vectors generate translations from one primitive cell
        to neighboring primitive cells. Whether they are also valid "sites"
        depends on the chosen unit-cell basis positions.
        """
        vectors = []
        for j in range(self.dim):
            rep = ImmutableDenseMatrix(
                [sy.Integer(1) if i == j else sy.Integer(0) for i in range(self.dim)]
            )
            candidate = Offset(rep=rep, space=self.affine)
            vectors.append(candidate.rebase(self) if candidate in self else candidate)
        return tuple(vectors)

    def at(
        self, unit_cell: str = "r", cell_offset: Sequence[int] | None = None
    ) -> "Offset[Lattice]":
        """
        Create a lattice offset from a unit-cell site and an integer cell offset.

        Parameters
        ----------
        unit_cell : str
            Label of the site within the unit cell.
        cell_offset : Sequence[int] | None
            Integer translation in lattice coordinates. If omitted, the origin
            cell is used.

        Returns
        -------
        Offset[Lattice]
            The site with fractional coordinates `n + τ`, where `n` is the
            integer `cell_offset` and `τ` is the selected unit-cell position.

        Physically this picks a specific basis site in a specific translated
        unit cell, then wraps it into the canonical representative of the
        finite periodic lattice.
        """
        try:
            site = self.unit_cell[unit_cell]
        except KeyError as e:
            raise KeyError(f"Unknown unit-cell site {unit_cell!r}.") from e

        if cell_offset is None:
            cell_offset = (0,) * self.dim
        elif len(cell_offset) != self.dim:
            raise ValueError(
                f"cell_offset must have length {self.dim}, got {len(cell_offset)}."
            )

        rep = ImmutableDenseMatrix(tuple(cell_offset)) + site.rep
        return Offset(rep=ImmutableDenseMatrix(rep), space=self)


def _lattice_coords(
    lattice: Lattice, *, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Return Cartesian coordinates for every site in a finite lattice.

    Internally this enumerates lattice representatives in fractional
    coordinates, adds the unit-cell offsets, wraps them through the boundary
    identification, and finally converts them to Cartesian coordinates via the
    lattice basis.
    """
    precision = get_precision_config()
    torch_device = device.torch_device() if device is not None else None

    def _as_numeric_row(mat: ImmutableDenseMatrix) -> np.ndarray:
        return np.array(mat.evalf(), dtype=precision.np_float).reshape(-1)

    cell_reps = lattice.boundaries.representatives()
    if not cell_reps:
        return torch.empty(
            (0, lattice.dim), dtype=precision.torch_float, device=torch_device
        )

    lat_reps = np.stack([_as_numeric_row(rep) for rep in cell_reps])

    sorted_unit_cell = sorted(lattice.unit_cell.items(), key=lambda x: str(x[0]))
    basis_reps = np.stack(
        [_as_numeric_row(site_offset.rep) for _, site_offset in sorted_unit_cell]
    )

    total_fractional = lat_reps[:, np.newaxis, :] + basis_reps[np.newaxis, :, :]
    total_fractional_flat = total_fractional.reshape(-1, lattice.dim)

    N_inv = np.array(lattice.boundaries.basis.inv().evalf(), dtype=precision.np_float)
    b = total_fractional_flat @ N_inv.T

    b_rounded = np.round(b)
    b_snapped = np.where(np.isclose(b, b_rounded, atol=1e-10), b_rounded, b)
    b_wrapped = b_snapped % 1.0

    N_mat = np.array(lattice.boundaries.basis.evalf(), dtype=precision.np_float)
    wrapped_fractional_flat = b_wrapped @ N_mat.T

    basis_mat = np.array(lattice.basis.evalf(), dtype=precision.np_float)
    coords_np = wrapped_fractional_flat @ basis_mat.T
    return torch.tensor(coords_np, dtype=precision.torch_float, device=torch_device)


@dataclass(frozen=True)
class ReciprocalLattice(AbstractLattice["Momentum"]):
    r"""
    Reciprocal-space lattice dual to a real-space [`Lattice`][qten.geometries.spatials.Lattice].

    This object represents the finite set of crystal momenta compatible with
    the periodic real-space lattice. Its basis vectors are the reciprocal
    primitive vectors, and its canonical points are the sampled momenta of the
    discrete Brillouin-zone mesh induced by the real-space supercell.

    Registered operations
    ---------------------
    The inherited operator dunders from [`Operable`][qten.abstracts.Operable]
    are hidden from the generated API page. For `ReciprocalLattice`, the
    concrete multimethod behavior defined in this module is:

    - `momentum in reciprocal_lattice`: membership test for
      [`Momentum`][qten.geometries.spatials.Momentum] values. This checks that
      the queried point belongs to the same reciprocal lattice and lies on the
      sampled discrete momentum grid modulo reciprocal lattice vectors.

    No arithmetic operators are registered directly on `ReciprocalLattice`.

    String representations
    ----------------------
    - `str(reciprocal)` returns `ReciprocalLattice(basis=..., shape=...)`.
    - `basis` is shown as a nested list of stringified SymPy entries.
    - `shape` is the canonical finite reciprocal-grid shape derived from the
      dual direct lattice.
    - `repr(reciprocal)` is identical to `str(reciprocal)`.

    The display emphasizes the reciprocal primitive vectors and the sampled
    grid size rather than printing the full underlying direct lattice.

    Attributes
    ----------
    basis : ImmutableDenseMatrix
        Reciprocal basis matrix including the conventional \(2\pi\) factor.
    lattice : Lattice
        Real-space lattice from which this reciprocal lattice is derived.
    """

    lattice: Lattice
    """
    Real-space lattice from which this reciprocal lattice is derived. Its
    boundary data determines the discrete Brillouin-zone sampling shape.
    """

    def __str__(self):
        """Return `ReciprocalLattice(basis=..., shape=...)` for readable inspection."""
        basis_data = [[str(sympify(x)) for x in row] for row in self.basis.tolist()]
        return f"ReciprocalLattice(basis={basis_data}, shape={self.shape})"

    def __repr__(self):
        """Return the same display string as [`__str__()`][qten.geometries.spatials.ReciprocalLattice.__str__]."""
        return str(self)

    @property
    @lru_cache
    def shape(self) -> Tuple[int, ...]:
        """
        Return the reciprocal-grid periods.

        These match the invariant factors of the direct finite lattice. In a
        finite periodic system, the number of allowed momentum samples along
        each independent reciprocal direction is therefore the same as the
        number of real-space periods along the dual direct direction.
        """
        return self.lattice.shape

    @property
    @lru_cache
    def size(self) -> int:
        """
        Return the number of distinct sampled momentum points.

        For a finite periodic lattice, this equals the number of unit-cell
        translation sectors in real space, i.e. the size of the discrete
        translation group.
        """
        return int(np.prod(self.shape))

    @property
    @lru_cache
    def dual(self) -> Lattice:
        """Return the underlying direct-space lattice whose Fourier dual this is."""
        return self.lattice

    @lru_cache
    @override
    def cartes(
        self,
        T: Type[Union["Momentum", torch.Tensor, np.ndarray]] | None = None,
        *,
        device: Optional[Device] = None,
    ) -> Union[Tuple["Momentum", ...], torch.Tensor, np.ndarray]:
        r"""
        Enumerate canonical momentum points.

        Parameters
        ----------
        T : Type[Union[Momentum, torch.Tensor, np.ndarray]]
            Requested return type. [`Momentum`][qten.geometries.spatials.Momentum] returns momentum-point objects, while
            `torch.Tensor` and `np.ndarray` return Cartesian coordinates with
            shape `(n_points, dim)`.

        Notes
        -----
        The allowed points are representatives of the quotient
        \(\mathbb{Z}^d / N^{\mathsf{T}}\mathbb{Z}^d\), where \(N\) is the
        direct-lattice boundary matrix. In fractional reciprocal coordinates
        this means points of the form
        \(\kappa = N^{-\mathsf{T}}m\) modulo integers, which are then wrapped
        into the first reciprocal cell.
        """
        torch_device = device.torch_device() if device is not None else None
        # Enumerate one representative per class in Z^d / N^T Z^d, where N is
        # the direct-lattice boundary basis. Allowed fractional reciprocal
        # coordinates are then N^{-T} m modulo Z^d.
        direct_boundary = self.lattice.boundaries.basis
        dual_boundary = PeriodicBoundary(ImmutableDenseMatrix(direct_boundary.T))
        integer_reps = dual_boundary.representatives()
        dual_transform = ImmutableDenseMatrix(direct_boundary.inv().T)
        momenta = tuple(
            Momentum(
                rep=ImmutableDenseMatrix(dual_transform @ rep), space=self
            ).fractional()
            for rep in integer_reps
        )
        if T in (None, Momentum):
            return momenta
        if T == np.ndarray:
            precision = get_precision_config()
            return np.stack(
                [
                    np.array(momentum.to_vec(np.ndarray), dtype=precision.np_float)
                    for momentum in momenta
                ]
            )
        if T == torch.Tensor:
            precision = get_precision_config()
            return torch.tensor(
                self.cartes(np.ndarray),
                dtype=precision.torch_float,
                device=torch_device,
            )
        raise TypeError(
            f"Unsupported type {T} for cartes. Supported types: Momentum, torch.Tensor, np.ndarray"
        )

    @lru_cache
    def basis_vectors(self) -> Tuple["Offset", ...]:
        """
        Return the primitive reciprocal basis vectors as spatial objects.

        If a primitive reciprocal vector coincides with a sampled momentum point,
        it is returned as a [`Momentum`][qten.geometries.spatials.Momentum] in `self`. Otherwise it is returned as an
        [`Offset`][qten.geometries.spatials.Offset] in `self.affine`.

        Physically, these are the reciprocal vectors that generate translations
        in momentum space by one reciprocal lattice period. A primitive
        reciprocal vector need not itself be one of the finite sampled momenta
        of the discrete grid.
        """
        vectors = []
        for j in range(self.dim):
            rep = ImmutableDenseMatrix(
                [sy.Integer(1) if i == j else sy.Integer(0) for i in range(self.dim)]
            )
            candidate = Offset(rep=rep, space=self.affine)
            momentum_candidate = Momentum(rep=rep, space=self)
            vectors.append(
                momentum_candidate
                if _is_reciprocal_grid_point(self, momentum_candidate, canonical=True)
                else candidate
            )
        return tuple(vectors)


_VecType = TypeVar("_VecType", bound=Union[np.ndarray, ImmutableDenseMatrix])
"""Type variable for vector types that can be returned by `Offset.to_vec()`."""


def _matrix_to_ndarray(mat: ImmutableDenseMatrix) -> np.ndarray:
    precision = get_precision_config()
    return np.array(mat.evalf(), dtype=precision.np_float)


@lru_cache
def _space_basis_as_ndarray(space: AffineSpace) -> np.ndarray:
    return _matrix_to_ndarray(space.basis)


def _cartesian_delta(a: "Offset", b: "Offset", target_space: AffineSpace) -> np.ndarray:
    delta = a - b.rebase(target_space)
    return _space_basis_as_ndarray(target_space) @ _matrix_to_ndarray(delta.rep)


def _is_reciprocal_grid_point(
    lattice: ReciprocalLattice, momentum: "Momentum", *, canonical: bool
) -> bool:
    if momentum.rep.shape != (lattice.dim, 1):
        return False

    rep = momentum.rep if canonical else momentum.fractional().rep
    return all(
        (not canonical or sy.simplify(coord - sy.floor(coord)) == coord)
        and sy.nsimplify(coord * period).is_integer is True
        for coord, period in zip(rep, lattice.shape)
    )


def _check_offset_matches_space(r: "Offset") -> None:
    if r.rep.shape != (r.space.dim, 1):
        raise ValueError(
            f"Invalid Shape: Offset.rep must have shape {(r.space.dim, 1)} to match its affine space, "
            f"got {r.rep.shape}."
        )


S = TypeVar("S", bound=AffineSpace)
"""Generic type for the `AffineSpace`."""

OffsetType = TypeVar("OffsetType", bound="Offset")
"""Type variable for spatial point-like coordinates such as `Offset` and `Momentum`."""


@need_validation(_check_offset_matches_space, check_numerical("rep"))
@dataclass(frozen=True)
class Offset(Generic[S], Spatial, HasBase[S]):
    r"""
    Offset vector in an affine basis.

    An `Offset` stores coordinates in the basis of some
    [`AffineSpace`][qten.geometries.spatials.AffineSpace]. It can represent a
    displacement, a point relative to an origin, or a lattice site position,
    depending on the surrounding context. The physically meaningful Cartesian
    vector is always \(A r\), where \(A\) is `space.basis` and \(r\) is `rep`.

    Registered operations
    ---------------------
    The public arithmetic and comparison operators for
    [`Offset`][qten.geometries.spatials.Offset] are implemented by multimethod
    registrations on [`Operable`][qten.abstracts.Operable]. Those inherited
    `__xxx__` members are hidden from the generated API page, so this section
    is the canonical reference for Offset-specific operator behavior.

    Addition, subtraction, and negation
    -----------------------------------
    - `x + y`: add two offsets. If they are expressed in different affine
      spaces, `y` is first rebased into `x.space`; the result is returned in
      `x.space`.
    - `x - y`: subtract two offsets via `x + (-y)`, again rebasing the
      right-hand operand when needed.
    - `-x`: negate the coordinates while preserving the ambient space.

    These operations preserve the represented geometric vector, up to the
    periodic wrapping rules of a finite [`Lattice`][qten.geometries.spatials.Lattice].

    Scalar multiplication
    ---------------------
    - `c * x` and `x * c` are registered for numeric scalars
      (`numbers.Number`).
    - `expr * x` and `x * expr` are also registered for non-numeric
      `sympy.Expr` values.

    In all four cases, the coordinates are scaled symbolically and the result
    stays in the same ambient space. When that space is a finite lattice, the
    result is normalized to the canonical wrapped representative by
    [`Offset.__post_init__()`][qten.geometries.spatials.Offset.__post_init__].

    Ordered comparisons
    -------------------
    - `x < y`
    - `x > y`

    These are registered only for offset-offset comparisons. If dimensions
    differ, comparison is by dimension. If dimensions match, both operands are
    converted to Cartesian vectors and compared lexicographically. This gives a
    deterministic ordering useful for sorting and tie-breaking, not a physical
    partial order.

    Unsupported operators
    ---------------------
    The following `Operable` operators have no registrations for `Offset` in
    this module and therefore raise `NotImplementedError` when dispatched:

    - containment as the queried object, e.g. `offset in something` unless that
      container type registers support,
    - `<=`, `>=`,
    - matrix multiplication `@`,
    - true division `/` and reflected true division,
    - floor division `//`,
    - exponentiation `**`,
    - logical `&` and `|`.

    String representations
    ----------------------
    - `str(offset)` returns `Offset(rep ∈ basis)` in a compact symbolic form.
    - If `rep` is a column vector, it is flattened and shown as a one-dimensional
      Python list like `['1/2', '0']`.
    - If `rep` is not a single column, it is shown as a nested list preserving
      its matrix shape.
    - The ambient-space basis is always shown as a nested list of stringified
      SymPy entries after the `∈` symbol.
    - `repr(offset)` is identical to `str(offset)`.

    This means the string form shows the coordinate representation and the
    basis it lives in, not the Cartesian vector `space.basis @ rep`
    (mathematically \(A r\)).

    Let \(x = (r_x, S_x)\) and \(y = (r_y, S_y)\), where
    \(r_x, r_y \in \mathbb{R}^{d \times 1}\) are coordinate columns and
    \(S_x, S_y\) are affine spaces with basis matrices \(B_x, B_y\).

    Algebra
    -------
    Negation is \(-x = (-r_x, S_x)\).

    Addition is \(x + y = (r_x + \tilde r_y, S_x)\), where
    \(\tilde r_y = B_x^{-1} B_y r_y\) if \(S_x \ne S_y\)
    (equivalently, rebase \(y\) into \(S_x\) first).

    Subtraction is \(x - y = x + (-y)\).

    Equality
    --------
    Equality is \(x = y \iff (r_x = r_y) \land (S_x = S_y)\). This is exact
    structural equality; no implicit rebasing is applied.

    Order
    -----
    For \(x < y\) and \(x > y\): compare \(d_x\) and \(d_y\) first.
    If \(d_x = d_y\), compare Cartesian tuples
    \(\mathrm{tuple}(B_x r_x)\) and \(\mathrm{tuple}(B_y r_y)\)
    lexicographically.

    Attributes
    ----------
    rep : ImmutableDenseMatrix
        Column vector of coordinates expressed in `space`.
    space : AffineSpace
        Affine space that defines the coordinate basis for `rep`.
    """

    rep: ImmutableDenseMatrix
    r"""
    Column vector of coordinates expressed in `space`. The physically
    represented Cartesian vector is obtained from \(A r\), using
    `space.basis` as \(A\) and `rep` as \(r\).
    """
    space: S
    """
    Affine space that defines the coordinate basis for `rep`, including how
    those coordinates should be interpreted and rebased.
    """

    def __post_init__(self):
        """
        Normalize lattice offsets into the canonical wrapped representative.

        When `space` is a finite [`Lattice`][qten.geometries.spatials.Lattice],
        positions related by the boundary condition are physically equivalent.
        This hook stores the canonical representative chosen by
        `space.boundaries.wrap`.
        """
        if isinstance(self.space, Lattice):
            wrapped = self.space.boundaries.wrap(self.rep)
            object.__setattr__(self, "rep", ImmutableDenseMatrix(wrapped))

    @property
    def dim(self) -> int:
        """
        Return the coordinate dimension of this offset.

        This equals the dimension of the ambient space, not the number of
        physically distinct periodic images.
        """
        return self.rep.rows

    def fractional(self) -> "Offset":
        r"""
        Return the intra-cell fractional part of this offset.

        If \(\mathrm{rep} = n + s\) with integer vector
        \(n = \lfloor\mathrm{rep}\rfloor\) and \(0 \le s_j < 1\), this returns
        the offset with coordinates `s` in the
        same space. On a direct lattice, `n` labels which primitive cell the
        point lies in and `s` labels where it sits inside the cell.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Offset(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)  # Prevent mypy type checking issues

    def base(self) -> S:
        r"""
        Return the affine space whose basis defines these coordinates.

        Mathematically, this is the object that supplies the matrix \(A\) in the
        Cartesian embedding \(x = A\,\mathrm{rep}\). In code, this is
        `space.basis @ rep`.
        """
        return self.space

    def rebase(self, space: S) -> "Offset[S]":
        r"""
        Re-express this Offset in a different AffineSpace.

        Parameters
        ----------
        space : AffineSpace
            The new affine space to express this Offset in.

        Returns
        -------
        Offset
            New Offset expressed in the given affine space.

        Notes
        -----
        Rebasing changes only the coordinates, not the physical vector. If
        \(x = A_{\mathrm{old}}r_{\mathrm{old}} =
        A_{\mathrm{new}}r_{\mathrm{new}}\), this method computes
        \(r_{\mathrm{new}} =
        A_{\mathrm{new}}^{-1}A_{\mathrm{old}}r_{\mathrm{old}}\). In code,
        the transform matrix is `space.basis.inv() @ self.space.basis`.
        """
        rebase_transform_mat = _rebase_transform_matrix(self.space, space)
        new_rep = rebase_transform_mat @ self.rep
        return Offset(rep=ImmutableDenseMatrix(new_rep), space=space)

    def to_vec(self, T: Type[_VecType] = sy.ImmutableMatrix) -> _VecType:
        r"""
        Convert this offset from basis coordinates to Cartesian coordinates.

        If `rep` stores coefficients in the primitive basis, this method returns
        the physical vector \(A r\), using `space.basis` as \(A\) and `rep` as
        \(r\). This is the quantity that should be used in Euclidean geometry
        and Fourier phases.

        Returns
        -------
        ImmutableDenseMatrix
            The Cartesian coordinate vector corresponding to this offset.
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

    def distance(self, r: "Offset") -> float:
        """
        Return the geometric distance to another offset.

        If either offset is expressed on a lattice with periodic boundary
        conditions, the distance is computed using the nearest periodic image
        of the displacement in that lattice. Otherwise, the plain Euclidean
        norm of the displacement in the current affine space is returned.

        Physically, for lattice points this is the minimum-image distance on the
        torus defined by the finite supercell, not the naive distance between
        two unwrapped representatives.
        """
        if isinstance(self.space, Lattice):
            delta = self - r.rebase(self.space)
            return self.space.boundaries.distance(delta.rep, self.space.basis)

        if isinstance(r.space, Lattice):
            delta = r - self.rebase(r.space)
            return r.space.boundaries.distance(delta.rep, r.space.basis)

        target_space = self.space
        delta_cart = _cartesian_delta(self, r, target_space)
        return float(np.linalg.norm(delta_cart.reshape(-1)))

    def __str__(self):
        """
        Return a symbolic display of the stored coordinates and ambient basis.

        Column-vector coordinates are flattened for readability; higher-rank
        matrix representations keep their nested-list structure. The basis of
        `space` is always included so the printed coordinates remain
        unambiguous.
        """
        # If it's a column vector, flatten to 1D python list
        if self.rep.shape[1] == 1:
            vec = [str(sympify(v)) for v in list(self.rep)]
        else:
            vec = [[str(sympify(x)) for x in row] for row in self.rep.tolist()]
        basis = [[str(sympify(x)) for x in row] for row in self.space.basis.tolist()]
        return f"Offset({vec} ∈ {basis})"

    def __repr__(self):
        """Return the same display string as [`__str__()`][qten.geometries.spatials.Offset.__str__]."""
        return str(self)


@Operable.__lt__.register
def _(a: Offset, b: Offset) -> bool:
    if a.dim != b.dim:
        return a.dim < b.dim
    va = a.to_vec(np.ndarray)
    vb = b.to_vec(np.ndarray)
    return tuple(va) < tuple(vb)


@Operable.__gt__.register
def _(a: Offset, b: Offset) -> bool:
    if a.dim != b.dim:
        return a.dim > b.dim
    va = a.to_vec(np.ndarray)
    vb = b.to_vec(np.ndarray)
    return tuple(va) > tuple(vb)


@Operable.__contains__.register
def _(lattice: Lattice, offset: Offset) -> bool:
    rebased = offset.rebase(lattice)
    fractional = rebased.fractional().rep
    unit_cell_offsets = {site.rep for site in lattice.unit_cell.values()}
    return fractional in unit_cell_offsets


@dataclass(frozen=True)
class Momentum(Offset[ReciprocalLattice], Convertible):
    r"""
    Reciprocal-space coordinate expressed in a [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice].

    A `Momentum` is the reciprocal-space analogue of
    [`Offset`][qten.geometries.spatials.Offset]. Its fractional coordinates are
    coefficients in the reciprocal basis vectors. Because the reciprocal basis
    already contains the conventional \(2\pi\), the Cartesian vector returned by
    [`to_vec()`][qten.geometries.spatials.Offset.to_vec] can be inserted
    directly into phases such as \(\exp(-\mathrm{i}\, k\cdot r)\). In code,
    that Cartesian vector is computed from `space.basis @ rep`.

    Registered operations
    ---------------------
    [`Momentum`][qten.geometries.spatials.Momentum] overrides the
    [`Offset`][qten.geometries.spatials.Offset] arithmetic registrations with
    momentum-preserving versions where appropriate.

    Addition, subtraction, and negation
    -----------------------------------
    - `k + q`: add two momenta. If they are expressed in different reciprocal
      lattices, the right-hand operand is first rebased into the left-hand
      space. The result is a `Momentum`.
    - `k - q`: subtract two momenta via `k + (-q)`.
    - `-k`: negate the momentum coordinates and return a `Momentum`.

    These operations preserve the interpretation as reciprocal-space vectors
    instead of falling back to plain [`Offset`][qten.geometries.spatials.Offset]
    results.

    Scalar multiplication
    ---------------------
    `Momentum` does not define separate multiplication registrations in this
    module. It inherits the [`Offset`][qten.geometries.spatials.Offset]
    registrations, and those dispatch through `type(r)(...)`, so both numeric
    and symbolic scalar multiplication still return a `Momentum`:

    - `c * k`, `k * c` for numeric scalars,
    - `expr * k`, `k * expr` for non-numeric `sympy.Expr` scalars.

    Containment and unsupported operators
    -------------------------------------
    `Momentum` itself is the queried value in `momentum in reciprocal_lattice`;
    the actual membership registration lives on
    [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice].

    As for [`Offset`][qten.geometries.spatials.Offset], there are no
    registrations here for `<=`, `>=`, `@`, `/`, reflected `/`, `//`, `**`,
    `&`, or `|`.

    String representations
    ----------------------
    `Momentum` inherits [`Offset.__str__()`][qten.geometries.spatials.Offset.__str__]
    and [`Offset.__repr__()`][qten.geometries.spatials.Offset.__repr__].
    Concretely:

    - `str(momentum)` prints `Offset(rep ∈ basis)`, not a separate
      `Momentum(...)` wrapper.
    - `rep` is the reciprocal-coordinate column, flattened when it is a single
      column.
    - `basis` is the reciprocal-lattice basis, so the display still makes it
      clear that the object lives in momentum space.
    - `repr(momentum)` is identical to `str(momentum)`.

    This is intentionally representation-centric: it shows the stored
    reciprocal coordinates and reciprocal basis directly.

    Attributes
    ----------
    rep : ImmutableDenseMatrix
        Column vector of reciprocal coordinates in fractional form.
    space : ReciprocalLattice
        Reciprocal lattice that defines the basis for `rep`.
    """

    @override
    def fractional(self) -> "Momentum":
        """
        Return the representative in the first reciprocal cell.

        This removes integer reciprocal-lattice translations from the
        coordinates, leaving the sampled momentum modulo reciprocal lattice
        vectors. Physically, momenta differing by an integer reciprocal vector
        represent the same Bloch phase on the direct lattice.
        """
        n = sy.Matrix([sy.floor(x) for x in self.rep])
        s = self.rep - n
        return Momentum(rep=sy.ImmutableDenseMatrix(s), space=self.space)

    fractional = lru_cache(fractional)  # Prevent mypy type checking issues

    def base(self) -> ReciprocalLattice:  # type: ignore[override]
        """
        Return the reciprocal lattice whose basis defines these coordinates.

        This is the momentum-space frame supplying the reciprocal basis vectors
        in which `rep` is expanded.
        """
        assert isinstance(self.space, ReciprocalLattice), (
            "Momentum.space must be a ReciprocalLattice"
        )
        return self.space

    def rebase(self, space: ReciprocalLattice) -> "Momentum":  # type: ignore[override]
        """
        Re-express this Momentum in a different ReciprocalLattice.

        Parameters
        ----------
        space : AffineSpace
            The new affine space (must be a ReciprocalLattice) to express this Momentum in.

        Returns
        -------
        Momentum
            New Momentum expressed in the given reciprocal lattice.

        Notes
        -----
        As with [`Offset.rebase()`][qten.geometries.spatials.Offset.rebase],
        this preserves the physical Cartesian wavevector and only changes the
        coordinate description.
        """
        rebase_transform_mat = _rebase_transform_matrix(self.space, space)
        new_rep = rebase_transform_mat @ self.rep
        return Momentum(rep=ImmutableDenseMatrix(new_rep), space=space)


@Operable.__contains__.register
def _(lattice: ReciprocalLattice, momentum: Momentum) -> bool:
    if momentum.space != lattice:
        return False
    return _is_reciprocal_grid_point(lattice, momentum, canonical=False)


@Operable.__add__.register
def _(a: Offset, b: Offset) -> Offset:
    if a.space != b.space:
        b = b.rebase(a.space)
    new_rep = a.rep + b.rep
    return Offset(rep=ImmutableDenseMatrix(new_rep), space=a.space)


@Operable.__add__.register
def _(a: Momentum, b: Momentum) -> Momentum:
    if a.space != b.space:
        b = b.rebase(a.space)
    new_rep = a.rep + b.rep
    return Momentum(rep=ImmutableDenseMatrix(new_rep), space=a.space)


@Operable.__neg__.register
def _(r: Offset) -> Offset:
    return Offset(rep=-r.rep, space=r.space)


@Operable.__neg__.register
def _(r: Momentum) -> Momentum:
    return Momentum(rep=-r.rep, space=r.space)


@Operable.__sub__.register
def _(a: Offset, b: Offset) -> Offset:
    return a + (-b)


@Operable.__sub__.register
def _(a: Momentum, b: Momentum) -> Momentum:
    return a + (-b)


def _scale_offset(r: _O, scalar: Number | sy.Expr) -> _O:
    return type(r)(rep=ImmutableDenseMatrix(r.rep * scalar), space=r.space)


NonNumericExpr = parametric(sy.Expr, lambda expr: not isinstance(expr, Number))


@Operable.__mul__.register
def _(left: Number, right: Offset) -> Offset:
    return _scale_offset(right, left)


@Operable.__mul__.register
def _(left: Offset, right: Number) -> Offset:
    return _scale_offset(left, right)


@Operable.__mul__.register(NonNumericExpr, Offset)
def _(left: sy.Expr, right: Offset) -> Offset:
    return _scale_offset(right, left)


@Operable.__mul__.register(Offset, NonNumericExpr)
def _(left: Offset, right: sy.Expr) -> Offset:
    return _scale_offset(left, right)
