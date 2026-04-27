"""
Boundary-condition primitives for finite lattice geometry.

This module defines the abstraction used to identify lattice-coordinate points
modulo a finite translation subgroup and thereby describe bounded lattice
systems. A boundary condition determines three core pieces of finite-geometry
behavior:

- how lattice coordinates are wrapped into a canonical representative region,
- which finite set of cell representatives is enumerated, and
- how boundary-equivalent displacements are compared when measuring distance.

In QTen, boundary objects are attached to
[`Lattice`][qten.geometries.spatials.Lattice] instances and are therefore part
of the geometry contract rather than passive metadata. They control
canonicalization of [`Offset`][qten.geometries.spatials.Offset] coordinates,
enumeration of finite direct-space sites, minimum-image distances on a torus,
and the induced discrete momentum grid used by
[`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice].

The abstract interface is provided by
[`BoundaryCondition`][qten.geometries.boundary.BoundaryCondition]. The concrete
implementation used throughout the current repository is
[`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary], which models a
finite periodic quotient lattice. It uses Smith normal form to support both
diagonal and non-diagonal integer boundary bases while preserving an exact
symbolic description of the finite quotient.

Repository usage
----------------
This module is used by several major geometry workflows:

- [`Lattice`][qten.geometries.spatials.Lattice] uses a boundary object to
  infer shape, enumerate sites, and normalize lattice offsets.
- [`ReciprocalLattice`][qten.geometries.spatials.ReciprocalLattice] derives its
  sampled Brillouin-zone representatives from the direct-lattice boundary
  basis.
- [`BasisTransform`][qten.geometries.basis_transform.BasisTransform] and
  [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
  propagate periodic boundary bases through supercell construction and
  unfolding.
- Region-selection and nearest-neighbor helpers iterate over boundary
  representatives and use boundary-aware distances.

Notes
-----
Although the interface is generic, the surrounding repository currently
assumes finite full-rank boundary data and explicitly supports
[`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary] in basis
transforms and reciprocal-grid construction.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import sympy as sy
from sympy import ImmutableDenseMatrix
from sympy.matrices.normalforms import smith_normal_decomp  # type: ignore[import-untyped]

from ..precision import get_precision_config


def _matrix_to_ndarray(mat: ImmutableDenseMatrix) -> np.ndarray:
    precision = get_precision_config()
    return np.array(mat.evalf(), dtype=precision.np_float)


class BoundaryCondition(ABC):
    r"""
    Abstract interface for identifying lattice-coordinate points modulo a
    finite boundary lattice.

    A `BoundaryCondition` defines how integer or fractional lattice
    coordinates are quotient-ed to obtain a finite simulation region. In this
    package, boundary objects are attached to
    [`Lattice`][qten.geometries.spatials.Lattice] instances and provide the
    canonicalization and finite-cell logic used throughout the geometry,
    Fourier, and band-structure layers.

    Conceptually, the boundary basis specifies a subgroup of lattice
    translations that should be treated as equivalent. The quotient by that
    subgroup determines:

    $$
    \mathbb{Z}^d / B\mathbb{Z}^d,
    $$

    where \(B\) is the boundary basis matrix, stored in code as `basis`.

    - which coordinate representative is considered canonical,
    - which finite set of unit cells should be enumerated,
    - how displacements are reduced when computing distances on a torus, and
    - how finite direct-space boundaries induce the corresponding reciprocal
      sampling grid.

    Repository usage
    ----------------
    `BoundaryCondition` is not only a storage object; it is part of the
    operational contract of several geometry types:

    - [`Lattice`][qten.geometries.spatials.Lattice]
      stores a boundary object in `lattice.boundaries` and uses
      [`representatives`][qten.geometries.boundary.BoundaryCondition.representatives]
      to enumerate the finite set of translated unit cells returned by
      [`Lattice.cartes`][qten.geometries.spatials.Lattice.cartes].
    - [`Offset`][qten.geometries.spatials.Offset]
      automatically applies
      [`wrap`][qten.geometries.boundary.BoundaryCondition.wrap] in
      `Offset.__post_init__` whenever the ambient space is a lattice, so every
      stored lattice-site coordinate is normalized into the boundary's
      canonical fundamental domain.
    - [`Offset.distance`][qten.geometries.spatials.Offset.distance]
      delegates to
      [`distance`][qten.geometries.boundary.BoundaryCondition.distance] to
      compute minimum-image distances whenever either operand lives on a
      bounded lattice.
    - [`ReciprocalLattice.cartes`][qten.geometries.spatials.ReciprocalLattice.cartes]
      derives the discrete Brillouin-zone sampling from the direct-lattice
      boundary basis, so the boundary condition controls both real-space
      finite-size structure and reciprocal-space momentum enumeration.
    - [`BasisTransform`][qten.geometries.basis_transform.BasisTransform] and
      [`InverseBasisTransform`][qten.geometries.basis_transform.InverseBasisTransform]
      transform the boundary basis alongside the lattice basis during
      supercell construction and unfolding. In the current repository
      implementation, these transforms explicitly support
      [`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary].

    Required semantics
    ------------------
    Concrete subclasses are expected to satisfy the following behavioral
    contract:

    - `basis` returns a square matrix describing the translation generators
      that define the identification.
    - `wrap(index)` returns a canonical representative of the equivalence class
      containing `index`.
    - `representatives()` returns exactly one canonical representative for each
      equivalence class in the finite quotient induced by `basis`.
    - `distance(delta, lattice_basis)` measures the physical length of a
      displacement after applying the boundary's identification rule, typically
      by choosing the shortest equivalent image.

    Notes
    -----
    The abstract interface is intentionally generic, but the rest of the
    repository currently assumes a finite, full-rank identification lattice.
    In practice, the concrete implementation used across QTen is
    [`PeriodicBoundary`][qten.geometries.boundary.PeriodicBoundary], which
    interprets the boundary basis as periodic wrapping data and uses Smith
    normal form to enumerate quotient representatives.
    """

    @property
    @abstractmethod
    def basis(self) -> ImmutableDenseMatrix:
        r"""
        Return the matrix that generates the boundary-identification lattice.

        The columns of this square matrix specify the lattice translations
        that are declared equivalent to zero under the boundary condition.
        Equivalently, the boundary identifies coordinates modulo the subgroup
        $$
        B\mathbb{Z}^d.
        $$
        In code, \(B\) is the returned `basis` matrix.

        Repository code uses this matrix as the canonical description of the
        finite geometry:

        - [`Lattice.shape`][qten.geometries.spatials.Lattice.shape] extracts
          the Smith-normal-form invariants of `basis`.
        - [`ReciprocalLattice.cartes`][qten.geometries.spatials.ReciprocalLattice.cartes]
          derives the discrete reciprocal grid from the direct-space boundary
          basis.
        - Basis transforms update this matrix when constructing or inverting
          supercells.

        Returns
        -------
        ImmutableDenseMatrix
            Square matrix describing the translation subgroup used by the
            boundary condition.
        """
        pass

    @abstractmethod
    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """
        Map a lattice-coordinate vector to the canonical representative of its
        boundary-equivalence class.

        Two coordinates are equivalent if they differ by a boundary
        translation generated by [`basis`][qten.geometries.boundary.BoundaryCondition.basis].
        This method chooses one distinguished representative of that class and
        returns it in the same coordinate system.

        In the repository, this operation is performance-critical and
        semantically important because
        [`Offset`][qten.geometries.spatials.Offset] applies it automatically
        when an offset is created on a bounded
        [`Lattice`][qten.geometries.spatials.Lattice]. As a result, many
        higher-level geometry objects rely on `wrap` to keep coordinates in a
        stable canonical form.

        Parameters
        ----------
        index : ImmutableDenseMatrix
            Lattice-coordinate column vector to be reduced modulo the boundary
            identification.

        Returns
        -------
        ImmutableDenseMatrix
            Canonical representative of the equivalence class containing
            `index`.
        """
        pass

    @abstractmethod
    def representatives(self) -> tuple[ImmutableDenseMatrix, ...]:
        """
        Enumerate the finite canonical representative set induced by the
        boundary condition.

        The returned tuple must contain exactly one representative from each
        equivalence class of the quotient lattice. This is the finite set of
        cell translations used to enumerate a bounded lattice.

        Repository code depends on this method in several places:

        - [`Lattice.cartes`][qten.geometries.spatials.Lattice.cartes] builds
          all finite-lattice sites by combining these representatives with the
          unit-cell offsets.
        - [`ReciprocalLattice.cartes`][qten.geometries.spatials.ReciprocalLattice.cartes]
          uses an analogous construction derived from the transposed boundary
          basis to enumerate sampled momenta.
        - Region and nearest-neighbor helpers iterate over these
          representatives when searching a finite torus.

        Returns
        -------
        tuple[ImmutableDenseMatrix, ...]
            One canonical lattice-coordinate representative for each boundary
            equivalence class.
        """
        pass

    @abstractmethod
    def distance(
        self, delta: ImmutableDenseMatrix, lattice_basis: ImmutableDenseMatrix
    ) -> float:
        """
        Measure the physical distance associated with a lattice displacement
        after boundary identification.

        The input `delta` is expressed in lattice coordinates, not Cartesian
        coordinates. Implementations should account for the boundary condition
        when comparing equivalent images of that displacement, then use
        `lattice_basis` to convert the chosen image into physical space before
        computing its norm.

        This method underlies
        [`Offset.distance`][qten.geometries.spatials.Offset.distance] for
        bounded lattices, so it defines the minimum-image or analogous metric
        used by higher-level geometry and region-selection routines.

        Parameters
        ----------
        delta : ImmutableDenseMatrix
            Displacement vector in lattice coordinates.
        lattice_basis : ImmutableDenseMatrix
            Direct-space basis matrix mapping lattice coordinates into
            Cartesian vectors.

        Returns
        -------
        float
            Physical distance assigned to `delta` under the boundary rule.
        """
        pass


@dataclass(frozen=True)
class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary: wraps indices using modulo arithmetic via Smith Normal Form.

    Attributes
    ----------
    _basis : ImmutableDenseMatrix
        Square integer matrix whose columns generate the periodic
        identification lattice.
    _U : ImmutableDenseMatrix
        Left unimodular factor from the Smith normal form of `_basis`.
    _U_inv : ImmutableDenseMatrix
        Inverse of `_U`, cached for coordinate conversions during wrapping.
    _periods : tuple[int, ...]
        Positive Smith invariants defining the finite quotient periods.
    """

    _basis: ImmutableDenseMatrix = field(repr=False)
    """
    Square integer matrix whose columns generate the periodic identification
    lattice. This is the user-supplied periodic cell data that later gets
    decomposed into Smith-normal-form invariants.
    """
    _U: ImmutableDenseMatrix = field(init=False, repr=False, compare=False)
    """
    Left unimodular factor from the Smith normal form of `_basis`, cached so
    lattice coordinates can be moved into the quotient basis efficiently.
    """
    _U_inv: ImmutableDenseMatrix = field(init=False, repr=False, compare=False)
    """
    Inverse of `_U`, cached for coordinate conversions during wrapping and
    representative enumeration.
    """
    _periods: tuple[int, ...] = field(init=False, repr=False, compare=False)
    """
    Positive Smith invariants defining the finite quotient periods of the
    periodic identification lattice.
    """

    def __post_init__(self):
        """
        Validate the boundary basis and cache Smith-normal-form data.

        `PeriodicBoundary` accepts a square integer matrix whose columns
        generate the periodic identification lattice. During initialization,
        the matrix is decomposed via Smith normal form so later calls to
        [`wrap`][qten.geometries.boundary.PeriodicBoundary.wrap] and
        [`representatives`][qten.geometries.boundary.PeriodicBoundary.representatives]
        can work with either diagonal or non-diagonal periodic cells using a
        canonical finite quotient description.

        Raises
        ------
        ValueError
            If the supplied boundary basis is not square, or if its Smith
            invariants indicate a non-full-rank or sign-invalid periodic cell.
        """
        if self._basis.rows != self._basis.cols:
            raise ValueError(f"boundary basis must be square, got {self._basis.shape}.")

        S, U, _ = smith_normal_decomp(self._basis, domain=sy.ZZ)
        S = ImmutableDenseMatrix(S)
        U = ImmutableDenseMatrix(U)
        periods = self._snf_periods(S)

        object.__setattr__(self, "_U", U)
        object.__setattr__(self, "_U_inv", ImmutableDenseMatrix(U.inv()))
        object.__setattr__(self, "_periods", periods)

    @property
    def basis(self) -> ImmutableDenseMatrix:
        """
        Return the periodic identification matrix stored by this boundary.

        For a diagonal matrix, the diagonal entries are the periods along the
        primitive lattice directions. For a non-diagonal matrix, the columns
        span the translation lattice whose quotient defines the periodic
        torus. This matrix is the quantity propagated through lattice basis
        transforms and inspected by lattice-shape and reciprocal-grid code.

        Returns
        -------
        ImmutableDenseMatrix
            Square matrix whose columns generate the periodic identification
            lattice.
        """
        return self._basis

    def wrap(self, index: ImmutableDenseMatrix) -> ImmutableDenseMatrix:
        """
        Reduce a lattice coordinate to this periodic boundary's canonical
        fundamental-domain representative.

        For diagonal boundary bases, this is ordinary component-wise modulo
        reduction. For general full-rank integer bases, the index is first
        expressed in boundary-lattice coordinates, each coefficient is reduced
        modulo one, and the result is mapped back into the original lattice
        coordinates. The returned vector therefore represents the same point on
        the torus as `index`, but in the canonical region used internally by
        the repository.

        This is the normalization used by
        [`Offset.__post_init__`][qten.geometries.spatials.Offset.__post_init__],
        so any offset created on a
        [`Lattice`][qten.geometries.spatials.Lattice] with
        `PeriodicBoundary` is stored in this wrapped form.

        Parameters
        ----------
        index : ImmutableDenseMatrix
            Lattice-coordinate column vector to wrap. The shape must be
            `(dim, 1)` where `dim` matches the boundary basis dimension.

        Returns
        -------
        ImmutableDenseMatrix
            Canonical wrapped column vector representing the same
            boundary-equivalence class as `index`.

        Raises
        ------
        ValueError
            If `index` does not have the expected column-vector shape.
        """
        expected_shape = (self.basis.rows, 1)
        if index.shape != expected_shape:
            raise ValueError(
                f"index shape {index.shape} does not match expected {expected_shape}."
            )

        if self.basis.is_diagonal():
            wrapped_entries = [
                sy.Mod(index[i, 0], int(self.basis[i, i]))
                for i in range(self.basis.rows)
            ]
            return ImmutableDenseMatrix(self.basis.rows, 1, wrapped_entries)

        coordinates = ImmutableDenseMatrix(self.basis.inv() @ index)
        wrapped_entries = [
            coordinates[i, 0] - sy.floor(coordinates[i, 0])
            for i in range(self.basis.rows)
        ]
        wrapped_coords = ImmutableDenseMatrix(self.basis.rows, 1, wrapped_entries)

        # Avoid applying sy.Rational blindly as the wrapped result could contain
        # irrational expressions (e.g. sqrt). We rationalise Floats within expressions.
        def _rationalize(expr):
            if hasattr(expr, "is_Float") and expr.is_Float:
                return sy.nsimplify(expr)
            elif hasattr(expr, "args") and expr.args:
                return expr.func(*[_rationalize(arg) for arg in expr.args])
            return expr

        return ImmutableDenseMatrix(self.basis @ wrapped_coords).applyfunc(_rationalize)

    def representatives(self) -> tuple[ImmutableDenseMatrix, ...]:
        r"""
        Enumerate the canonical finite set of lattice representatives for this
        periodic torus.

        For diagonal periodicities, the representatives are the obvious
        integer box
        $$
        0 \le n_i < \mathrm{basis}_{ii}.
        $$
        In code, the upper bound is `basis[i, i]`.
        For non-diagonal cells, the method enumerates the quotient described by
        the Smith normal form and then maps those elements back into canonical
        wrapped lattice coordinates.

        The size of the returned tuple is the index of the boundary lattice in
        the ambient lattice, which is the number of translated unit cells in
        the finite periodic system. This tuple drives finite-lattice site
        enumeration in
        [`Lattice.cartes`][qten.geometries.spatials.Lattice.cartes].

        Returns
        -------
        tuple[ImmutableDenseMatrix, ...]
            Tuple of wrapped lattice-coordinate representatives spanning the
            finite quotient \(\mathbb{Z}^d / B\mathbb{Z}^d\), where \(B\) is
            the `basis` matrix.
        """
        if self.basis.is_diagonal():
            elements = product(
                *(range(int(self.basis[i, i])) for i in range(self.basis.rows))
            )
            return tuple(
                ImmutableDenseMatrix(self.basis.rows, 1, el) for el in elements
            )

        elements = product(*(range(period) for period in self._periods))
        # Ensure representatives are within the fundamental domain by wrapping them
        return tuple(
            self.wrap(
                ImmutableDenseMatrix(
                    self._U_inv @ ImmutableDenseMatrix(self.basis.rows, 1, el)
                )
            )
            for el in elements
        )

    def distance(
        self, delta: ImmutableDenseMatrix, lattice_basis: ImmutableDenseMatrix
    ) -> float:
        """
        Compute the minimum-image distance associated with a periodic lattice
        displacement.

        The displacement `delta` is given in lattice coordinates. This method
        converts it to Cartesian space using `lattice_basis`, considers nearby
        periodic images obtained by adding boundary translations, and returns
        the Euclidean norm of the shortest candidate. That is the metric used
        throughout the repository for distances on lattices with periodic
        boundaries.

        The current implementation evaluates the nearest images generated by
        shifts with coefficients in `{-1, 0, 1}` along the boundary
        generators, which is sufficient for the fundamental displacements
        produced by the surrounding geometry code.

        Parameters
        ----------
        delta : ImmutableDenseMatrix
            Lattice-coordinate displacement column vector with shape
            `(dim, 1)`.
        lattice_basis : ImmutableDenseMatrix
            Real-space lattice basis used to convert lattice coordinates into
            Cartesian displacements.

        Returns
        -------
        float
            Euclidean norm of the shortest boundary-equivalent Cartesian
            displacement.

        Raises
        ------
        ValueError
            If `delta` does not have the expected column-vector shape.
        """
        expected_shape = (self.basis.rows, 1)
        if delta.shape != expected_shape:
            raise ValueError(
                f"delta shape {delta.shape} does not match expected {expected_shape}."
            )
        coeffs = np.array(
            tuple(product((-1, 0, 1), repeat=self.basis.rows)),
            dtype=get_precision_config().np_float,
        )
        physical_boundaries = _matrix_to_ndarray(lattice_basis) @ _matrix_to_ndarray(
            self.basis
        )
        delta_cart = _matrix_to_ndarray(lattice_basis) @ _matrix_to_ndarray(delta)
        candidate_displacements = (
            delta_cart.reshape(1, -1) + coeffs @ physical_boundaries.T
        )
        return float(np.linalg.norm(candidate_displacements, axis=1).min())

    def __str__(self):
        data = [[str(sy.sympify(x)) for x in row] for row in self._basis.tolist()]
        return f"PeriodicBoundary(basis={data})"

    def __repr__(self):
        return str(self)

    @staticmethod
    def _snf_periods(S: ImmutableDenseMatrix) -> tuple[int, ...]:
        periods: list[int] = []
        for i in range(S.rows):
            invariant = S[i, i]
            period = int(invariant)
            if period == 0:
                raise ValueError(
                    "boundary basis must be full-rank (non-zero SNF invariants)."
                )
            if period < 0:
                raise ValueError(
                    f"SNF invariant factors must be positive, got {invariant}."
                )
            periods.append(period)
        return tuple(periods)
