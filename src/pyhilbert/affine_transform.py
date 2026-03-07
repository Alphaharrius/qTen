from dataclasses import dataclass
import re
from typing import Dict, Literal, Tuple, cast
from collections import OrderedDict
from itertools import product
from functools import lru_cache, partial, reduce

import sympy as sy
import torch

from .abstracts import HasBase, Functional, Gaugable, GaugeBasis, Gauged, GaugeInvariant
from .spatials import Lattice, Spatial, Offset, Momentum
from .hilbert import Mode, MomentumSpace, HilbertSpace, hilbert
from .tensors import Tensor, mapping_matrix
from .fourier import fourier_transform
from .boundary import PeriodicBoundary
from .utils import FrozenDict


@dataclass(frozen=True)
class AbelianIrrep(Spatial, Gaugable, GaugeBasis):
    """
    Symbolic affine function expressed in a polynomial basis over given axes.

    Attributes
    ----------
    `expr`: `sy.Expr`
        Symbolic expression in `axes` representing the affine function.
    `axes`: `Tuple[sy.Symbol, ...]`
        Ordered tuple of symbols defining the coordinate axes.
    `order`: `int`
        Polynomial order used to build the basis representation.
    `rep`: `sy.ImmutableDenseMatrix`
        Coefficient vector in the Euclidean monomial basis (column matrix).
    """

    expr: sy.Expr
    axes: Tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix

    @property
    def dim(self):
        """Number of axes (spatial dimension) for this affine function."""
        return len(self.axes)

    def __str__(self):
        return f"AbelianIrrep({str(self.expr)})"

    def __repr__(self):
        return f"AbelianIrrep({repr(self.expr)})"


@dataclass(frozen=True)
class AbelianIrrepSet(Gaugable, GaugeBasis):
    """
    Immutable collection of :class:`AbelianIrrep` objects that share a common
    algebraic context.

    This container is used to pass multiple symbolic affine irreducible
    representations through APIs that operate on gauge bases and gaugeable
    objects. Keeping them in a dedicated dataclass makes the set hashable,
    explicit, and compatible with memoization-heavy workflows in this module.

    Use this class when you need to:
    - represent several affine irreps as a single semantic unit,
    - preserve deterministic ordering of irreps for basis construction, or
    - provide a gauge-aware batch input to downstream transformations.

    Parameters
    ----------
    `irreps`: `Tuple[AbelianIrrep, ...]`
        Ordered, immutable tuple of affine irreps. The tuple order defines the
        canonical iteration order for any downstream basis or mapping logic.

    Notes
    -----
    This class intentionally does not coerce or validate element compatibility
    (e.g. axis set or polynomial order alignment). Upstream construction code
    should enforce those invariants when required by a specific algorithm.
    """

    irreps: Tuple[AbelianIrrep, ...]


@dataclass(frozen=True)
class AffineGroupElement(Functional, HasBase[Lattice]):
    """
    Affine group element acting on polynomial coordinate functions.

    This class combines a linear representation (`irrep`) with a translation
    (`offset`) and exposes multiple representations:
    - `full_rep`: Kronecker power of `irrep` for the full tensor product basis.
    - `rep`: Symmetrized representation on the commuting Euclidean monomial basis.
    - `affine_rep`: Homogeneous affine matrix in the physical basis of `offset.space`.

    Parameters
    ----------
    irrep : sy.ImmutableDenseMatrix
        Linear representation matrix acting on the coordinate axes.
    axes : Tuple[sy.Symbol, ...]
        Ordered symbols defining the coordinate axes (used to build monomials).
    offset : Offset
        Translation component with its associated lattice space.
    basis_function_order : int
        Polynomial order used to build the monomial basis (degree).
    """

    irrep: sy.ImmutableDenseMatrix
    axes: Tuple[sy.Symbol, ...]
    offset: Offset
    basis_function_order: int

    @lru_cache
    def __full_indices(self):
        return tuple(product(*((self.axes,) * self.basis_function_order)))

    @lru_cache
    def __commute_indices(self):
        indices = self.__full_indices()
        _, select_rules = AffineGroupElement.__get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @property
    @lru_cache
    def euclidean_basis(self) -> sy.ImmutableDenseMatrix:
        indices = self.__commute_indices()
        return sy.ImmutableDenseMatrix([sy.prod(idx) for idx in indices]).T

    @staticmethod
    @lru_cache
    def __get_contract_select_rules(indices: Tuple[Tuple[sy.Symbol, ...], ...]):
        commute_index_table: OrderedDict[Tuple[sy.Symbol, ...], int] = OrderedDict()
        contract_indices = []
        select_indices = []
        order_indices = set()
        order_idx = 0
        for n, idx in enumerate(indices):
            key = tuple(sorted(idx, key=lambda s: s.name))
            m = commute_index_table.setdefault(key, order_idx)

            contract_indices.append((n, m))
            if m not in order_indices:
                select_indices.append((n, m))
                order_indices.add(m)
                order_idx += 1

        return contract_indices, select_indices

    @property
    @lru_cache
    def full_rep(self):
        return reduce(sy.kronecker_product, (self.irrep,) * self.basis_function_order)

    @property
    @lru_cache
    def rep(self) -> sy.ImmutableDenseMatrix:
        indices = self.__full_indices()
        contract_indices, select_indices = self.__get_contract_select_rules(indices)

        contract_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in contract_indices:
            contract_matrix[i, j] = 1

        select_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in select_indices:
            select_matrix[i, j] = 1

        return select_matrix.T @ self.full_rep @ contract_matrix

    @property
    @lru_cache
    def affine_rep(self) -> sy.ImmutableDenseMatrix:
        """
        Use the lattice space of `offset` to build the affine transform matrix in
        physical (space-basis) coordinates.
        It will take the form of:
        ```
        [ R | t ]
        [ 0 | 1 ]
        ```
        where R and t are mapped into the `offset.space` basis via:
        `R = B * irrep * B^-1` and `t = B * offset.rep`, with `B = offset.space.basis`.
        """
        space = self.offset.space
        B = space.basis
        if not isinstance(B, sy.ImmutableDenseMatrix):
            B = sy.ImmutableDenseMatrix(B)
        B_inv = B.inv()

        R = self.irrep
        if not isinstance(R, sy.ImmutableDenseMatrix):
            R = sy.ImmutableDenseMatrix(R)
        R = B @ R @ B_inv

        t = self.offset.rep
        if not isinstance(t, sy.ImmutableDenseMatrix):
            t = sy.ImmutableDenseMatrix(t)
        t = B @ t

        top = R.row_join(t)
        bottom = sy.zeros(1, R.cols).row_join(sy.ones(1, 1))
        return sy.ImmutableDenseMatrix(top.col_join(bottom))

    @property
    @lru_cache
    def basis(self) -> FrozenDict:
        transform = self.rep
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            # principle term is the first non-zero term
            principle_term = next(x for x in vec if x != 0)

            rep = vec / principle_term
            expr = sy.simplify(rep.dot(self.euclidean_basis))
            tbl[v] = AbelianIrrep(
                expr=expr, axes=self.axes, order=self.basis_function_order, rep=rep
            )

        return FrozenDict(tbl)

    def base(self) -> Lattice:
        """Get the acting space of this affine group element."""
        return self.offset.space

    def rebase(self, new_base: Lattice) -> "AffineGroupElement":
        """
        Change the acting space of this affine group element to a new `Lattice`.
        """
        return AffineGroupElement(
            irrep=self.irrep,
            axes=self.axes,
            offset=self.offset.rebase(new_base),
            basis_function_order=self.basis_function_order,
        )

    def with_origin(self, origin: Offset) -> "AffineGroupElement":
        """
        Return an equivalent affine group element expressed relative to a new origin.

        Given the affine action in coordinate form:
            `x -> R x + t`
        shifting the origin by `o` (so x = x' + o) yields:
            `x' -> R x' + t'`
        with:
            `t' = t + (R - I) o`

        Parameters
        ----------
        `origin` : `Offset`
            The new origin expressed in an affine space. If it differs from this
            element's space, the element is rebased to `origin.space` first.

        Returns
        -------
        `AffineGroupElement`
            A new affine group element with the same linear part and adjusted
            translation so the action is expressed about `origin`.
        """
        if origin.space != self.offset.space:
            t = self.rebase(origin.space)
        else:
            t = self

        irrep = t.irrep
        if not isinstance(irrep, sy.ImmutableDenseMatrix):
            irrep = sy.ImmutableDenseMatrix(irrep)

        o_rep = origin.rep
        if not isinstance(o_rep, sy.ImmutableDenseMatrix):
            o_rep = sy.ImmutableDenseMatrix(o_rep)

        t_rep = t.offset.rep
        if not isinstance(t_rep, sy.ImmutableDenseMatrix):
            t_rep = sy.ImmutableDenseMatrix(t_rep)

        ident = sy.eye(irrep.rows)
        if not isinstance(ident, sy.ImmutableDenseMatrix):
            ident = sy.ImmutableDenseMatrix(ident)

        new_rep = t_rep + (irrep - ident) @ o_rep
        new_offset = Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=origin.space)

        return AffineGroupElement(
            irrep=irrep,
            axes=t.axes,
            offset=new_offset,
            basis_function_order=t.basis_function_order,
        )

    @lru_cache
    def group_elements(self, max_order: int = 128) -> Tuple["AffineGroupElement", ...]:
        """
        Generate the cyclic group elements produced by this irrep.

        Starts from the identity and repeatedly multiplies by this element,
        stopping once an element repeats or `max_order` is reached.
        """
        if max_order <= 0:
            return tuple()

        irrep = self.irrep
        axes = self.axes
        basis_order = self.basis_function_order

        current = sy.eye(irrep.rows)
        if not isinstance(current, sy.ImmutableDenseMatrix):
            current = sy.ImmutableDenseMatrix(current)

        elements = []
        seen = set()
        for _ in range(max_order):
            if current in seen:
                break
            seen.add(current)
            elements.append(
                AffineGroupElement(
                    irrep=current,
                    axes=axes,
                    offset=self.offset,
                    basis_function_order=basis_order,
                )
            )
            next_mat = current @ irrep
            if not isinstance(next_mat, sy.ImmutableDenseMatrix):
                next_mat = sy.ImmutableDenseMatrix(next_mat)
            current = next_mat

        return tuple(elements)

    def irreps(self) -> FrozenDict:
        """
        Build a lookup table of basis functions transformed by increasing powers.

        The method repeatedly constructs powers of this affine group element by
        varying `basis_function_order` from 1 upward and merges each generated
        `basis` mapping into a single dictionary. Iteration stops as soon as the
        table reaches the finite group order inferred from `group_elements()`.

        Returns
        -------
        `FrozenDict`
            Mapping from transformed basis functions to their
            associated irreducible representation values.
        """
        group_order = len(self.group_elements())
        tbl: Dict[sy.Expr, AbelianIrrep] = {}
        for n in range(group_order):
            order_element = AffineGroupElement(
                irrep=self.irrep,
                axes=self.axes,
                offset=self.offset,
                basis_function_order=n + 1,
            )
            tbl = {**order_element.basis, **tbl}

            if len(tbl) == group_order:
                break

        return FrozenDict(tbl)


@AffineGroupElement.register(GaugeInvariant)
def _affine_transform_gauge_invariant(
    t: AffineGroupElement, v: GaugeInvariant
) -> Gauged[GaugeInvariant, sy.Expr]:
    """Apply an affine group element to a gauge-invariant object."""
    return Gauged(gaugable=v, gauge=sy.Integer(1))


@AffineGroupElement.register(AbelianIrrep)
def _affine_transform_abelian_irrep(
    t: AffineGroupElement, f: AbelianIrrep
) -> Gauged[AbelianIrrep, sy.Expr]:
    """
    Apply an affine group element to a basis function and extract its phase factor.

    This treats the affine group element as a linear operator on the monomial basis
    of order `f.order`. The result of applying the representation to `f.rep`
    (the coefficient vector of `f` in that basis) must be a scalar multiple of the
    original vector for `f` to be an eigenfunction of the transform. When that holds,
    the scalar is the phase factor and we return it together with the original
    function `f` (the basis function itself does not change, only its phase).

    The procedure is as follows:
    - Compute `transformed_rep = t.rep @ f.rep`.
    - If `transformed_rep == phase * f.rep` for a single scalar `phase`, then
      `f` is a basis function for this transform and `phase` is returned.
    - If no such scalar exists (or the basis vector is zero), raise a ValueError.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element (transform) to apply.
    `f` : `AbelianIrrep`
        The basis function to be transformed.

    Returns
    -------
    `Gauged[AbelianIrrep, sy.Expr]`
        A named tuple containing:
        - `gauge`: The symbolic phase factor (sy.Expr) such that
          `t.rep @ f.rep == phase * f.rep`.
        - `gaugable`: The original `AbelianIrrep` (unchanged).

    Raises
    ------
    `ValueError`
        If the axes of `t` and `f` do not match, or if `f` is not an eigenfunction
        of the transform represented by `t`.
    """
    if set(t.axes) != set(f.axes):
        raise ValueError(
            f"Axes of AbelianGroup and PointGroupBasis must match: {t.axes} != {f.axes}"
        )

    if t.basis_function_order != f.order:
        t = AffineGroupElement(
            irrep=t.irrep,
            axes=t.axes,
            offset=t.offset,
            basis_function_order=f.order,
        )

    g_irrep = t.rep
    basis_rep = f.rep
    transformed_rep = g_irrep @ basis_rep

    phase = None
    for n in range(transformed_rep.rows):
        basis_term = basis_rep[n]
        transformed_term = transformed_rep[n]
        if basis_term != 0:
            if phase is None:
                phase = sy.simplify(transformed_term / basis_term)
            else:
                if sy.simplify(transformed_term - phase * basis_term) != 0:
                    raise ValueError(f"{f} is not a basis function!")
        else:
            if sy.simplify(transformed_term) != 0:
                raise ValueError(f"{f} is not a basis function!")

    if phase is None:
        raise ValueError(f"{f} is a trivial basis function: zero")

    return Gauged(gauge=phase, gaugable=f)


@AffineGroupElement.register(AbelianIrrepSet)
def _affine_transform_abelian_irrep_set(
    t: AffineGroupElement, s: AbelianIrrepSet
) -> Gauged[AbelianIrrepSet, sy.ImmutableDenseMatrix]:
    irrep_gauges = (
        cast(Gauged[AbelianIrrep, sy.Expr], t(irrep)).gauge for irrep in s.irreps
    )
    gauge: sy.ImmutableDenseMatrix = sy.ImmutableDenseMatrix.diag(*irrep_gauges)
    return Gauged(gauge=gauge, gaugable=s)


@AffineGroupElement.register(Offset)
def _affine_transform_offset(t: AffineGroupElement, offset: Offset) -> Offset:
    """
    Apply an affine group element to a spatial Offset using homogeneous coordinates.

    This implementation:
    - Ensures the transform acts in the same lattice space as the input Offset by
      rebasing the transform if necessary.
    - Uses the affine (homogeneous) representation of the transform to combine
      rotation/shear and translation in a single matrix multiply.
    - Preserves the input Offset's space in the returned result.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element to apply. If its internal `offset.space` does
        not match `offset.space`, the transform is rebased to the Offset's space.
    `offset` : `Offset`
        The spatial offset (column vector) to transform.

    Returns
    -------
    `Offset`
        A new `Offset` expressed in the same lattice as the input `offset`.

    Notes
    -----
    The method constructs a homogeneous coordinate vector:
    `[offset.rep; 1]`, multiplies by `t.affine_rep`, then discards the trailing
    homogeneous component. The result remains a column vector of shape `(dim, 1)`.
    """
    if offset.space != t.offset.space:
        t = t.rebase(offset.space)

    affine_rep = t.affine_rep
    rep = offset.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)

    hom = rep.col_join(sy.ones(1, 1))
    new_hom = affine_rep @ hom
    new_rep = new_hom[:-1, :]
    return Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=offset.space)


@AffineGroupElement.register(Momentum)
def _affine_transform_momentum(t: AffineGroupElement, k: Momentum) -> Momentum:
    """
    Apply an affine group element to a Momentum in fractional reciprocal coordinates.

    This implementation assumes:
    - `k.rep` stores fractional coordinates in the reciprocal lattice basis
      (values typically in [0, 1) per component).
    - The affine group's linear part is expressed in the *physical* real-space
      basis of `t.base()` via `t.affine_rep`.
    - Translations do not act on momenta, so only the linear part is used.

    Let:
    - `R_phys` be the real-space linear map in physical coordinates,
      i.e. the top-left block of `t.affine_rep`.
    - `G` be the reciprocal lattice basis (columns), `G = k.base().basis`.
      In this codebase `G = 2π * B^{-T}` for real-space basis `B`.
    - `k_frac` be the fractional reciprocal coordinates (`k.rep`).

    Then physical momentum is `k_phys = G * k_frac`, and it transforms as
    `k_phys' = (R_phys^{-1})^T * k_phys` (contravariant rule).
    Mapping back to fractional coordinates gives:

        k_frac' = G^{-1} * (R_phys^{-1})^T * G * k_frac

    The `2π` factor in `G` cancels with `G^{-1}`, so it does not appear explicitly.
    The output is wrapped with `.fractional()` to keep components in the first
    Brillouin zone.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        The affine group element to apply. If its base affine space does not
        match the real-space dual of `k`, it is rebased accordingly.
    `k` : `Momentum`
        The momentum expressed in fractional reciprocal coordinates of its
        reciprocal lattice basis.

    Returns
    -------
    `Momentum`
        The transformed momentum in the same reciprocal lattice space as `k`,
        wrapped into the first Brillouin zone via `.fractional()`.
    """
    real_space = k.base().dual
    if t.base() != real_space:
        t = t.rebase(real_space)

    linear_rep = t.affine_rep[:-1, :-1]
    if not isinstance(linear_rep, sy.ImmutableDenseMatrix):
        linear_rep = sy.ImmutableDenseMatrix(linear_rep)

    # Transform fractional reciprocal coordinates using
    # k_frac' = G^{-1} * (R_phys^{-1})^T * G * k_frac
    recip_basis = k.base().basis
    if not isinstance(recip_basis, sy.ImmutableDenseMatrix):
        recip_basis = sy.ImmutableDenseMatrix(recip_basis)
    recip_basis_inv = recip_basis.inv()
    reciprocal_rep = recip_basis_inv @ linear_rep.inv().T @ recip_basis

    rep = k.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)
    new_rep = reciprocal_rep @ rep
    return Momentum(rep=sy.ImmutableDenseMatrix(new_rep), space=k.base()).fractional()


@AffineGroupElement.register(Gaugable, order="back")
def _affine_transform_gaugable(
    t: AffineGroupElement, v: Gaugable
) -> Gauged[Gaugable, sy.Expr | sy.ImmutableDenseMatrix]:
    """Transform a gaugable object by updating its gauge phase.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry operation applied to the gauge representation of `v`.
        The operation is evaluated through `t.apply(...)`, and only its
        gauge phase contribution is used by this method.
    `v` : `Gaugable`
        Object that provides a gauge representation via `.gauge_repr()`. The
        original value is preserved and wrapped in a `Gauged` container.

    Returns
    -------
    `Gauged[Gaugable, sy.Expr | sy.ImmutableDenseMatrix]`
        A gauged wrapper whose `gaugable` field is the original input `v` and
        whose `gauge` field is the gauge returned by applying `t` to `v`'s gauge
        representation.
    """
    basis = v.gauge_repr()
    gauge, _ = t.apply(basis)
    return Gauged(gaugable=v, gauge=gauge)


def _optional_transform_mode_attr(t: AffineGroupElement, v: Mode):
    """Transform a Mode attribute if the transform allows it."""
    if not t.allows(v):
        return v
    return t.apply(v)


@AffineGroupElement.register(Mode, order="front")
def _affine_transform_mode(t: AffineGroupElement, m: Mode) -> Mode:
    """Apply an affine transformation to each transformable attribute of a mode.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine transformation to apply. For each attribute in `m`, this
        function checks `t.allows(attr)` and applies `t.transform(attr)` only
        when that attribute is supported by the transform.
    `m` : `Mode`
        Input mode whose named attributes are visited and conditionally
        transformed.

    Returns
    -------
    `Mode`
        A mode instance with the same attribute structure as `m`, where each
        attribute is transformed by `t` if allowed, and left unchanged
        otherwise.
    """
    attr_names = m.attr_names()
    func = partial(_optional_transform_mode_attr, t)
    apply = {name: func for name in attr_names}
    return m.update(**apply)


@AffineGroupElement.register(HilbertSpace)
def _affine_transform_hilbert(t: AffineGroupElement, h: HilbertSpace) -> Tensor:
    """
    Apply an affine transformation to a Hilbert-space basis and build the
    corresponding basis-change matrix.

    This routine transforms each mode in `h` under `t`, collects:
    - the transformed mode mapping `m -> g(m)`, and
    - the associated gauge factor for each mapped pair.

    The result is returned as a `Tensor` from the original basis to the
    transformed basis via `mapping_matrix`.

    Gauge handling:
    - Scalar gauges are converted to Python `complex`.
    - Matrix gauges are assumed diagonal in the mode basis and are converted
      to a diagonal `torch.Tensor`.
    - Symbolic (non-numeric) gauges are rejected.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry element used to transform each mode in `h`. The mode
        transform is expected to produce a `Gauged[Mode, ...]` object carrying
        both transformed mode and gauge factor.
    `h` : `HilbertSpace`
        Input Hilbert space whose modes define the source basis.

    Returns
    -------
    `Tensor`
        Basis-change tensor representing the action of `t` on `h`, with source
        dimension `h` and target dimension `gh = hilbert({t(m)})` (up to gauge
        factors).

    Raises
    ------
    `ValueError`
        If a collected gauge factor remains symbolic (has free symbols), since
        a numeric mapping matrix is required.
    """
    mode_mapping: Dict[Mode, Mode] = {}
    gauge_table: Dict[Tuple[Mode, Mode], complex | torch.Tensor] = {}
    for m in h:
        gauged = cast(Gauged[Mode, sy.Expr | sy.ImmutableDenseMatrix], t(m))
        gm = cast(Mode, gauged.gaugable)
        gauge = gauged.gauge
        if gauge.free_symbols:
            raise ValueError(f"Gauge factor must be numeric, got symbolic {gauge}")
        mode_mapping[m] = gm
        if isinstance(gauge, sy.ImmutableDenseMatrix):
            diag_vals = [
                complex(gauge[i, i]) for i in range(min(gauge.rows, gauge.cols))
            ]
            gauge_table[(m, gm)] = torch.diag(torch.tensor(diag_vals))
        else:
            gauge_table[(m, gm)] = complex(gauge)

    gh = hilbert(mode_mapping.values())
    return mapping_matrix(h, gh, mode_mapping, factors=gauge_table)  # (h, gh)


def bandtransform(
    t: AffineGroupElement,
    tensor: Tensor,
    opt: Literal["left", "right", "both"] = "both",
) -> Tensor:
    """
    Apply an affine symmetry action to a momentum-resolved operator tensor.

    The expected tensor shape is `(K, B_left, B_right)` where `K` is a
    `MomentumSpace` and `B_left`, `B_right` are `HilbertSpace`s. Depending on
    `opt`, this function applies the symmetry-induced basis transform on the
    left side, right side, or both sides of the band tensor.

    For each transformed side, a k-dependent matrix is built from:
    - the affine action on the Hilbert space basis (`t(space)`), and
    - Fourier transforms that connect Bloch and real-space sectors.

    Momentum handling:
    - The k action is treated as a relabeling/permutation of sectors.
    - We align the k-axis of the transform tensors to the canonical `kspace`
      ordering before multiplication.
    - The input tensor itself is not pre-remapped in k; remapping is used only
      to align transform blocks with each momentum sector.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine transformation to apply.
    `tensor` : `Tensor`
        Momentum-space tensor with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`.
    `opt` : `Literal["left", "right", "both"]`, default `"both"`
        Which side(s) to transform.

    Returns
    -------
    `Tensor`
        The transformed tensor with the same dimension types.

    Raises
    ------
    `ValueError`
        If `opt` is invalid, if `tensor` is not rank-3 with dims
        `(MomentumSpace, HilbertSpace, HilbertSpace)`, or if a Hilbert space
        side is not symmetry-compatible with `t`.
    """
    if opt not in ("both", "left", "right"):
        raise ValueError(f"Invalid option {opt} for bandtransform!")
    if not len(tensor.dims) == 3:
        raise ValueError("Input tensor must have exactly 3 dimensions.")
    if not isinstance(tensor.dims[0], MomentumSpace):
        raise ValueError("First dimension of tensor must be a MomentumSpace.")
    if not isinstance(tensor.dims[1], HilbertSpace):
        raise ValueError("Second dimension of tensor must be a HilbertSpace.")
    if not isinstance(tensor.dims[2], HilbertSpace):
        raise ValueError("Third dimension of tensor must be a HilbertSpace.")

    kspace: MomentumSpace = cast(MomentumSpace, tensor.dims[0])

    def build_transform(space: HilbertSpace) -> Tensor:
        bloch_transform: Tensor = cast(Tensor, t(space)).h(-2, -1)  # (B', B)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        bloch_transform = bloch_transform.replace_dim(
            0, cast(HilbertSpace, bloch_transform.dims[0]).update(r=Offset.fractional)
        )  # (B'=B, B)
        gspace = cast(HilbertSpace, bloch_transform.dims[0])
        if not space.same_span(gspace):
            raise ValueError(
                f"Hilbert space {space} is not symmetric under the transform {t}!"
            )
        left_fourier = fourier_transform(kspace, space, gspace)  # (K, B, B'=B)
        right_fourier = fourier_transform(kspace, space, space)  # (K, B, B)
        # (K, B, B'=B) @ (B'=B, B) @ (B, B)
        transform = (
            left_fourier @ bloch_transform @ right_fourier.h(-2, -1)
        )  # (K, B, B)
        return transform

    mapped_kspace = kspace.map(lambda k: cast(Momentum, t(k)).fractional())

    if opt in ("both", "left"):
        left_fourier = build_transform(cast(HilbertSpace, tensor.dims[1]))  # (K, B, B)
        left_fourier = left_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (left_fourier @ tensor))  # (K, B, B)

    if opt in ("both", "right"):
        right_fourier = build_transform(cast(HilbertSpace, tensor.dims[2]))  # (K, B, B)
        right_fourier = right_fourier.replace_dim(0, mapped_kspace).align(
            0, kspace
        )  # (K, B, B)
        tensor = cast(Tensor, (tensor @ right_fourier.h(-2, -1)))  # (K, B, B)

    return tensor


_AFFINE_QUERY_RE = re.compile(
    r"^(?P<group>c\d+|m)-(?P<ambient>[xyz]+):(?P<target>[xyz]+)-o(?P<order>\d+)$"
)


def _parse_affine_query(query: str):
    match = _AFFINE_QUERY_RE.fullmatch(query.strip())
    if match is None:
        raise ValueError(
            "Invalid query format. Expected '<group>-<ambient>:<target>-o<order>', "
            "for example 'c3-xy:xy-o2'."
        )

    group = match.group("group")
    ambient = match.group("ambient")
    target = match.group("target")
    order = int(match.group("order"))

    if order <= 0:
        raise ValueError("Basis function order must be a positive integer.")

    if len(set(ambient)) != len(ambient):
        raise ValueError(f"Ambient axes must be unique, got '{ambient}'.")
    if len(set(target)) != len(target):
        raise ValueError(f"Target axes must be unique, got '{target}'.")
    if not set(target).issubset(set(ambient)):
        raise ValueError(
            f"Target axes '{target}' must be a subset of ambient axes '{ambient}'."
        )

    return group, ambient, target, order


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
                "In 1D, mirror target must match ambient axis (e.g. 'm-x:x-o1')."
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


def pointgroup(query: str) -> AffineGroupElement:
    """
    Build an `AffineGroupElement` from a compact query string.

    This is a user-facing constructor for common point operations in Cartesian
    axes (`x`, `y`, `z`), currently supporting cyclic rotations and mirrors.
    Only these two group families are implemented at present; other group types
    are not yet supported and will raise `ValueError`.

    Query grammar
    -------------
    The accepted format is:

    `"<group>-<ambient>:<target>-o<order>"`

    where:
    - `<group>` is:
      - `c{n}` for cyclic rotation of order `n` (e.g. `c2`, `c3`, `c6`, ...),
      - `m` for mirror reflection.
    - `<ambient>` is the ordered ambient axis string (`x`, `y`, `z` without repeats),
      defining the space dimension and basis-axis order in the returned transform.
      Examples: `x`, `xy`, `xyz`, `yzx`.
    - `<target>` chooses where the group action lives (must be subset of ambient).
    - `<order>` is the polynomial basis-function order for `AffineGroupElement`.

    Group semantics
    ---------------
    Cyclic (`c{n}`)
    - Always interpreted as a 2D rotation block with angle `2*pi/n`.
    - `<target>` must have exactly 2 axes, defining the rotation plane.
    - In 2D ambient, `<target>` must use the same two axes as ambient.
      Example: `c3-xy:xy-o2` is valid, `c3-xy:xz-o2` is invalid.
    - Target order controls orientation:
      - `c3-xy:xy-o2` and `c3-xy:yx-o2` act on the same plane,
      - the second is the inverse orientation of the first.
    - In 3D ambient, the remaining axis is unchanged.
      Example: `c6-xyz:yz-o2` rotates in the `yz` plane, keeps `x` fixed.

    Mirror (`m`)
    - 1D (`ambient` length 1):
      - `<target>` must match ambient axis,
      - action is sign flip in 1D (`x -> -x`).
    - 2D (`ambient` length 2):
      - `<target>` must have exactly 1 axis and denotes the fixed axis.
      - Example: `m-xy:y-o1` mirrors about the y-axis (`x -> -x`, `y -> y`).
    - 3D (`ambient` length 3):
      - `<target>` must have exactly 2 axes and denotes the fixed plane.
      - Example: `m-xyz:yz-o1` mirrors about the yz-plane (`x -> -x`).

    Validation rules
    ----------------
    - `ambient` and `target` cannot contain repeated axis letters.
    - `target` must be a subset of `ambient`.
    - `order` must be a positive integer.
    - Invalid dimensional/group combinations raise `ValueError`.

    Return value
    ------------
    Returns an `AffineGroupElement` with:
    - `irrep`: the linear matrix representation from query semantics,
    - `axes`: symbols in ambient order,
    - `offset`: zero offset in the identity lattice,
    - `basis_function_order`: parsed `<order>`.

    Examples
    --------
    Cyclic, 2D:
    - `pointgroup("c6-xy:xy-o2")`:
      60-degree rotation in `xy`.
    - `pointgroup("c6-xy:yx-o2")`:
      inverse orientation of the same `c6`.

    Cyclic, 3D:
    - `pointgroup("c6-xyz:yz-o2")`:
      rotate `yz` by 60 degrees, keep `x` fixed.

    Mirror, 1D:
    - `pointgroup("m-x:x-o1")`:
      reflection in 1D (`x -> -x`).

    Mirror, 2D:
    - `pointgroup("m-xy:y-o2")`:
      mirror about y-axis (`x -> -x`, `y -> y`).
    - `pointgroup("m-xy:x-o2")`:
      mirror about x-axis (`x -> x`, `y -> -y`).

    Mirror, 3D:
    - `pointgroup("m-xyz:yz-o1")`:
      mirror about yz-plane (`x -> -x`).
    - `pointgroup("m-xyz:xz-o1")`:
      mirror about xz-plane (`y -> -y`).
    - `pointgroup("m-xyz:xy-o1")`:
      mirror about xy-plane (`z -> -z`).
    """
    group, ambient, target, basis_order = _parse_affine_query(query)

    axes_symbols = {
        "x": sy.Symbol("x"),
        "y": sy.Symbol("y"),
        "z": sy.Symbol("z"),
    }
    axes = tuple(axes_symbols[c] for c in ambient)
    dim = len(axes)

    if group.startswith("c"):
        n = int(group[1:])
        irrep = _build_cyclic_irrep(n=n, ambient=ambient, target=target)
    elif group == "m":
        irrep = _build_mirror_irrep(ambient=ambient, target=target)
    else:
        raise ValueError(
            f"Unsupported group '{group}'. Supported groups are cyclic and mirror."
        )

    space = Lattice(
        basis=sy.ImmutableDenseMatrix.eye(dim),
        boundaries=PeriodicBoundary(sy.ImmutableDenseMatrix.eye(dim)),
        unit_cell={"r": sy.ImmutableDenseMatrix([0] * dim)},
    )
    zero = Offset(rep=sy.ImmutableDenseMatrix([0] * dim), space=space)
    return AffineGroupElement(
        irrep=irrep,
        axes=axes,
        offset=zero,
        basis_function_order=basis_order,
    )
