from dataclasses import dataclass
import re
from typing import Any, Dict, Literal, Tuple, cast
from collections import OrderedDict
from itertools import product
from functools import lru_cache, reduce
from multipledispatch import dispatch  # type: ignore[import-untyped]

import sympy as sy

from .abstracts import HasBase
from .spatials import AffineSpace, Spatial, Offset, Momentum
from .state_space import MomentumSpace, same_span
from .hilbert_space import (
    U1Operator,
    HilbertSpace,
    U1Basis,
    U1Span,
    FuncOpr,
    hilbert,
)
from .tensors import Tensor
from .fourier import fourier_transform
from .validations import need_validation
from .validations.symbolics import check_invertibility, check_numerical
from .utils import FrozenDict


@dataclass(frozen=True)
class AbelianBasis(Spatial):
    """
    Symbolic abelian eigen-basis expressed in a polynomial basis over given axes.

    Ordering
    --------
    `AbelianBasis` comparison (`<`, `>`) is defined by lexicographic string
    ordering of `expr` (`str(expr)`).

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
        return f"AbelianBasis({str(self.expr)})"

    def __repr__(self):
        return f"AbelianBasis({repr(self.expr)})"


@dispatch(AbelianBasis, AbelianBasis)
def operator_lt(a: AbelianBasis, b: AbelianBasis) -> bool:
    return str(a.expr) < str(b.expr)


@dispatch(AbelianBasis, AbelianBasis)
def operator_gt(a: AbelianBasis, b: AbelianBasis) -> bool:
    return str(a.expr) > str(b.expr)


@need_validation(check_invertibility("irrep"), check_numerical("irrep"))
@dataclass(frozen=True)
class AffineGroupElement(U1Operator, HasBase[AffineSpace]):
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
        """
        Enumerate all ordered monomial indices for the tensor-product basis.

        Returns
        -------
        `Tuple[Tuple[sy.Symbol, ...], ...]`
            Cartesian product of `axes` repeated `basis_function_order` times.
            Each inner tuple represents one ordered monomial index before
            commutative contraction.
        """
        return tuple(product(*((self.axes,) * self.basis_function_order)))

    @lru_cache
    def __commute_indices(self):
        """
        Build canonical monomial indices under symbol commutation.

        Returns
        -------
        `Tuple[Tuple[sy.Symbol, ...], ...]`
            Ordered subset of `__full_indices()` where permutations that differ
            only by factor ordering are collapsed to a single representative.
        """
        indices = self.__full_indices()
        _, select_rules = AffineGroupElement.__get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @property
    @lru_cache
    def euclidean_basis(self) -> sy.ImmutableDenseMatrix:
        """
        Return commuting Euclidean monomials spanning the polynomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix`
            Row matrix whose entries are monomials formed from canonical
            commuting indices of degree `basis_function_order`.
        """
        indices = self.__commute_indices()
        return sy.ImmutableDenseMatrix([sy.prod(idx) for idx in indices]).T

    @staticmethod
    @lru_cache
    def __get_contract_select_rules(indices: Tuple[Tuple[sy.Symbol, ...], ...]):
        """
        Compute contraction and selection rules for commutative symmetrization.

        Parameters
        ----------
        `indices` : `Tuple[Tuple[sy.Symbol, ...], ...]`
            Full ordered tensor-product indices.

        Returns
        -------
        `Tuple[list[Tuple[int, int]], list[Tuple[int, int]]]`
            Two index maps:
            - contract rules mapping each full index position to a commutative class
            - select rules picking one representative position per class
        """
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
        """
        Representation on the full ordered tensor-product monomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix | sy.MatrixBase`
            Kronecker power `irrep ⊗ ... ⊗ irrep` with
            `basis_function_order` factors.
        """
        return reduce(sy.kronecker_product, (self.irrep,) * self.basis_function_order)

    @property
    @lru_cache
    def rep(self) -> sy.ImmutableDenseMatrix:
        """
        Symmetrized representation on the commuting polynomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix`
            Matrix representation after contracting permutation-equivalent
            tensor-product monomials and selecting canonical representatives.
        """
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
        """
        Compute affine eigen-basis functions from `rep` eigenvectors.

        Returns
        -------
        `FrozenDict`
            Mapping from eigenvalue to normalized `AbelianBasis` eigenfunction.
            Normalization is fixed by dividing by the first non-zero coefficient
            in each eigenvector.
        """
        transform = self.rep
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            # principle term is the first non-zero term
            principle_term = next(x for x in vec if x != 0)

            rep = vec / principle_term
            expr = sy.simplify(rep.dot(self.euclidean_basis))
            tbl[v] = AbelianBasis(
                expr=expr, axes=self.axes, order=self.basis_function_order, rep=rep
            )

        return FrozenDict(tbl)

    def base(self) -> AffineSpace:
        """
        Get the affine space where this element acts.

        Returns
        -------
        `AffineSpace`
            Acting space, identical to `offset.space`.
        """
        return self.offset.space

    def rebase(self, new_base: AffineSpace) -> "AffineGroupElement":
        """
        Re-express this transform in a different affine space basis.

        Parameters
        ----------
        `new_base` : `AffineSpace`
            Target affine space for the transformed representation.

        Returns
        -------
        `AffineGroupElement`
            New element with the same symbolic linear representation and axes,
            but with `offset` rebased to `new_base`.
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
        Generate powers of the linear irrep with a shared translation component.

        Starts from the identity matrix and repeatedly multiplies by `self.irrep`,
        stopping once the linear matrix repeats or `max_order` is reached.
        Each returned `AffineGroupElement` uses:
        - the current linear power (`I, R, R^2, ...`) as `irrep`,
        - the same `axes` and `basis_function_order`,
        - the original `self.offset` unchanged.

        Note that this method does not compose affine translations across powers;
        cycle detection is performed only on the linear part.

        Parameters
        ----------
        `max_order` : `int`
            Maximum number of powers to generate.

        Returns
        -------
        `Tuple[AffineGroupElement, ...]`
            Sequence of elements whose linear parts are `[I, R, R^2, ...]`,
            truncated at linear cycle closure or `max_order`.
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
            Mapping from irrep eigenvalues (`sy.Expr`) to corresponding
            `AbelianBasis` eigenfunctions.
        """
        group_order = len(self.group_elements())
        tbl: Dict[sy.Expr, AbelianBasis] = {}
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


@AffineGroupElement.register(AbelianBasis)
def _affine_transform_abelian_basis(
    t: AffineGroupElement, f: AbelianBasis
) -> Tuple[sy.Expr, AbelianBasis]:
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
    `f` : `AbelianBasis`
        The basis function to be transformed.

    Returns
    -------
    `Tuple[sy.Expr, AbelianBasis]`
        A symbolic phase factor (sy.Expr) such that `t.rep @ f.rep == phase * f.rep`;
        and the original `AbelianBasis` (unchanged).

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

    return phase, f


@AffineGroupElement.register(Offset)
def _affine_transform_offset(
    t: AffineGroupElement, offset: Offset
) -> Tuple[sy.Expr | None, Offset]:
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
    `Tuple[sy.Expr | None, Offset]`
        The irrep of this transformation, `None` if the `offset` is not a fix point; and new `Offset`
        expressed in the same `AffineSpace` as the input `offset`.

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
    new_offset = Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=offset.space)
    irrep = sy.Integer(1) if new_offset == offset else None
    return irrep, new_offset


@AffineGroupElement.register(Momentum)
def _affine_transform_momentum(
    t: AffineGroupElement, k: Momentum
) -> Tuple[sy.Expr | None, Momentum]:
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
    `Tuple[sy.Expr | None, Momentum]`
        The irrep of this transformation, `None` if `k` is not a fix point;
        and the transformed momentum in the same reciprocal lattice space as `k`,
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
    new_k = Momentum(rep=sy.ImmutableDenseMatrix(new_rep), space=k.base()).fractional()
    irrep = sy.Integer(1) if new_k == k else None
    return irrep, new_k


@AffineGroupElement.register(U1Basis)
def _affine_transform_u1_state(
    t: AffineGroupElement, psi: U1Basis
) -> Tuple[sy.Expr | None, U1Basis]:
    """
    Apply an affine transform to a `U1Basis` by transforming each irrep component.

    For each irrep in `psi.rep`, this method applies `t` when compatible
    (`t.allows(irrep)`), multiplies returned gauge factors, and rebuilds the state
    from transformed irreps. The state's U(1) label is
    updated by the same accumulated gauge product when well-defined.

    If any irrep transform reports `None` gauge, the overall state gauge is set to
    `None` (non-eigenstate under this symmetry), but irrep transformations are still
    applied and returned.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry operation.
    `psi` : `U1Basis`
        State to transform.

    Returns
    -------
    `Tuple[sy.Expr | None, U1Basis]`
        Overall gauge (or `None` if not a symmetry eigenstate) and transformed
        `U1Basis`.
    """
    overall_gauge: sy.Expr | None = psi.u1
    new_u1: sy.Expr = psi.u1
    new_rep: Tuple[Any, ...] = tuple()
    for irrep in psi.rep:
        if t.allows(irrep):
            gauge, transformed_irrep = t(irrep)
        else:
            gauge, transformed_irrep = sy.Integer(1), irrep
        if gauge is None:
            overall_gauge = None
        elif overall_gauge is not None:
            overall_gauge *= gauge
            new_u1 *= gauge
        new_rep += (transformed_irrep,)
    return overall_gauge, U1Basis(new_u1, new_rep)


@AffineGroupElement.register(U1Span)
def _affine_transform_u1_span(
    t: AffineGroupElement, v: U1Span
) -> Tuple[sy.ImmutableDenseMatrix | None, U1Span]:
    """
    Apply an affine transform to a span of `U1Basis`s and extract its matrix irrep.

    Each basis state in `v.span` is transformed independently. The transformed
    span is then compared against the original span using `same_span`.
    - If the span is not invariant, returns `(None, transformed_span)`.
    - If invariant, returns the Gram/overlap matrix `v.gram(new_v)`, which is the
      representation of the symmetry action in this basis.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry operation.
    `v` : `U1Span`
        Span to transform.

    Returns
    -------
    `Tuple[sy.ImmutableDenseMatrix | None, U1Span]`
        Matrix-valued irrep on the span basis when invariant, otherwise `None`,
        together with the transformed span.
    """
    new_span: Tuple[U1Basis, ...] = tuple(t @ psi for psi in v.span)
    new_v = U1Span(new_span)
    if not same_span(v, new_v):
        return None, new_v
    irrep = v.gram(new_v)
    return irrep, new_v


@AffineGroupElement.register(HilbertSpace)
def _affine_transform_hilbert(
    t: AffineGroupElement, h: HilbertSpace
) -> Tuple[Tensor | None, HilbertSpace]:
    """
    Apply an affine transform to a `HilbertSpace` basis and compute its action.

    The transform is applied elementwise to build a new Hilbert space basis.
    Invariance is checked by `same_span(h, new_h)`:
    - If the transformed basis leaves the span, returns `(None, new_h)`.
    - If the span is preserved, returns the basis-action matrix `h.gram(new_h)`.

    Parameters
    ----------
    `t` : `AffineGroupElement`
        Affine symmetry operation.
    `h` : `HilbertSpace`
        Hilbert-space basis to transform.

    Returns
    -------
    `Tuple[Tensor | None, HilbertSpace]`
        Tensor representation of the symmetry action in the original basis when
        invariant, otherwise `None`; and the transformed Hilbert space.
    """
    new_h = hilbert(t @ el for el in h)
    if not same_span(h, new_h):
        return None, new_h
    gram = h.gram(new_h)
    return gram, new_h


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
        fractional = FuncOpr(Offset, Offset.fractional)
        new_space = fractional @ (t @ space)
        bloch_transform: Tensor = cast(Tensor, space.gram(new_space)).h(
            -2, -1
        )  # (B', B)
        # The transformation will distort the unit-cell of the Hilbert space,
        # we will use fractional to return it to the original unit-cell.
        if not space.same_span(new_space):
            raise ValueError(
                f"Hilbert space {space} is not symmetric under the transform {t}!"
            )
        left_fourier = fourier_transform(kspace, space, space)  # (K, B, B'=B)
        right_fourier = fourier_transform(kspace, space, space)  # (K, B, B)
        # (K, B, B'=B) @ (B'=B, B) @ (B, B)
        transform = (
            left_fourier @ bloch_transform @ right_fourier.h(-2, -1)
        )  # (K, B, B)
        return transform

    mapped_kspace = kspace.map(lambda k: cast(Momentum, t @ k).fractional())

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

    space = AffineSpace(basis=sy.ImmutableDenseMatrix.eye(dim))
    zero = Offset(rep=sy.ImmutableDenseMatrix([0] * dim), space=space)
    return AffineGroupElement(
        irrep=irrep,
        axes=axes,
        offset=zero,
        basis_function_order=basis_order,
    )
