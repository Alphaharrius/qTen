from dataclasses import dataclass
from typing import Dict, Tuple
from collections import OrderedDict
from itertools import product
from functools import lru_cache, reduce
from multipledispatch import dispatch  # type: ignore[import-untyped]

import sympy as sy

from ..abstracts import HasBase
from ..geometries.spatials import AffineSpace, Spatial, Offset, Momentum
from ..symbolics.hilbert_space import Opr
from ..validations import need_validation
from ..validations.symbolics import check_invertibility, check_numerical
from ..utils.collections_ext import FrozenDict
from ..symbolics import Multiple


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
        if sy.simplify(self.expr - 1) == 0:
            return "e"
        return str(self.expr)

    def __repr__(self):
        return self.__str__()


@dispatch(AbelianBasis, AbelianBasis)
def operator_lt(a: AbelianBasis, b: AbelianBasis) -> bool:
    return str(a.expr) < str(b.expr)


@dispatch(AbelianBasis, AbelianBasis)
def operator_gt(a: AbelianBasis, b: AbelianBasis) -> bool:
    return str(a.expr) > str(b.expr)


@need_validation(check_invertibility("irrep"), check_numerical("irrep"))
@dataclass(frozen=True)
class AbelianGroup(Opr):
    """
    Abelian linear operator represented on Cartesian coordinate functions.

    `AbelianGroup` stores the linear part `g` of a symmetry/operator as an
    exact matrix `irrep` acting on the coordinate axes `axes`. It provides the
    order-dependent polynomial representations induced by that linear action
    and the corresponding eigen-basis functions (`AbelianBasis`).

    Mathematical meaning
    --------------------
    Let the coordinate vector be `x = (x_1, ..., x_d)^T`. The matrix `irrep`
    defines a linear action

    `x -> irrep * x`.

    From this degree-1 action, the class constructs higher-order polynomial
    representations on homogeneous monomials of total degree `order`. For
    example:

    - `order = 0` acts on constant functions and is always the trivial `1x1`
      representation `[1]`
    - `order = 1` is the original Euclidean representation `irrep`
    - `order = 2` acts on quadratic monomials such as `x^2`, `xy`, `y^2`

    Because coordinate symbols commute, the raw tensor-product representation is
    symmetrized onto the commuting monomial basis. The resulting matrix is
    returned by `euclidean_repr(order)`.

    Parameters
    ----------
    `irrep` : `sy.ImmutableDenseMatrix`
        Exact linear representation matrix of the operator in the coordinate
        basis defined by `axes`.
    `axes` : `Tuple[sy.Symbol, ...]`
        Ordered coordinate symbols on which `irrep` acts.

    Main API
    --------
    - `euclidean_repr(order)`
      Symmetrized linear action on homogeneous commuting monomials of degree
      `order`.
    - `basis(order)`
      Eigen-basis functions of `euclidean_repr(order)` returned as
      `AbelianBasis` objects keyed by eigenvalue.
    - `basis_table`
      Aggregate lookup table of eigen-basis functions collected across
      increasing polynomial orders until all characters/eigenvalues of the
      finite represented element are found.
    - `group_order(max_order=128)`
      Order of the represented matrix, i.e. the smallest positive integer `n`
      such that `irrep**n = I`.

    Notes
    -----
    `AbelianGroup` is the linear object. To obtain an affine operator of the
    form `x -> g x + t`, wrap it in `AbelianOpr`. In that sense, `AbelianOpr`
    is the affine extension of `AbelianGroup`.

    The `group_order()` and `basis_table` utilities assume the represented
    element has finite order. They are appropriate for finite abelian point
    symmetries, but may fail or be incomplete for infinite-order linear maps.
    """

    irrep: sy.ImmutableDenseMatrix
    axes: Tuple[sy.Symbol, ...]

    @lru_cache
    def _full_indices(self, order: int):
        """
        Enumerate all ordered monomial indices for the tensor-product basis.

        Returns
        -------
        `Tuple[Tuple[sy.Symbol, ...], ...]`
            Cartesian product of `axes` repeated `order` times.
            Each inner tuple represents one ordered monomial index before
            commutative contraction.
        """
        if order == 0:
            return ((),)
        return tuple(product(*((self.axes,) * order)))

    @lru_cache
    def _commute_indices(self, order: int):
        """
        Build canonical monomial indices under symbol commutation.

        Returns
        -------
        `Tuple[Tuple[sy.Symbol, ...], ...]`
            Ordered subset of `_full_indices(order)` where permutations that differ
            only by factor ordering are collapsed to a single representative.
        """
        indices = self._full_indices(order)
        _, select_rules = AbelianGroup._get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @lru_cache
    def _euclidean_basis(self, order: int) -> sy.ImmutableDenseMatrix:
        """
        Return commuting Euclidean monomials spanning the polynomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix`
            Row matrix whose entries are monomials formed from canonical
            commuting indices of degree `order`.
        """
        indices = self._commute_indices(order)
        return sy.ImmutableDenseMatrix([sy.prod(idx) for idx in indices]).T

    @staticmethod
    @lru_cache
    def _get_contract_select_rules(indices: Tuple[Tuple[sy.Symbol, ...], ...]):
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

    @lru_cache
    def _raw_euclidean_repr(self, order: int) -> sy.ImmutableDenseMatrix:
        """
        Representation on the raw ordered tensor-product monomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix | sy.MatrixBase`
            Kronecker power `irrep ⊗ ... ⊗ irrep` with `order` factors.
        """
        if order == 0:
            return sy.ImmutableDenseMatrix([[1]])
        return reduce(sy.kronecker_product, (self.irrep,) * order)

    @lru_cache
    def euclidean_repr(self, order: int) -> sy.ImmutableDenseMatrix:
        """
        Symmetrized representation on the commuting polynomial basis.

        Returns
        -------
        `sy.ImmutableDenseMatrix`
            Matrix representation after contracting permutation-equivalent
            tensor-product monomials and selecting canonical representatives.
        """
        indices = self._full_indices(order)
        contract_indices, select_indices = self._get_contract_select_rules(indices)

        contract_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in contract_indices:
            contract_matrix[i, j] = 1

        select_matrix = sy.zeros(len(indices), len(select_indices))
        for i, j in select_indices:
            select_matrix[i, j] = 1

        return select_matrix.T @ self._raw_euclidean_repr(order) @ contract_matrix

    @lru_cache
    def group_order(self, max_order: int = 128) -> int:
        """
        Return the order of this represented group element.

        The order is the smallest positive integer `n` such that
        `irrep**n = I`, where `I` is the identity matrix of matching size.

        Returns
        -------
        `int`
            The smallest positive exponent for which the represented matrix
            returns to the identity.

        Raises
        ------
        `ValueError`
            If no finite order is found within the bounded exact search.

        Notes
        -----
        This computes the order of the matrix image under the representation.
        For a faithful representation, this equals the abstract group-element
        order; otherwise it may be smaller.
        """
        ident = sy.ImmutableDenseMatrix.eye(self.irrep.rows)
        power = ident
        for n in range(1, max_order + 1):
            power = sy.ImmutableDenseMatrix(sy.simplify(power @ self.irrep))
            if power.equals(ident):
                return n
        raise ValueError(
            f"Failed to determine a finite group order within max_order={max_order} "
            f"for irrep={self.irrep!r}."
        )

    @lru_cache
    def basis(self, order: int) -> FrozenDict:
        """
        Compute abelian eigen-basis functions from `euclidean_repr(order)` eigenvectors.

        Returns
        -------
        `FrozenDict`
            Mapping from eigenvalue to normalized `AbelianBasis` eigenfunction.
            Normalization is fixed by dividing by the first non-zero coefficient
            in each eigenvector.
        """
        transform = self.euclidean_repr(order)
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            # principle term is the first non-zero term
            principle_term = next(x for x in vec if x != 0)

            rep = vec / principle_term
            expr = sy.simplify(rep.dot(self._euclidean_basis(order)))
            tbl[v] = AbelianBasis(expr=expr, axes=self.axes, order=order, rep=rep)

        return FrozenDict(tbl)

    @property
    @lru_cache
    def basis_table(self) -> FrozenDict:
        g_order = self.group_order()
        tbl: Dict[sy.Expr, AbelianBasis] = {}
        for order in range(g_order):
            tbl = {**self.basis(order), **tbl}
            if len(tbl) == g_order:
                return FrozenDict(tbl)
        raise ValueError(
            f"Failed to build a complete basis table up to order {g_order - 1}."
        )


@AbelianGroup.register(AbelianBasis)
def _(g: AbelianGroup, f: AbelianBasis) -> Multiple[AbelianBasis]:
    """
    Apply an abelian-group element to an abelian eigen-basis function.

    The action is computed in the Euclidean monomial basis of degree
    `f.order`. If `f` is an eigenfunction of the represented group element,
    then `g.euclidean_repr(f.order) @ f.rep` must be a scalar multiple of
    `f.rep`. That scalar is returned as the phase/character factor while the
    basis function itself remains unchanged.

    Parameters
    ----------
    `g` : `AbelianGroup`
        Group element represented by a Euclidean matrix.
    `f` : `AbelianBasis`
        Basis function to transform.

    Returns
    -------
    `Multiple[AbelianBasis]`
        `Multiple(phase, f)` where `phase` is the scalar eigenvalue of `f`
        under the action of `g`.

    Raises
    ------
    `ValueError`
        If `g.axes` and `f.axes` do not match, if `f` is not an eigenfunction
        of `g`, or if `f.rep` is the zero vector.
    """
    if set(g.axes) != set(f.axes):
        raise ValueError(
            f"Axes of AbelianGroup and AbelianBasis must match: {g.axes} != {f.axes}"
        )

    g_irrep = g.euclidean_repr(f.order)
    basis_rep = f.rep
    transformed_rep = g_irrep @ basis_rep

    phase = None
    for n in range(transformed_rep.rows):
        basis_term = basis_rep[n]
        transformed_term = transformed_rep[n]
        if basis_term != 0:
            if phase is None:
                phase = sy.simplify(transformed_term / basis_term)
            elif sy.simplify(transformed_term - phase * basis_term) != 0:
                raise ValueError(f"{f} is not a basis function for {g}!")
        elif sy.simplify(transformed_term) != 0:
            raise ValueError(f"{f} is not a basis function for {g}!")

    if phase is None:
        raise ValueError(f"{f} is a trivial basis function: zero")

    return Multiple(phase, f)


@dataclass(frozen=True, init=False)
class AbelianOpr(Opr, HasBase[AffineSpace]):
    """
    Abelian operator acting on polynomial coordinate functions.

    This class combines an abelian linear representation with a translation:
    `x -> g x + t`, where `g` is carried by `AbelianGroup` and `t` by `offset`.

    Parameters
    ----------
    g : `AbelianGroup`
        Linear part of the affine transformation.
    The operator is initialized at the canonical origin of the identity affine
    basis. To center it at a specific point, construct it first and then call
    `fixpoint_at(...)`.
    """

    g: AbelianGroup
    offset: Offset

    @classmethod
    def _from_parts(cls, g: AbelianGroup, offset: Offset) -> "AbelianOpr":
        obj = object.__new__(cls)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "offset", offset)
        return obj

    def __init__(
        self,
        g: AbelianGroup,
        offset: Offset | None = None,
    ):
        if offset is not None:
            raise TypeError(
                "AbelianOpr does not accept offset=... directly. "
                "Construct AbelianOpr(g) and use fixpoint_at(...) to set its center."
            )
        dim = g.irrep.rows
        base = AffineSpace(basis=sy.ImmutableDenseMatrix.eye(dim))
        offset = Offset(rep=sy.ImmutableDenseMatrix([0] * dim), space=base)
        object.__setattr__(self, "g", g)
        object.__setattr__(self, "offset", offset)

    def base(self) -> AffineSpace:
        """
        Get the affine space where this element acts.

        Returns
        -------
        `AffineSpace`
            Acting space, identical to `offset.space`.
        """
        return self.offset.space

    @lru_cache(maxsize=None)
    def rebase(self, new_base: AffineSpace) -> "AbelianOpr":
        """
        Re-express this transform in a different affine space basis.

        Parameters
        ----------
        `new_base` : `AffineSpace`
            Target affine space for the transformed representation.

        Returns
        -------
        `AbelianOpr`
            New element with both linear and translation parts expressed in
            `new_base` coordinates.
        """
        old_base = self.offset.space
        B_old = old_base.basis
        if not isinstance(B_old, sy.ImmutableDenseMatrix):
            B_old = sy.ImmutableDenseMatrix(B_old)
        B_new = new_base.basis
        if not isinstance(B_new, sy.ImmutableDenseMatrix):
            B_new = sy.ImmutableDenseMatrix(B_new)

        irrep = self.g.irrep
        if not isinstance(irrep, sy.ImmutableDenseMatrix):
            irrep = sy.ImmutableDenseMatrix(irrep)

        change = B_new.inv() @ B_old
        new_irrep = change @ irrep @ change.inv()
        return AbelianOpr._from_parts(
            g=AbelianGroup(irrep=sy.ImmutableDenseMatrix(new_irrep), axes=self.g.axes),
            offset=self.offset.rebase(new_base),
        )

    def fixpoint_at(self, r: Offset, rebase: bool = False) -> "AbelianOpr":
        """
        Return a transform with the same linear part whose invariant fixed point is `r`.

        For the affine action `x -> R x + t`, requiring `r` to be fixed means
        `R r + t = r`, so the translation must be `t = (I - R) r`.

        Parameters
        ----------
        `r` : `Offset`
            Desired fixed point.
        `rebase` : `bool`, default `False`
            Base-handling mode when `r.space` differs from this transform's base:
            if `False`, rebase `r` to this transform's base and keep the
            returned transform in its current base; if `True`, rebase the
            transform to `r.space` and return the result there.

        Returns
        -------
        `AbelianOpr`
            A new affine operator with the same linear part and with `r` as an
            invariant point.
        """
        t = self.rebase(r.space) if rebase and r.space != self.offset.space else self
        r_target = r if t.offset.space == r.space else r.rebase(t.offset.space)

        irrep = t.g.irrep
        if not isinstance(irrep, sy.ImmutableDenseMatrix):
            irrep = sy.ImmutableDenseMatrix(irrep)

        r_rep = r_target.rep
        if not isinstance(r_rep, sy.ImmutableDenseMatrix):
            r_rep = sy.ImmutableDenseMatrix(r_rep)

        ident = sy.eye(irrep.rows)
        if not isinstance(ident, sy.ImmutableDenseMatrix):
            ident = sy.ImmutableDenseMatrix(ident)

        fixed_offset = Offset(
            rep=sy.ImmutableDenseMatrix((ident - irrep) @ r_rep),
            space=t.offset.space,
        )
        return AbelianOpr._from_parts(
            g=AbelianGroup(irrep=irrep, axes=t.g.axes),
            offset=fixed_offset,
        )


@AbelianOpr.register(AbelianBasis)
def _(t: AbelianOpr, f: AbelianBasis) -> Multiple[AbelianBasis]:
    """
    Apply an affine operator to an abelian basis function.

    For `AbelianBasis`, the affine translation is intentionally ignored, so
    this action is exactly the same as applying the underlying linear
    `AbelianGroup`.

    Parameters
    ----------
    `t` : `AbelianOpr`
        The affine operator to apply.
    `f` : `AbelianBasis`
        The basis function to be transformed.

    Returns
    -------
    `Tuple[sy.Expr, AbelianBasis]`
        A symbolic phase factor (sy.Expr) such that
        `t.g.euclidean_repr(f.order) @ f.rep == phase * f.rep`;
        and the original `AbelianBasis` (unchanged).

    Raises
    ------
    `ValueError`
        Propagated from `t.g @ f`.
    """
    return t.g @ f


@lru_cache(
    maxsize=None
)  # The maximum number of Offset is restricted by the current system.
def _apply_abelian_opr_to_offset_cached(t: AbelianOpr, offset: Offset) -> Offset:
    if offset.space != t.offset.space:
        t = t.rebase(offset.space)

    linear_rep = t.g.irrep
    if not isinstance(linear_rep, sy.ImmutableDenseMatrix):
        linear_rep = sy.ImmutableDenseMatrix(linear_rep)

    translation = t.offset.rep
    if not isinstance(translation, sy.ImmutableDenseMatrix):
        translation = sy.ImmutableDenseMatrix(translation)

    top = linear_rep.row_join(translation)
    bottom = sy.zeros(1, linear_rep.cols).row_join(sy.ones(1, 1))
    affine_rep = sy.ImmutableDenseMatrix(top.col_join(bottom))

    rep = offset.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)
    hom = rep.col_join(sy.ones(1, 1))
    new_hom = affine_rep @ hom
    new_rep = new_hom[:-1, :]
    return Offset(rep=sy.ImmutableDenseMatrix(new_rep), space=offset.space)


@AbelianOpr.register(Offset)
def _(t: AbelianOpr, offset: Offset) -> Offset:
    """
    Apply an affine operator to an `Offset`.

    This implementation rebases the transform into the input offset's space and
    then applies the homogeneous affine matrix in those coordinates.

    Parameters
    ----------
    `t` : `AbelianOpr`
        The affine operator to apply. If its internal `offset.space` does
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
    After `AbelianOpr.rebase`, the transform's linear part and `offset.rep` are
    all expressed in the same coordinate system, so the homogeneous affine
    action is valid directly.
    """
    return _apply_abelian_opr_to_offset_cached(t, offset)


@lru_cache(maxsize=None)
def _abelian_momentum_action_matrix(
    t: AbelianOpr, real_space: AffineSpace
) -> sy.ImmutableDenseMatrix:
    if t.base() != real_space:
        t = t.rebase(real_space)

    linear_rep = t.g.irrep
    if not isinstance(linear_rep, sy.ImmutableDenseMatrix):
        linear_rep = sy.ImmutableDenseMatrix(linear_rep)

    return sy.ImmutableDenseMatrix(linear_rep.inv().T)


@lru_cache(
    maxsize=None
)  # The maximum number of Momentum is restricted by the current system.
def _apply_abelian_opr_to_momentum_cached(t: AbelianOpr, k: Momentum) -> Momentum:
    real_space = k.base().dual
    action = _abelian_momentum_action_matrix(t, real_space)

    rep = k.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)
    new_rep = action @ rep
    return Momentum(rep=sy.ImmutableDenseMatrix(new_rep), space=k.base())


@AbelianOpr.register(Momentum)
def _(t: AbelianOpr, k: Momentum) -> Momentum:
    """
    Apply an affine operator to a Momentum in fractional reciprocal coordinates.

    This implementation assumes:
    - `k.rep` stores fractional coordinates in the reciprocal lattice basis.
    - after `t.rebase(real_space)`, `t.g.irrep` is expressed in the same real-space
      coordinates as `real_space.basis`.
    - translations do not act on momenta, so only the linear part is used.

    If `R` is the real-space linear map in those coordinates, then reciprocal
    fractional coordinates transform contravariantly as `k' = (R^{-1})^T k`.

    Parameters
    ----------
    `t` : `AbelianOpr`
        The affine operator to apply. If its base affine space does not
        match the real-space dual of `k`, it is rebased accordingly.
    `k` : `Momentum`
        The momentum expressed in fractional reciprocal coordinates of its
        reciprocal lattice basis.

    Returns
    -------
    `Tuple[sy.Expr | None, Momentum]`
        The irrep of this transformation, `None` if `k` is not a fix point;
        and the transformed momentum in the same reciprocal lattice space as `k`.
    """
    return _apply_abelian_opr_to_momentum_cached(t, k)
