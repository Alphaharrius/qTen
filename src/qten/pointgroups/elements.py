"""
Symbolic point-group element representations.

This module defines the core point-group objects used by QTen's symmetry
machinery. [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] stores an
abelian generator representation, derives Euclidean polynomial bases, and
computes symbolic eigen-basis sectors. [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis]
labels those sectors, while [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr]
couples a group action with an affine offset for use as a symbolic operator.

Repository usage
----------------
Use this module for explicit point-group construction and algebra. Higher-level
query-string construction is available through
[`pointgroup()`][qten.pointgroups._pointgroups.pointgroup], and tensor/Hilbert
space projection helpers live in [`qten.pointgroups.ops`][qten.pointgroups.ops].
"""

from dataclasses import dataclass
from typing import Dict, Tuple, cast
from collections import OrderedDict
from itertools import product
from functools import lru_cache, reduce
import sympy as sy

from ..abstracts import HasBase, Operable
from ..geometries import AffineSpace, Momentum, Offset
from ..symbolics import Opr
from ..validations import need_validation
from ..validations.symbolics import check_invertibility, check_numerical
from ..utils.collections_ext import FrozenDict
from ..symbolics import Multiple
from .basis import PointGroupBasis


def _require_unique_axes(axes: Tuple[sy.Symbol, ...], *, role: str) -> None:
    if len(set(axes)) != len(axes):
        raise ValueError(
            f"PointGroupElement {role} axes must be unique for composition, got {axes}."
        )


def _merged_axes(
    left_axes: Tuple[sy.Symbol, ...], right_axes: Tuple[sy.Symbol, ...]
) -> Tuple[sy.Symbol, ...]:
    """
    Build the canonical merged axis order for composing two groups.

    The merge preserves the full left-axis order and then appends only those
    right axes that are not already present.
    """
    return left_axes + tuple(axis for axis in right_axes if axis not in left_axes)


def _embed_irrep_to_axes(
    irrep: sy.ImmutableDenseMatrix,
    src_axes: Tuple[sy.Symbol, ...],
    dst_axes: Tuple[sy.Symbol, ...],
) -> sy.ImmutableDenseMatrix:
    """
    Embed an operator into a larger/reordered axis basis.

    Axes present in `src_axes` are mapped into `dst_axes` by symbol identity.
    Any axis present in `dst_axes` but absent in `src_axes` is treated as an
    untouched coordinate and therefore carries the identity action.
    """
    axis_to_dst = {axis: i for i, axis in enumerate(dst_axes)}
    embedded = sy.ImmutableDenseMatrix.eye(len(dst_axes))
    data = sy.Matrix(embedded)
    for i, row_axis in enumerate(src_axes):
        for j, col_axis in enumerate(src_axes):
            data[axis_to_dst[row_axis], axis_to_dst[col_axis]] = irrep[i, j]
    return sy.ImmutableDenseMatrix(sy.simplify(data))


@need_validation(check_invertibility("irrep"), check_numerical("irrep"))
@dataclass(frozen=True)
class PointGroupElement(Opr):
    r"""
    Abelian linear operator represented on Cartesian coordinate functions.

    [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] stores the linear part `g` of a symmetry/operator as an
    exact matrix `irrep` acting on the coordinate axes `axes`. It provides the
    order-dependent polynomial representations induced by that linear action
    and the corresponding eigen-basis functions ([`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis]).

    Mathematical meaning
    --------------------
    Let the coordinate vector be \(x = (x_1, \ldots, x_d)^{\mathsf{T}}\).
    The matrix `irrep` defines a linear action \(x \mapsto Gx\), where \(G\)
    is the stored `irrep` matrix.

    From this degree-1 action, the class constructs higher-order polynomial
    representations on homogeneous monomials of total degree `order`. For
    example:

    For `order = 0`, the representation acts on constant functions and is
    always the trivial `1x1` representation `[1]`. For `order = 1`, the
    representation is the original Euclidean representation `irrep`. For
    `order = 2`, the representation acts on quadratic monomials such as `x^2`,
    `xy`, and `y^2`.

    Because coordinate symbols commute, the raw tensor-product representation is
    symmetrized onto the commuting monomial basis. The resulting matrix is
    returned by [`euclidean_repr(order)`][qten.pointgroups.elements.PointGroupElement.euclidean_repr].

    For a homogeneous monomial basis \(\phi_m(x)\), the derived representation
    acts by rewriting \(\phi_m(Gx)\) back in the commuting monomial basis.

    Parameters
    ----------
    irrep : sy.ImmutableDenseMatrix
        Exact linear representation matrix of the operator in the coordinate
        basis defined by `axes`.
    axes : Tuple[sy.Symbol, ...]
        Ordered coordinate symbols on which `irrep` acts.

    Attributes
    ----------
    irrep : sy.ImmutableDenseMatrix
        Exact linear representation matrix of the operator in the coordinate
        basis defined by `axes`.
    axes : Tuple[sy.Symbol, ...]
        Ordered coordinate symbols on which `irrep` acts.

    Main API
    --------
    [`euclidean_repr(order)`][qten.pointgroups.elements.PointGroupElement.euclidean_repr]
    returns the symmetrized linear action on homogeneous commuting monomials of
    degree `order`. [`basis(order)`][qten.pointgroups.elements.PointGroupElement.basis]
    returns eigen-basis functions of that representation as
    [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis] objects keyed by
    eigenvalue. [`basis_table`][qten.pointgroups.elements.PointGroupElement.basis_table]
    collects representative eigen-basis functions across increasing polynomial
    orders until all characters of the finite represented element are found.
    [`group_order(max_order=128)`][qten.pointgroups.elements.PointGroupElement.group_order]
    returns the smallest positive integer `n` such that `irrep**n = I`.

    Notes
    -----
    [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] is the linear object. To obtain an affine operator of the
    form \(x \mapsto gx + t\), wrap it in [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr]. In that sense, [`PointGroupOpr`][qten.pointgroups.elements.PointGroupOpr]
    is the affine extension of [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement].

    `PointGroupElement @ PointGroupElement` composes linear maps in the same algebraic
    order as every other [`Opr`][qten.symbolics.hilbert_space.Opr]: `(a @ b) @ x == a(b(x))`. When the two groups
    use different but compatible ordered axis tuples, composition first embeds
    both matrices into a common axis basis. The merged basis preserves the full
    left-axis order and appends only unseen right axes. Missing axes act by the
    identity, while shared axes are aligned by symbol and reordered as needed.

    The [`group_order()`][qten.pointgroups.elements.PointGroupElement.group_order] and [`basis_table`][qten.pointgroups.elements.PointGroupElement.basis_table] utilities assume the represented
    element has finite order. They are appropriate for finite abelian point
    symmetries, but may fail or be incomplete for infinite-order linear maps.
    """

    irrep: sy.ImmutableDenseMatrix
    """
    Exact linear representation matrix of the operator in the coordinate
    basis defined by `axes`. This is the degree-1 action from which higher
    polynomial representations are constructed.
    """
    axes: Tuple[sy.Symbol, ...]
    """
    Ordered coordinate symbols on which `irrep` acts. Their order fixes the
    ambient coordinate basis for all derived polynomial representations.
    """

    @lru_cache
    def _full_indices(self, order: int):
        """
        Enumerate all ordered monomial indices for the tensor-product basis.

        Returns
        -------
        Tuple[Tuple[sy.Symbol, ...], ...]
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
        Tuple[Tuple[sy.Symbol, ...], ...]
            Ordered subset of `_full_indices(order)` where permutations that differ
            only by factor ordering are collapsed to a single representative.
        """
        indices = self._full_indices(order)
        _, select_rules = PointGroupElement._get_contract_select_rules(indices)
        sorted_rules = sorted(select_rules, key=lambda x: x[1])
        return tuple(indices[n] for n, _ in sorted_rules)

    @lru_cache
    def euclidean_basis(self, order: int) -> sy.ImmutableDenseMatrix:
        """
        Return commuting Euclidean monomials spanning the polynomial basis.

        Parameters
        ----------
        order : int
            Homogeneous polynomial degree. `order=0` returns the constant
            monomial basis.

        Returns
        -------
        sy.ImmutableDenseMatrix
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

        Returned maps
        -------------
        Contract rules map each full tensor-product index position to a
        commutative monomial class. Select rules pick one representative full index
        position for each commutative monomial class.

            Parameters
            ----------
            indices : Tuple[Tuple[sy.Symbol, ...], ...]
                Full ordered tensor-product indices.

            Returns
            -------
            Tuple[list[Tuple[int, int]], list[Tuple[int, int]]]
                Pair `(contract_rules, select_rules)` used to contract the raw
                tensor-product representation onto commuting monomials.
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
        sy.ImmutableDenseMatrix | sy.MatrixBase
            Kronecker power `irrep ⊗ ... ⊗ irrep` with `order` factors.
        """
        if order == 0:
            return sy.ImmutableDenseMatrix([[1]])
        return reduce(sy.kronecker_product, (self.irrep,) * order)

    @lru_cache
    def euclidean_repr(self, order: int) -> sy.ImmutableDenseMatrix:
        """
        Symmetrized representation on the commuting polynomial basis.

        Parameters
        ----------
        order : int
            Homogeneous polynomial degree for the induced representation.
            `order=0` returns the trivial one-dimensional representation.

        Returns
        -------
        sy.ImmutableDenseMatrix
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
        r"""
        Return the order of this represented group element.

        The order is the smallest positive integer `n` such that \(G^n = I\),
        where \(G\) is `irrep` and \(I\) is the identity matrix of matching size.

        Parameters
        ----------
        max_order : int, default 128
            Maximum positive exponent to test during the exact search.

        Returns
        -------
        int
            The smallest positive exponent for which the represented matrix
            returns to the identity.

        Raises
        ------
        ValueError
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
    def inv(self) -> "PointGroupElement":
        """
        Return the inverse linear operator in the same ordered axis basis.

        The inverse is computed exactly from `irrep.inv()` and keeps the same
        `axes`, so `self @ self.inv()` and `self.inv() @ self` both represent
        the identity map on that coordinate system.
        """
        return PointGroupElement(
            irrep=sy.ImmutableDenseMatrix(sy.simplify(self.irrep.inv())),
            axes=self.axes,
        )

    @lru_cache
    def basis(self, order: int) -> FrozenDict:
        """
        Compute abelian eigen-basis functions from [`euclidean_repr(order)`][qten.pointgroups.elements.PointGroupElement.euclidean_repr] eigenvectors.

        Parameters
        ----------
        order : int
            Homogeneous polynomial degree used to build the Euclidean
            representation before diagonalization.

        Returns
        -------
        FrozenDict
            Mapping from eigenvalue to normalized [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis] eigenfunction.
            Normalization is fixed by dividing by the first non-zero coefficient
            in each eigenvector.
        """
        transform = self.euclidean_repr(order)
        eig = transform.eigenvects()

        tbl = {}
        for v, _, vec_group in eig:
            vec = vec_group[0]
            tbl[v] = PointGroupBasis.from_rep(
                rep=sy.ImmutableDenseMatrix(vec),
                euclidean_basis=self.euclidean_basis(order),
                axes=self.axes,
                order=order,
                irrep=sy.simplify(v),
            )

        return FrozenDict(tbl)

    @property
    @lru_cache
    def basis_table(self) -> FrozenDict:
        """
        Build a complete eigen-basis lookup table across polynomial orders.

        The table is accumulated by increasing homogeneous order, starting from
        `0`, until enough eigen-basis functions have been found to cover the
        full finite group order returned by
        [`group_order`][qten.pointgroups.elements.PointGroupElement.group_order].

        Returns
        -------
        FrozenDict
            Mapping from eigenvalue/character to a representative
            [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis].

        Raises
        ------
        ValueError
            If no complete table is found up to order `group_order() - 1`.
        """
        g_order = self.group_order()
        tbl: Dict[sy.Expr, PointGroupBasis] = {}
        for order in range(g_order):
            tbl = {**self.basis(order), **tbl}
            if len(tbl) == g_order:
                return FrozenDict(tbl)
        raise ValueError(
            f"Failed to build a complete basis table up to order {g_order - 1}."
        )


@Operable.__matmul__.register
def _(left: PointGroupElement, right: PointGroupElement) -> PointGroupElement:
    """
    Compose two abelian linear operators in algebraic `@` order.

    The returned group represents the map `left(right(x))`.

    Axis handling
    -------------
    If `left.axes` and `right.axes` differ, both operators are first embedded
    into a common axis basis before multiplication:

    The merged axis order preserves all of `left.axes`, then appends any
    right-only axes in their original order. Shared axes are aligned by symbol,
    even if their positions differ. Axes missing from one operator act
    trivially and therefore contribute an identity block along that coordinate.

    For example:

    For example, `(x, y)` composed with `(y, x)` aligns both operators to
    `(x, y)` by permutation. `(x, y)` composed with `(y, z)` aligns both to
    `(x, y, z)`, with the first operator acting as identity on `z` and the
    second as identity on `x`.

    Composition requires each operand's axis tuple to contain unique symbols.
    Repeated axes are rejected because they do not define an unambiguous
    coordinate alignment.

    Parameters
    ----------
    left : PointGroupElement
        Operator applied after `right`.
    right : PointGroupElement
        Operator applied before `left`.

    Returns
    -------
    PointGroupElement
        Composed linear operator expressed on the merged ordered axis basis.

    Raises
    ------
    ValueError
        If either operand has repeated axes.
    """
    _require_unique_axes(left.axes, role="left")
    _require_unique_axes(right.axes, role="right")

    merged = _merged_axes(left.axes, right.axes)
    left_irrep = _embed_irrep_to_axes(left.irrep, left.axes, merged)
    right_irrep = _embed_irrep_to_axes(right.irrep, right.axes, merged)
    return PointGroupElement(
        irrep=sy.ImmutableDenseMatrix(sy.simplify(left_irrep @ right_irrep)),
        axes=merged,
    )


@PointGroupElement.register(PointGroupBasis)
def _(g: PointGroupElement, f: PointGroupBasis) -> Multiple[PointGroupBasis]:
    """
    Apply a generated group element directly to a point-group basis label.

    Unlike [`PointGroupBasis`][qten.pointgroups.elements.PointGroupBasis], a
    [`PointGroupBasis`][qten.pointgroups.basis.PointGroupBasis] need not be a
    one-dimensional eigenfunction of a single element. For example, in the
    two-dimensional `E` sector of `C4v`, a 90-degree rotation maps `x` to `y`.
    This implementation therefore transforms the stored polynomial
    representation and returns the transformed basis label rather than forcing
    a scalar phase.
    """
    if set(g.axes) != set(f.axes):
        raise ValueError(
            f"Axes of PointGroupElement and PointGroupBasis must match: {g.axes} != {f.axes}"
        )

    transformed_rep = sy.ImmutableDenseMatrix(g.euclidean_repr(f.order) @ f.rep)
    try:
        scale = next(entry for entry in transformed_rep if entry != 0)
    except StopIteration as exc:
        raise ValueError(f"{f} is a trivial basis function: zero") from exc
    transformed_basis = PointGroupBasis.from_rep(
        rep=transformed_rep,
        euclidean_basis=g.euclidean_basis(f.order),
        axes=g.axes,
        order=f.order,
        group=f.group,
        irrep=f.irrep,
        irrep_dim=f.irrep_dim,
        copy_index=f.copy_index,
        component_index=f.component_index,
    )
    return Multiple(sy.Abs(scale), transformed_basis)


@dataclass(frozen=True, init=False)
class PointGroupOpr(Opr, HasBase[AffineSpace]):
    r"""
    Abelian operator acting on polynomial coordinate functions.

    This class combines an abelian linear representation with a translation:
    \(x \mapsto gx + t\), where `g` is carried by
    [`PointGroupElement`][qten.pointgroups.elements.PointGroupElement] and \(t\) by
    `offset`.

    Parameters
    ----------
    g : PointGroupElement
        Linear part of the affine transformation.
    offset : Offset
        Translation part of the affine transformation, stored in the same
        affine space on which ``g`` acts.

    Attributes
    ----------
    g : PointGroupElement
        Linear part of the affine transformation.
    offset : Offset
        Translation part of the affine transformation, stored in the same
        affine space on which `g` acts.

    Notes
    -----
    The operator is initialized at the canonical origin of the identity affine
    basis. To center it at a specific point, construct it first and then call
    [`fixpoint_at(...)`][qten.pointgroups.elements.PointGroupOpr.fixpoint_at].
    """

    g: PointGroupElement
    """
    Linear part of the affine transformation, represented exactly on the
    ordered coordinate axes of the operator's ambient affine space.
    """
    offset: Offset
    r"""
    Translation part of the affine transformation, stored in the same affine
    space on which `g` acts so the full map has the form
    \(x \mapsto gx + \mathrm{offset}\).
    """

    @classmethod
    def _from_parts(cls, g: PointGroupElement, offset: Offset) -> "PointGroupOpr":
        obj = object.__new__(cls)
        object.__setattr__(obj, "g", g)
        object.__setattr__(obj, "offset", offset)
        return obj

    def __init__(
        self,
        g: PointGroupElement,
        offset: Offset | None = None,
    ):
        if offset is not None:
            raise TypeError(
                "PointGroupOpr does not accept offset=... directly. "
                "Construct PointGroupOpr(g) and use fixpoint_at(...) to set its center."
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
        AffineSpace
            Acting space, identical to `offset.space`.
        """
        return self.offset.space

    @lru_cache(maxsize=None)
    def rebase(self, new_base: AffineSpace) -> "PointGroupOpr":
        """
        Re-express this transform in a different affine space basis.

        Parameters
        ----------
        new_base : AffineSpace
            Target affine space for the transformed representation.

        Returns
        -------
        PointGroupOpr
            New element with both linear and translation parts expressed in
            new_base coordinates.
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
        return PointGroupOpr._from_parts(
            g=PointGroupElement(irrep=sy.ImmutableDenseMatrix(new_irrep), axes=self.g.axes),
            offset=self.offset.rebase(new_base),
        )

    def fixpoint_at(self, r: Offset, rebase: bool = False) -> "PointGroupOpr":
        r"""
        Return a transform with the same linear part whose invariant fixed point is `r`.

        For the affine action \(x \mapsto R x + t\), requiring \(r\) to be
        fixed means \(Rr + t = r\), so the translation must be
        \(t = (I - R)r\).

        Parameters
        ----------
        r : Offset
            Desired fixed point.
        rebase : bool, default `False`
            Base-handling mode when `r.space` differs from this transform's base:
            if `False`, rebase `r` to this transform's base and keep the
            returned transform in its current base; if `True`, rebase the
            transform to `r.space` and return the result there.

        Returns
        -------
        PointGroupOpr
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
        return PointGroupOpr._from_parts(
            g=PointGroupElement(irrep=irrep, axes=t.g.axes),
            offset=fixed_offset,
        )


@PointGroupOpr.register(PointGroupBasis)
def _(t: PointGroupOpr, f: PointGroupBasis) -> Multiple[PointGroupBasis]:
    """Apply an affine wrapper to a point-group basis through its linear part."""

    return cast(Multiple[PointGroupBasis], t.g @ f)


@lru_cache(
    maxsize=None
)  # The maximum number of Offset is restricted by the current system.
def _apply_point_group_opr_to_offset_cached(t: PointGroupOpr, offset: Offset) -> Offset:
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


@PointGroupOpr.register(Offset)
def _(t: PointGroupOpr, offset: Offset) -> Offset:
    """
    Apply an affine operator to an [`Offset`][qten.geometries.spatials.Offset].

    This implementation rebases the transform into the input offset's space and
    then applies the homogeneous affine matrix in those coordinates.

    Parameters
    ----------
    t : PointGroupOpr
        The affine operator to apply. If its internal `offset.space` does
        not match `offset.space`, the transform is rebased to the Offset's space.
    offset : Offset
        The spatial offset (column vector) to transform.

    Returns
    -------
    Offset
        Transformed offset expressed in the same
        [`AffineSpace`][qten.geometries.spatials.AffineSpace] as the input
        `offset`.

    Notes
    -----
    After `PointGroupOpr.rebase`, the transform's linear part and `offset.rep` are
    all expressed in the same coordinate system, so the homogeneous affine
    action is valid directly.
    """
    return _apply_point_group_opr_to_offset_cached(t, offset)


@lru_cache(maxsize=None)
def _abelian_momentum_action_matrix(
    t: PointGroupOpr, real_space: AffineSpace
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
def _apply_point_group_opr_to_momentum_cached(t: PointGroupOpr, k: Momentum) -> Momentum:
    real_space = k.base().dual
    action = _abelian_momentum_action_matrix(t, real_space)

    rep = k.rep
    if not isinstance(rep, sy.ImmutableDenseMatrix):
        rep = sy.ImmutableDenseMatrix(rep)
    new_rep = action @ rep
    return Momentum(rep=sy.ImmutableDenseMatrix(new_rep), space=k.base())


@PointGroupOpr.register(Momentum)
def _(t: PointGroupOpr, k: Momentum) -> Momentum:
    r"""
    Apply an affine operator to a Momentum in fractional reciprocal coordinates.

    Assumptions
    -----------
    `k.rep` stores fractional coordinates in the reciprocal lattice basis.
    After `t.rebase(real_space)`, `t.g.irrep` is expressed in the same
    real-space coordinates as `real_space.basis`. Translations do not act on
    momenta, so only the linear part is used.

    If \(R\) is the real-space linear map in those coordinates, then reciprocal
    fractional coordinates transform contravariantly as
    \(k' = (R^{-1})^{\mathsf{T}} k\).

    Parameters
    ----------
    t : PointGroupOpr
        The affine operator to apply. If its base affine space does not
        match the real-space dual of `k`, it is rebased accordingly.
    k : Momentum
        The momentum expressed in fractional reciprocal coordinates of its
        reciprocal lattice basis.

    Returns
    -------
    Momentum
        Transformed momentum in the same reciprocal lattice space as `k`.
    """
    return _apply_point_group_opr_to_momentum_cached(t, k)
