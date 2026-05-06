"""
Symbolic Hilbert-space bases and operators.

This module defines QTen's symbolic single-particle basis labels, spans, and
operators. [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] represents a
single symbolic basis state, [`U1Span`][qten.symbolics.hilbert_space.U1Span]
collects compatible basis states, and
[`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] provides the indexed
state space used by labelled tensors. Operator classes such as
[`Opr`][qten.symbolics.hilbert_space.Opr] and
[`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] encode symbolic actions on
those basis labels.

Repository usage
----------------
Use this module when defining, combining, converting, or applying symbolic
Hilbert-space basis data. Spatial state-space containers live in
[`qten.symbolics.state_space`][qten.symbolics.state_space], while convenience
operators for common transformations live in [`qten.symbolics.ops`][qten.symbolics.ops].
"""

from abc import ABC
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Generic,
    Union,
)
from typing import cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from itertools import product

import numpy as np
import torch
import sympy as sy

from ..utils.collections_ext import FrozenDict
from ..utils.types_ext import full_typename
from ..validations import need_validation
from ..abstracts import (
    AbstractKet,
    Convertible,
    Operable,
    Functional,
    Span,
    HasRays,
)
from ..geometries.spatials import Spatial
from .state_space import StateSpace, StateSpaceFactorization, same_rays
from ..linalg.tensors import Tensor
from ..precision import get_precision_config
from ..utils.devices import Device
from . import Multiple


_IrrepType = TypeVar("_IrrepType")
"""Defines a irreducible representation type."""


def _check_u1_multiplicity(value: "U1Basis") -> None:
    counts: Dict[Type, int] = {}
    for irrep in value.base:
        irrep_type = type(irrep)
        counts[irrep_type] = counts.get(irrep_type, 0) + 1
    non_singletons = {t: c for t, c in counts.items() if c != 1}
    if non_singletons:
        detail = ", ".join(f"{t.__name__}:{c}" for t, c in non_singletons.items())
        raise ValueError(
            "U1Basis allows only irrep with unity multiplicity; "
            f"got multiple non-singleton types ({detail})."
        )


@need_validation(_check_u1_multiplicity)
@dataclass(frozen=True)
class U1Basis(
    Spatial, Multiple[Tuple[Any, ...]], AbstractKet[sy.Expr], HasRays, Convertible
):
    r"""
    Immutable single-particle basis state built from typed irreps.

    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] is a symbolic tensor-product state with U(1) irreducible representation
    presented as an ordered tuple of irreps. This object is
    intentionally symbolic: it stores basis labels only and does not store amplitudes
    or coefficients.

    A key invariant is enforced at construction time: for each concrete irrep
    type, the state must contain exactly one ket of that type. In other words,
    irrep multiplicities must be unity. This guarantees that type-based updates
    via [`replace`][qten.symbolics.hilbert_space.U1Basis.replace] are unambiguous and fast.

    Mathematical convention
    -----------------------
    A basis state is represented symbolically as
    \(|\psi\rangle = c\,|\rho_1,\rho_2,\ldots,\rho_m\rangle\), where `coef`
    stores \(c\), `base` stores the construction-order tuple
    \((\rho_1,\rho_2,\ldots,\rho_m)\), and `rep` stores the canonical
    type-sorted tuple used for equality, hashing, and lookup. The object is a
    basis label with a U(1) prefactor, not a dense state vector.

    Attributes
    ----------
    coef : sy.Expr
        Symbolic U(1) coefficient carried by this basis state.
    base : Tuple[Any, ...]
        Immutable tuple of irrep labels before canonical sorting.
    rep : Tuple[Any, ...]
        Immutable canonical irrep order sorted by concrete irrep type name
        (`module.qualname`).

    Notes
    -----
    `dim` is always `1`; this type represents one basis vector. Irrep order is
    canonicalized at construction, so permutations of the same typed irreps
    produce the same internal `rep` tuple.
    [`replace(irrep)`][qten.symbolics.hilbert_space.U1Basis.replace] substitutes
    the unique irrep with the same concrete runtime type as `irrep`.

    The `@` dispatch overload combines two basis states as a symbolic tensor
    product. For `a @ b`, the output coefficient is
    `simplify(a.coef * b.coef)` and the output `base` tuple is
    `a.base + b.base` in tensor-product construction order. `__post_init__`
    then computes `rep` by sorting that `base` tuple by fully-qualified runtime
    type name (`module.qualname`). The original `base` order is retained for
    display and construction semantics, while `rep` is the canonical sorted
    representation used for type-order-sensitive behavior. If either operand
    has an empty `base`, that operand acts as the tensor-product identity and
    the other operand is returned unchanged. The resulting
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] is still validated by
    `__post_init__`, so combining states that contain the same concrete irrep
    type is rejected by the unity-multiplicity invariant.

    In coefficient notation, tensor-product composition multiplies the U(1)
    weights as \(c_{a \otimes b} = c_a c_b\). Equivalently,
    \((c_a|\alpha\rangle) \otimes (c_b|\beta\rangle)
    = c_a c_b\,|\alpha,\beta\rangle\). In code, this is `a @ b` and is implemented as
    `U1Basis(simplify(a.coef * b.coef), a.base + b.base)`.

    The `|` dispatch overloads build a
    [`U1Span`][qten.symbolics.hilbert_space.U1Span] of distinct
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] values. Ordering (`<`,
    `>`) compares the number of irreps, then the tuple of fully-qualified irrep
    type names (`module.qualname`) from `self.base`, then the canonical
    irrep-value tuple itself. If irrep values of matching types are not
    orderable, the comparison raises from the underlying irrep objects.

    Raises
    ------
    ValueError
        Raised in `__post_init__` when any irrep type appears with multiplicity
        different from `1`.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base",
            tuple(
                sorted(
                    self.base,
                    key=lambda irrep: full_typename(type(irrep)),
                )
            ),
        )

    @staticmethod
    def new(*rep: Any) -> "U1Basis":
        """
        Build a [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] with the given reps and a default U(1) value of `1`.

        Parameters
        ----------
        rep : Any
            Irreps to build the state from. Input order is canonicalized in
            `__post_init__` by sorting on [`full_typename(type(irrep))`][qten.utils.types_ext.full_typename].

        Returns
        -------
        U1Basis
            A [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] instance with the given reps and a default U(1) value of `1`.
        """
        return U1Basis(coef=sy.Integer(1), base=tuple(rep))

    @property
    def dim(self) -> int:
        """The dimension of a single particle state is always `1`."""
        return 1

    def replace(self, irrep: Any) -> "U1Basis":
        """
        Return a new state where the irrep of the same concrete type is replaced.

        The method searches this state's irrep tuple for the unique irrep whose type has
        the same *exact* runtime type as `irrep` (using `type(x) is type(y)`), then
        returns a new [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] with that irrep substituted. The original instance is
        not modified.

        Parameters
        ----------
        irrep : Any
            Replacement irrep instance. Its concrete type must already exist in this
            state exactly once (enforced by `U1Basis.__post_init__`).

        Returns
        -------
        U1Basis
            A new [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] with one irrep replaced.

        Raises
        ------
        ValueError
            If this state does not contain any irrep with the same concrete
            type as `irrep`.

        Examples
        --------
        ```python
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import U1Basis

        space = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), space)
        r1 = Offset(ImmutableDenseMatrix([1]), space)
        psi = U1Basis.new(r0, "spin-up")
        out = psi.replace(r1)

        assert out.irrep_of(Offset) == r1
        assert out.irrep_of(str) == "spin-up"
        ```
        """
        target_type = type(irrep)
        reps = self.base
        for i, x in enumerate(reps):
            if type(x) is target_type:
                return replace(self, base=reps[:i] + (irrep,) + reps[i + 1 :])
        raise ValueError(
            f"U1Basis has no irrep of type {target_type.__name__} to replace."
        )

    def without(self, *T: Type[Any]) -> "U1Basis":
        """
        Return a new state with irreps of the requested concrete types removed.

        Parameters
        ----------
        *T : Type[Any]
            Concrete irrep types to remove. Matching uses exact runtime type
            identity (`type(x) is T`), not subclass checks.

        Returns
        -------
        U1Basis
            A new [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] with all irreps whose concrete types are in `T`
            removed. If none of the requested types are present, `self` is
            returned unchanged.

        Examples
        --------
        ```python
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import U1Basis

        space = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), space)
        psi = U1Basis.new(r0, "spin-up")
        local_label = psi.without(Offset)

        assert local_label.irrep_of(str) == "spin-up"
        ```
        """
        if not T:
            return self
        targets = frozenset(T)
        filtered = tuple(irrep for irrep in self.base if type(irrep) not in targets)
        if len(filtered) == len(self.base):
            return self
        return replace(self, base=filtered)

    def irrep_of(self, T: Type[_IrrepType]) -> _IrrepType:
        """
        Return the unique irrep in this state whose concrete type is `T`.

        This method performs a direct scan over `self.base` and returns the
        first irrep satisfying `type(irrep) is T`. Because
        `U1Basis.__post_init__` enforces unity multiplicity for each irrep
        type, the match is unique whenever it exists.

        Parameters
        ----------
        T : Type[_IrrepType]
            Concrete irrep type to retrieve.

        Returns
        -------
        _IrrepType
            The irrep instance of type `T` contained in this state.

        Raises
        ------
        ValueError
            If no irrep with concrete type `T` exists in this state.

        Examples
        --------
        ```python
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import U1Basis

        space = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), space)
        psi = U1Basis.new(r0, "spin-up")

        assert psi.irrep_of(Offset) == r0
        assert psi.irrep_of(str) == "spin-up"
        ```

        Notes
        -----
        Matching uses exact runtime type identity (`type(x) is T`), not subclass
        checks. Runtime is linear in the number of irreps (`O(n)`), with no
        temporary mapping allocations.
        """
        for irrep in self.base:
            if type(irrep) is T:
                return cast(_IrrepType, irrep)
        raise ValueError(f"U1Basis {self} has no irrep of type {T.__name__}.")

    @override
    def ket(self, psi: "U1Basis") -> sy.Expr:
        """
        Return the overlap of this state with another state.

        Parameters
        ----------
        psi : U1Basis
            The state to compute the overlap with.

        Returns
        -------
        sy.Expr
            The symbolic overlap of this state with `psi`. If the irreps of the
            two states do not match, the overlap is `0`. If the irreps match, the
            overlap is the product of this state's U(1) value and the conjugate of
            psi's U(1) value.
        """
        if self.base != psi.base:
            return sy.Integer(0)
        return cast(sy.Expr, (sy.conjugate(self.coef) * psi.coef).simplify())

    def __str__(self) -> str:
        """
        Return a compact ket-style label for this basis state.

        Returns
        -------
        str
            Tensor-product ket label. If `coef != 1`, the coefficient is
            prepended as a symbolic scalar factor.
        """
        ket_repr = "⊗".join(
            f"|{irrep_repr}⟩"
            if len(irrep_repr := repr(irrep)) <= 32
            else f"|{type(irrep).__name__}⟩"
            for irrep in self.base
        )
        if self.coef != sy.Integer(1):
            ket_repr = f"({self.coef}) * " + ket_repr
        return ket_repr

    def __repr__(self) -> str:
        """
        Return the developer representation of this basis state.

        Returns
        -------
        str
            Same value as `str(self)`.
        """
        return self.__str__()

    @override
    def rays(self) -> "U1Basis":
        """
        Return the canonical ray representative with U(1) coefficient `1`.

        A [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] ray ignores the
        scalar `coef` and keeps only the basis labels. This method returns a new
        basis state with the same `base` tuple and coefficient `1`. The original
        object is unchanged.

        Returns
        -------
        U1Basis
            Basis state with the same irrep labels and `coef = 1`.

        Examples
        --------
        ```python
        import sympy as sy
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import U1Basis

        space = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), space)
        psi = U1Basis(sy.Integer(3), (r0,))
        ray = psi.rays()

        assert ray.coef == 1
        assert ray.base == psi.base
        ```
        """
        return replace(self, coef=sy.Integer(1))

    def repr_types(self) -> Tuple[Type, ...]:
        """
        Get a tuple of concrete irrep types in this [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] in canonical order.

        This is the same order as `self.base`, which is determined by
        sorting on [`full_typename(type(irrep))`][qten.utils.types_ext.full_typename].

        Returns
        -------
        Tuple[Type, ...]
            Tuple of concrete irrep types in deterministic type-name order.
        """
        return tuple(type(irrep) for irrep in self.base)


@Operable.__lt__.register
def _(a: U1Basis, b: U1Basis) -> bool:
    rep_a = a.base
    rep_b = b.base
    if len(rep_a) < len(rep_b):
        return True
    if len(rep_a) > len(rep_b):
        return False
    typenames_a = tuple(full_typename(type(v)) for v in rep_a)
    typenames_b = tuple(full_typename(type(v)) for v in rep_b)
    if typenames_a < typenames_b:
        return True
    if typenames_a > typenames_b:
        return False
    return rep_a < rep_b


@Operable.__gt__.register
def _(a: U1Basis, b: U1Basis) -> bool:
    rep_a = a.base
    rep_b = b.base
    if len(rep_a) > len(rep_b):
        return True
    if len(rep_a) < len(rep_b):
        return False
    typenames_a = tuple(full_typename(type(v)) for v in rep_a)
    typenames_b = tuple(full_typename(type(v)) for v in rep_b)
    if typenames_a > typenames_b:
        return True
    if typenames_a < typenames_b:
        return False
    return rep_a > rep_b


@dataclass(frozen=True)
class U1Span(Span[U1Basis], Spatial, HasRays, Convertible):
    r"""
    Finite span of distinct single-particle basis states.

    [`U1Span`][qten.symbolics.hilbert_space.U1Span] is the additive container used by [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]'s `*` operator. It
    stores an ordered tuple of basis states and represents the symbolic span
    generated by those states. The object is immutable (`frozen=True`) and
    preserves insertion order.

    This type is intentionally lightweight: it does not store amplitudes,
    coefficients, or perform linear-algebra simplification. Duplicate handling
    is implemented in `__or__` overloads, which keep only one copy of an
    existing state when building a span.

    Mathematical convention
    -----------------------
    If the span contains basis labels \(\psi_0,\ldots,\psi_{n-1}\), it
    represents the ordered symbolic basis
    \(\mathrm{span}\{|\psi_0\rangle,\ldots,|\psi_{n-1}\rangle\}\). The order is part of the object: it determines row/column ordering in Gram
    matrices and tensor dimensions.

    Parameters
    ----------
    span : Tuple[U1Basis, ...]
        Ordered tuple of [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] elements contained in this span.

    Attributes
    ----------
    span : Tuple[U1Basis, ...]
        The underlying immutable sequence of basis states.

    Notes
    -----
    The `dim` property returns `len(span)`, i.e., the number of basis states
    currently tracked by this symbolic span.
    """

    span: Tuple["U1Basis", ...]
    """Ordered tuple of `U1Basis` elements contained in this span."""

    @property
    def dim(self) -> int:
        """Get the length of this single particle state span."""
        return len(self.span)

    def __iter__(self) -> Iterator["U1Basis"]:
        """Iterate over states in this span preserving insertion order."""
        return iter(self.span)

    @override
    def elements(self) -> Tuple["U1Basis", ...]:
        """
        Return the basis states contained in this span.

        Returns
        -------
        Tuple[U1Basis, ...]
            Ordered immutable tuple of basis states.
        """
        return self.span

    @override
    def rays(self) -> "U1Span":
        """
        Return the span obtained by ray-normalizing each basis state.

        The output preserves span order and replaces every
        [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] by
        `basis.rays()`, which sets that basis state's coefficient to `1`.

        Returns
        -------
        U1Span
            Span with the same number of basis states, each replaced by its ray
            representative.

        Examples
        --------
        ```python
        import sympy as sy
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import U1Basis, U1Span

        space = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), space)
        span = U1Span((U1Basis(sy.Integer(2), (r0,)),))
        ray_span = span.rays()

        assert ray_span.elements()[0].coef == 1
        ```
        """
        return U1Span(tuple(m.rays() for m in self.span))

    def cross_gram(self, ket: "U1Span") -> sy.ImmutableDenseMatrix:
        r"""
        Compute the overlap matrix between this span and another span.

        For left span states \(\psi_i\) and right span states \(\phi_j\), the
        returned SymPy matrix has entries
        \(G_{ij} = \langle \psi_i \mid \phi_j \rangle\). The implementation only emits a nonzero entry when the two symbolic
        states have the same ray; the stored U(1) coefficients supply the
        overlap phase.

        Parameters
        ----------
        ket : U1Span
            Right-hand span supplying the ket states.

        Returns
        -------
        sy.ImmutableDenseMatrix
            Matrix whose `(i, j)` entry is the inner product between the `i`th
            basis state of `self` and the `j`th basis state of `ket` whenever
            the two states lie on the same ray, and `0` otherwise.
        """
        tbl: Dict["U1Basis", Tuple[int, "U1Basis"]] = {
            psi.rays(): (n, psi) for n, psi in enumerate(ket.span)
        }
        out = sy.zeros(self.dim, ket.dim)
        for n, psi in enumerate(self.span):
            rays = psi.rays()
            if rays not in tbl:
                continue
            m, kpsi = tbl[rays]
            out[n, m] = psi.ket(kpsi)
        return sy.ImmutableDenseMatrix(out)


@U1Basis.add_conversion(U1Span)
def _(basis: U1Basis) -> U1Span:
    """Convert a [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] to a [`U1Span`][qten.symbolics.hilbert_space.U1Span] containing just that basis state."""
    return U1Span((basis,))


@need_validation()
@dataclass(frozen=True)
class HilbertSpace(HasRays, StateSpace[U1Basis], Span[U1Basis]):
    r"""
    Composite local Hilbert space built from states and state spans.

    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] is the symbolic basis container used by tensor-network style
    operators in this module. It extends [`StateSpace`][qten.symbolics.state_space.StateSpace] with [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] sectors.
    The inherited `structure` mapping stores each sector together with its
    integer index in basis order.

    Mathematical convention
    -----------------------
    A Hilbert space is an ordered finite basis
    \(\mathcal{H} = \mathrm{span}\{|e_0\rangle,\ldots,|e_{d-1}\rangle\}\),
    where each \(|e_i\rangle\) is a [`U1Basis`][qten.symbolics.hilbert_space.U1Basis].
    The code-level `structure` mapping stores the correspondence
    `U1Basis -> i`. That index \(i\) is the coordinate used by
    [`Tensor`][qten.linalg.tensors.Tensor] axes.

    Ray normalization removes U(1) prefactors from the labels:
    \(\mathrm{ray}(c\,|\rho\rangle) = |\rho\rangle\). Phase information is not lost when constructing overlap maps: it is stored
    in the matrix entries produced by `cross_gram`.

    The class provides helpers for common basis-management workflows:
    `elements()` returns the flattened basis states as `Tuple[U1Basis, ...]`.
    [`lookup(query)`][qten.symbolics.hilbert_space.HilbertSpace.lookup] retrieves
    a unique basis state by exact typed-irrep match.
    [`group(**groups)`][qten.symbolics.hilbert_space.HilbertSpace.group]
    partitions basis elements into labeled grouped
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] values.

    As a [`Span`][qten.abstracts.Span], [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] supports overlap/mapping computations through
    `cross_gram`, which builds a [`Tensor`][qten.linalg.tensors.Tensor] map between two spaces using [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]
    overlap (`U1Span.cross_gram`). As a [`HasRays`][qten.abstracts.HasRays], [`rays()`][qten.symbolics.hilbert_space.HilbertSpace.rays] keeps basis
    structure while replacing each element by its canonical ray representative.

    Parameters
    ----------
    structure : OrderedDict[U1Basis, int]
        Ordered sector mapping inherited from [`StateSpace`][qten.symbolics.state_space.StateSpace], where each key is a
        [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] and each value is the sector index in basis coordinates.

    Notes
    -----
    `dim` is inherited from [`StateSpace`][qten.symbolics.state_space.StateSpace]
    and equals the total flattened basis size, i.e. the number of indexed basis
    sectors. [`group()`][qten.symbolics.hilbert_space.HilbertSpace.group]
    requires disjoint selectors; overlapping grouped spans raise `ValueError`.
    [`permutation_order`][qten.symbolics.state_space.permutation_order] and
    [`embedding_order`][qten.symbolics.state_space.embedding_order] operate on
    current `structure` keys.
    """

    def __hash__(self) -> int:
        """
        Return a hash derived from the ordered Hilbert-space basis structure.

        [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] uses the
        same structure-based hash as
        [`StateSpace`][qten.symbolics.state_space.StateSpace]. The hash includes
        each [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] sector and its
        stored integer basis index in insertion order. Consequently, two
        Hilbert spaces with the same sectors but different basis order hash
        differently.

        Returns
        -------
        int
            Hash value for the ordered `(U1Basis, index)` mapping.
        """
        return StateSpace.__hash__(self)

    @staticmethod
    def new(itr: Iterable[U1Basis]) -> "HilbertSpace":
        """
        Build a [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] from an ordered basis iterable.

        Parameters
        ----------
        itr : Iterable[U1Basis]
            Basis elements to insert into the space in iteration order. Each
            element is assigned its zero-based sector index according to its
            position in `itr`.

        Returns
        -------
        HilbertSpace
            A new space whose `structure` maps each basis element from `itr` to
            its enumerated sector index.

        Notes
        -----
        Later duplicate keys in `itr` overwrite earlier entries in the backing
        `OrderedDict`, while preserving insertion order for the surviving key.
        """
        structure: OrderedDict[U1Basis, int] = OrderedDict()
        for i, el in enumerate(itr):
            structure[el] = i
        return HilbertSpace(structure=structure)

    def __str__(self) -> str:
        """
        Return a compact Hilbert-space summary.

        Returns
        -------
        str
            Summary containing total dimension and sector count.
        """
        return f"HilbertSpace(dim={self.dim}, sectors={len(self.structure)})"

    def __repr__(self) -> str:
        """
        Return a multiline representation of indexed basis sectors.

        Returns
        -------
        str
            Summary header plus one line per basis sector. Empty spaces are
            marked as `<empty>`.
        """
        if not self.structure:
            return f"{self}: <empty>"

        body = "\n".join(
            f"\t{n}: {idx} {str(el)}"
            for n, (el, idx) in enumerate(self.structure.items())
        )
        return f"{self}:\n{body}"

    def lookup(self, query: Dict[Type[Any], Any]) -> U1Basis:
        """
        Return the unique element that exactly matches all typed-irrep query entries.

        Parameters
        ----------
        query : Dict[Type[Any], Any]
            Mapping from irrep runtime type to expected irrep value.
            A candidate element matches only if, for every `(T, value)` pair,
            it contains an irrep of exact type `T` and `irrep == value`.

        Returns
        -------
        U1Basis
            The unique matching element.

        Raises
        ------
        ValueError
            If no element matches, if multiple elements match, or if `query` is empty.
        """
        if not query:
            raise ValueError("lookup query cannot be empty.")

        matches: list[U1Basis] = []
        for el in self.elements():
            is_match = True
            for T, expected in query.items():
                try:
                    actual = el.irrep_of(T)
                except ValueError:
                    is_match = False
                    break
                if actual != expected:
                    is_match = False
                    break
            if is_match:
                matches.append(el)

        if not matches:
            raise ValueError(f"No state found for query={query}.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple states found for query={query}; expected a unique match."
            )
        return matches[0]

    def group(
        self, **groups: Union[Callable[[U1Basis], bool], Any]
    ) -> FrozenDict[str, "HilbertSpace"]:
        """
            Group basis elements into labeled subspaces.

        Supported selectors
        -------------------
        A `Callable[[U1Basis], bool]` selector includes states for which the
        predicate returns `True`. An irrep-object selector includes states whose
        irrep of the same exact runtime type equals the selector.

            Parameters
            ----------
            **groups : Union[Callable[[U1Basis], bool], Any]
                Label-to-selector mapping used to build grouped subspaces.

            Returns
            -------
            FrozenDict[str, HilbertSpace]
                Frozen mapping from labels to grouped [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] values.
                Each grouped subspace is sorted in ascending [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] order.

            Raises
            ------
            ValueError
                If two selectors include the same basis state.
        """
        elements = self.elements()

        grouped_by_label: OrderedDict[str, HilbertSpace] = OrderedDict()
        grouped_union: set[U1Basis] = set()

        for label, selector in groups.items():
            if callable(selector):
                pred = cast(Callable[[U1Basis], bool], selector)
                selected = tuple(el for el in elements if pred(el))
            else:
                target = selector
                target_type = type(target)
                selected_list: list[U1Basis] = []
                for el in elements:
                    try:
                        if el.irrep_of(target_type) == target:
                            selected_list.append(el)
                    except ValueError:
                        continue
                selected = tuple(selected_list)
            grouped_elements = tuple(sorted(selected))
            overlap = tuple(el for el in grouped_elements if el in grouped_union)
            if overlap:
                raise ValueError(
                    f"grouped spans overlap: state {overlap[0]!r} appears in multiple groups (including {label!r})."
                )

            grouped_union.update(grouped_elements)
            grouped_by_label[label] = HilbertSpace.new(grouped_elements)

        return FrozenDict(grouped_by_label)

    def group_by(self, *T: Type) -> Tuple["HilbertSpace", ...]:
        """
        Form groups by matching irrep types keyed by the irreps.

        This function will try to group the basis by the irrep in order specified by the types in `T`.

        Parameters
        ----------
        *T : Type
            The irrep types to group by. The grouping will be performed in the order of the types specified.
            For example [`group_by(A, B)`][qten.symbolics.hilbert_space.HilbertSpace.group_by] will group the basis by `(A, B)`, and
            basis states with the same irrep of `A` and `B` will be in the same
            grouped subspace.

        Returns
        -------
        Tuple[HilbertSpace, ...]
            Grouped subspaces in the order their irrep keys first appear in the
            current basis order.
        """
        keys: OrderedDict[Tuple[Any, ...], None] = OrderedDict()
        for el in self.elements():
            keys[tuple(el.irrep_of(t) for t in T)] = None

        def make_selector(key: Tuple[Any, ...]) -> Callable[[U1Basis], bool]:
            def selector(el: U1Basis) -> bool:
                return all(el.irrep_of(t) == target for t, target in zip(T, key))

            return selector

        selectors: OrderedDict[str, Callable[[U1Basis], bool]] = OrderedDict()
        for i, key in enumerate(keys):
            selectors[f"_{i}"] = make_selector(key)

        result = self.group(**selectors)
        return tuple(result[label] for label in selectors)

    def is_homogeneous(self) -> bool:
        """
        Check if this [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] is homogeneous, i.e., all basis states share the same irrep types.

        Returns
        -------
        bool
            True if all basis states in this [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] have the same set of irrep types, `False` otherwise.
        """
        elements = self.elements()
        if not elements:  # An empty Hilbert space is considered homogeneous.
            return True
        irrep_types = elements[0].repr_types()
        for el in elements[1:]:
            if el.repr_types() != irrep_types:
                return False
        return True

    def canonical_basis_types(self) -> Tuple[Type[Any], ...]:
        """
        Return the canonical irrep-type order shared by this [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] basis.

        A homogeneous Hilbert space has one consistent irrep-type layout across all
        basis states (for example `(int, str)` for every element). This method
        returns that layout in canonical order.

        For an empty space, the result is the empty tuple `()`.

        Returns
        -------
        Tuple[Type[Any], ...]
            The canonical irrep-type sequence of the basis representation.

        Raises
        ------
        ValueError
            If this space is not homogeneous, i.e. basis states do not share the
            same canonical irrep-type order.
        """
        if not self.is_homogeneous():
            raise ValueError(
                "Cannot get basis irrep types of a non-homogeneous HilbertSpace."
            )
        elements = self.elements()
        if not elements:
            return ()
        return elements[0].repr_types()

    def irrep_of(self, T: Type[_IrrepType]) -> Tuple[_IrrepType, ...]:
        """
        Return the irrep of type `T` for each basis state in this space.

        This is the [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] analogue of `U1Basis.irrep_of(T)`, lifted
        across the ordered basis of the space.

        Parameters
        ----------
        T : Type[_IrrepType]
            Concrete irrep type to retrieve.

        Returns
        -------
        Tuple[_IrrepType, ...]
            The irrep instance of type `T` for each basis state, in basis order.

        Raises
        ------
        ValueError
            If any basis state does not contain an irrep of type `T`.
        """
        return tuple(el.irrep_of(T) for el in self.elements())

    def factorize(
        self, *irrep_types: Tuple[Type, ...], coef_on: Optional[int] = None
    ) -> StateSpaceFactorization:
        r"""
        Factorize a homogeneous Hilbert space into irrep-type tensor factors.

        Each argument in `irrep_types` defines one output factor as a tuple of
        irrep types to group together. Across all groups, every irrep type in
        this space must appear exactly once. The result describes both the
        factor spaces and the basis reindexing needed to move between the
        original basis order and the factorized tensor-product structure.

        Mathematical convention
        -----------------------
        For a homogeneous basis whose labels split into groups \(a\) and \(b\),
        factorization checks that the basis is a complete Cartesian product:
        \(\mathcal{H} \cong \mathcal{H}_a \otimes \mathcal{H}_b\), with
        \(|a_i,b_j\rangle \leftrightarrow |a_i\rangle \otimes |b_j\rangle\).
        More generally, the requested `irrep_types` define factors
        \(\mathcal{H}_0,\ldots,\mathcal{H}_{m-1}\). The returned `align_dim`
        is the original basis reordered so that its flattened order matches
        `factorized[0] @ ... @ factorized[m-1]` in code.

        Notes
        -----
        The input space must be homogeneous, meaning every
        [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] element has the same
        canonical irrep-type layout. The basis must also be a complete
        Cartesian product for the requested groups. For example, if factors are
        `(int,)` and `(str,)`, then every observed integer label must appear
        with every observed string label.

        `coef_on` controls which output factor receives the original
        [`U1Basis.coef`][qten.symbolics.hilbert_space.U1Basis.coef]. All other
        factors are built with coefficient `1`. This is valid only when the
        chosen factor uniquely determines each original coefficient.

        Parameters
        ----------
        *irrep_types : Tuple[Type, ...]
            Factor specification. Each tuple is one output factor containing
            the irrep types assigned to that factor. Every irrep type present
            in this homogeneous space must appear exactly once across all
            factor specifications.
        coef_on : Optional[int], optional
            Index of the factor that inherits the original `U1Basis.coef`.
            `None` defaults to the leftmost factor (`0`). Negative indices are
            interpreted using normal Python indexing. All other factors are
            built with coefficient `1`.

        Returns
        -------
        StateSpaceFactorization
            Factorization metadata. `factorized` contains the output factor
            spaces in the same order as `irrep_types`. `align_dim` is a
            permutation of the original Hilbert space whose flattened order is
            compatible with reshaping into the tensor product of `factorized`.

        Raises
        ------
        ValueError
            If the space is not homogeneous.
        ValueError
            If the space is empty and non-empty factor groups were requested.
        ValueError
            If any requested factor group is empty.
        ValueError
            If an irrep type is missing, duplicated, or not present in the
            homogeneous basis.
        ValueError
            If `coef_on` is out of range.
        ValueError
            If the chosen `coef_on` factor does not determine a unique
            coefficient.
        ValueError
            If the basis is not a complete Cartesian product for the requested
            groups.

        Examples
        --------
        Suppose a Hilbert-space basis is labeled by two independent pieces of
        information: a site index (`int`) and an orbital label (`str`). The
        four basis states below form a complete product of two sites and two
        orbitals:

        ```python
        import sympy as sy
        from qten.symbolics import HilbertSpace, U1Basis

        space = HilbertSpace.new(
            U1Basis(sy.Integer(1), (i, label))
            for i in (1, 2)
            for label in ("a", "b")
        )

        factorization = space.factorize((int,), (str,))
        ```

        The first factor contains the unique site labels. The second factor
        contains the unique orbital labels:

        ```python
        site_factor, orbital_factor = factorization.factorized

        assert tuple(state.base for state in site_factor.elements()) == (
            (1,),
            (2,),
        )
        assert tuple(state.base for state in orbital_factor.elements()) == (
            ("a",),
            ("b",),
        )
        ```

        `align_dim` records the original basis order arranged so it can be
        reshaped as `site_factor @ orbital_factor`. In this example the input
        was already in product order, so `align_dim` has the same basis labels:

        ```python
        assert tuple(state.base for state in factorization.align_dim.elements()) == (
            (1, "a"),
            (1, "b"),
            (2, "a"),
            (2, "b"),
        )
        ```

        Factorization fails if the space is not homogeneous, if the requested
        groups omit or add irrep types, if `coef_on` is out of range, if
        coefficient assignment is ambiguous, or if the basis is not a complete
        Cartesian product for the requested groups. For example, a basis with
        `(1, "a")`, `(1, "b")`, `(2, "a")`, `(2, "b")`, and `(2, "c")` cannot
        factorize by `(int,)` and `(str,)` because the product would also
        require `(1, "c")`.
        """
        if not self.is_homogeneous():
            raise ValueError("Cannot factorize a non-homogeneous HilbertSpace.")

        elements = self.elements()
        if not elements:
            if irrep_types:
                raise ValueError(
                    "Cannot factorize an empty HilbertSpace with non-empty irrep groups."
                )
            return StateSpaceFactorization(factorized=(), align_dim=self)

        if any(not group for group in irrep_types):
            raise ValueError("Each irrep group in factorize must be non-empty.")

        if coef_on is None:
            coef_factor = 0
        else:
            coef_factor = coef_on
            if coef_factor < 0:
                coef_factor += len(irrep_types)
            if coef_factor < 0 or coef_factor >= len(irrep_types):
                raise ValueError(
                    f"`coef_on` index {coef_on} is out of range for {len(irrep_types)} factors."
                )

        canonical_types = elements[0].repr_types()
        requested_types = tuple(T for group in irrep_types for T in group)
        requested_set = set(requested_types)
        canonical_set = set(canonical_types)

        if len(requested_set) != len(requested_types):
            raise ValueError(
                "Each irrep type must appear exactly once in `irrep_types`."
            )

        missing_types = tuple(
            T.__name__ for T in canonical_types if T not in requested_set
        )
        extra_types = tuple(
            T.__name__ for T in requested_types if T not in canonical_set
        )
        if missing_types or extra_types:
            details = []
            if missing_types:
                details.append(f"missing types: {missing_types}")
            if extra_types:
                details.append(f"extra types: {extra_types}")
            raise ValueError(
                f"`irrep_types` does not match space irrep types ({', '.join(details)})."
            )

        factor_keys: list[Tuple[Tuple[Any, ...], ...]] = []
        factorized: list[HilbertSpace] = []
        for group_idx, group in enumerate(irrep_types):
            keys: OrderedDict[Tuple[Any, ...], sy.Expr] = OrderedDict()
            for el in elements:
                key = tuple(el.irrep_of(T) for T in group)
                coef = el.coef if group_idx == coef_factor else sy.Integer(1)
                if key in keys and keys[key] != coef:
                    raise ValueError(
                        "Requested factorization is not valid: "
                        "the requested `coef_on` factor does not determine a unique coefficient."
                    )
                keys[key] = coef
            grouped_basis = tuple(
                U1Basis(coef, tuple(key)) for key, coef in keys.items()
            )
            factor_keys.append(tuple(keys.keys()))
            factorized.append(HilbertSpace.new(grouped_basis))

        # Map each basis state to its grouped-irrep key tuple.
        combo_to_element: OrderedDict[Tuple[Tuple[Any, ...], ...], U1Basis] = (
            OrderedDict()
        )
        for el in elements:
            combo = tuple(tuple(el.irrep_of(T) for T in group) for group in irrep_types)
            combo_to_element[combo] = el

        expected_size = 1
        for group_keys in factor_keys:
            expected_size *= len(group_keys)
        if expected_size != len(combo_to_element):
            raise ValueError(
                "Requested factorization is not valid: basis is not a complete Cartesian product."
            )

        # Build reshape-compatible order: first factor varies slowest, last varies fastest.
        align_elements: list[U1Basis] = []
        for combo in product(*factor_keys):
            if combo not in combo_to_element:
                raise ValueError(
                    "Requested factorization is not valid: basis is not a complete Cartesian product."
                )
            align_elements.append(combo_to_element[combo])

        return StateSpaceFactorization(
            factorized=tuple(factorized),
            align_dim=HilbertSpace.new(align_elements),
        )

    @override
    def tensor_product(self, other: "HilbertSpace") -> "HilbertSpace":
        r"""
        Build the tensor-product space of this space with another [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].

        The resulting basis is generated by taking every ordered pair
        `(a, b)` from `self.elements()` and `other.elements()`, then forming the
        basis-level tensor product `a @ b` for each pair.

        Mathematically, if
        \(\mathcal{H}_L = \mathrm{span}\{|a_i\rangle\}\) and
        \(\mathcal{H}_R = \mathrm{span}\{|b_j\rangle\}\), then the result is
        ordered as \(\mathcal{H}_L \otimes \mathcal{H}_R
        = \mathrm{span}\{|a_i\rangle \otimes |b_j\rangle\}_{i,j}\).
        Ordering follows `itertools.product(self.elements(), other.elements())`:
        basis elements from `self` vary slowest, and basis elements from `other`
        vary fastest. This deterministic order is important for tensor reshape
        and alignment logic that depends on stable axis conventions.

        Parameters
        ----------
        other : HilbertSpace
            Right-hand tensor factor.

        Returns
        -------
        HilbertSpace
            A new space whose basis spans `self ⊗ other`.
        """
        elements = []
        for a, b in product(self.elements(), other.elements()):
            elements.append(a @ b)
        return HilbertSpace.new(elements)

    @override
    def rays(self) -> "HilbertSpace":
        r"""
        Return the Hilbert space obtained by ray-normalizing every basis state.

        The output keeps the same basis order after converting each
        [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] element to
        `element.rays()`. This removes U(1) coefficients from the symbolic
        basis labels while preserving the projective sector structure.

        In formula form:
        \(\{c_i|\rho_i\rangle\}_i \longmapsto \{|\rho_i\rangle\}_i\).

        Returns
        -------
        HilbertSpace
            Hilbert space built from the ray representatives of this space's
            basis elements.

        Examples
        --------
        ```python
        import sympy as sy
        from sympy import ImmutableDenseMatrix
        from qten.geometries import AffineSpace, Offset
        from qten.symbolics import HilbertSpace, U1Basis

        affine = AffineSpace(ImmutableDenseMatrix.eye(1))
        r0 = Offset(ImmutableDenseMatrix([0]), affine)
        hilbert = HilbertSpace.new((U1Basis(sy.Integer(5), (r0,)),))
        ray_space = hilbert.rays()

        assert ray_space.elements()[0].coef == 1
        ```
        """
        return HilbertSpace.new(el.rays() for el in self)

    def cross_gram(
        self, another: "HilbertSpace", *, device: Optional[Device] = None
    ) -> Tensor:
        r"""
        Build the cross-Gram overlap matrix between this basis and another basis.

        Matrix entries are computed from concrete basis overlaps
        \(G_{ij} = \langle \mathrm{self}_i \mid \mathrm{another}_j \rangle\),
        so any nontrivial U(1) irrep phase in basis vectors is encoded in
        `data`.

        Output dimension convention
        ---------------------------
        The returned tensor uses dims `(self, another.rays())`.
        The target (column) dimension is intentionally replaced by its canonical
        ray representative (phase removed)
        so the codomain metadata is gauge-fixed, while phase information remains
        in matrix elements.

        This is equivalent to using a representative of the same ray space
        (projective Hilbert space) for the target basis labels.

        Parameters
        ----------
        another : HilbertSpace
            Right-hand basis supplying ket states.
        device : Optional[Device], optional
            Device on which to allocate the returned tensor data.

        Returns
        -------
        Tensor
            Matrix tensor with dims `(self, another.rays())`.
        """
        span = U1Span(cast(Tuple[U1Basis, ...], self.elements()))
        new_span = U1Span(cast(Tuple[U1Basis, ...], another.elements()))
        irrep = span.cross_gram(new_span)
        precision = get_precision_config()
        torch_device = device.torch_device() if device is not None else None
        data = torch.from_numpy(
            np.asarray(irrep.tolist(), dtype=precision.np_complex)
        ).to(device=torch_device, dtype=precision.torch_complex)
        return Tensor(data=data, dims=(self, another.rays()))


@Operable.__matmul__.register
def _(space: HilbertSpace, basis: U1Basis) -> HilbertSpace:
    return HilbertSpace.new(element @ basis for element in space.elements())


@Operable.__matmul__.register
def _(basis: U1Basis, space: HilbertSpace) -> HilbertSpace:
    return HilbertSpace.new(basis @ element for element in space.elements())


@U1Basis.add_conversion(StateSpace)
def _u1basis_to_hilbertspace(basis: U1Basis) -> StateSpace:
    """Convert a [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] to a [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] containing only that basis state."""
    return HilbertSpace.new((basis,))


# Support conversion to HilbertSpace using `basis.convert(HilbertSpace)`.
U1Basis.add_conversion(HilbertSpace)(
    cast(Callable[[U1Basis], HilbertSpace], _u1basis_to_hilbertspace)
)


@U1Span.add_conversion(StateSpace)
def _u1span_to_hilbertspace(span: U1Span) -> StateSpace:
    """Convert a [`U1Span`][qten.symbolics.hilbert_space.U1Span] to a [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] containing the span's basis states."""
    return HilbertSpace.new(span.span)


U1Span.add_conversion(HilbertSpace)(
    cast(Callable[[U1Span], HilbertSpace], _u1span_to_hilbertspace)
)


@HilbertSpace.add_conversion(HilbertSpace)
def _(v: HilbertSpace) -> HilbertSpace:
    """Identity conversion for [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]."""
    return v


@same_rays.register
def _(a: HilbertSpace, b: HilbertSpace) -> bool:
    return set(m.rays() for m in a.structure.keys()) == set(
        m.rays() for m in b.structure.keys()
    )


_T = TypeVar("_T")


class Opr(Functional, Operable, ABC):
    """
    A composable operator that acts on observable-compatible objects.

    `Operator` combines two core behaviors:

    The first behavior is [`Functional`][qten.abstracts.Functional] dispatch
    and chaining. Implementations are registered via `Functional.register` for
    pairs of `(input_type, operator_subclass)`. At runtime, `invoke` resolves
    the function chain for the concrete input object and executes each function
    in order.

    The second behavior is [`Operable`][qten.abstracts.Operable]
    matrix-application syntax. Because `Operator` is
    [`Operable`][qten.abstracts.Operable], it participates in the overloaded
    `@` operator. This module defines `Operable.__matmul__(Operator, U1Basis)`,
    so `op @ value` applies the operator and returns only the transformed value
    component.

    Conceptually, an operator application returns either a transformed object
    of the same runtime type as the input, or
    [`Multiple(coef, transformed_object)`][qten.symbolics.Multiple] when the
    action also contributes a scalar prefactor.

    The [`Multiple`][qten.symbolics.Multiple] form is used when the operator naturally decomposes into a
    scalar factor times an object in the same representation family. Typical
    examples include phase factors, characters, gauge coefficients, and other
    symbolic amplitudes that should stay factored instead of being folded
    directly into the transformed object.

    Implementation Guidelines
    -------------------------
    A registered handler should return a plain transformed object when the
    operator acts without producing an extra scalar factor. It should return
    [`Multiple(coef, value)`][qten.symbolics.Multiple] when the action produces
    a scalar prefactor that should remain explicit. In both cases, the
    underlying transformed value must stay in the same representation family as
    the input: plain returns should have `type(result) is type(input)` or a
    compatible subclass, while multiple returns should have
    `type(result.base) is type(input)` or a compatible subclass.

    `Opr.invoke` automatically lifts operators over
    [`Multiple`][qten.symbolics.Multiple]. If `op(base) -> y`, then
    `op(Multiple(c, base)) -> Multiple(c, y)`. If
    `op(base) -> Multiple(c2, y)`, the coefficients are multiplied to produce
    [`Multiple(simplify(c * c2), y)`][qten.symbolics.Multiple].

    For symbolic container inputs such as
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis],
    [`U1Span`][qten.symbolics.hilbert_space.U1Span], and
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], the generic
    [`Opr`][qten.symbolics.hilbert_space.Opr] lifting in this module expects the
    final result to stay in-kind and not return
    [`Multiple`][qten.symbolics.Multiple] at the top level. Scalar factors
    should instead be attached to the contained irreps/components and
    accumulated into the container's internal coefficient structure.

    This module provides generic [`Opr`][qten.symbolics.hilbert_space.Opr]
    registrations for [`U1Basis`][qten.symbolics.hilbert_space.U1Basis],
    [`U1Span`][qten.symbolics.hilbert_space.U1Span], and
    [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]. Subclasses
    inherit those handlers through [`Functional`][qten.abstracts.Functional]
    MRO fallback unless they define more specific registrations. If no
    registration exists for `(type(input), type(operator))` after MRO fallback
    on both the input type and the operator type, `Functional.invoke` raises
    `NotImplementedError`.

    Usage Pattern
    -------------
    Define an `Operator` subclass, register behavior with
    `@YourOperatorSubclass.register(InputType)`, and apply it either as
    `out = op(input_obj)` to receive the transformed value or as
    `out = op @ input_obj` using infix syntax.

    Return-value Guidance
    ---------------------
    Return `value` when the operator only changes the representation/object
    itself. Return [`Multiple(coef, value)`][qten.symbolics.Multiple] when the
    operator contributes a symbolic scalar prefactor that should remain
    factored.

    In particular, for atomic irreps or basis-function-like objects, returning
    [`Multiple`][qten.symbolics.Multiple] is often appropriate. For higher-level symbolic containers such
    as [`U1Basis`][qten.symbolics.hilbert_space.U1Basis], [`U1Span`][qten.symbolics.hilbert_space.U1Span], and [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], prefer returning the same
    container type directly and let the generic lifting logic accumulate scalar
    factors internally.
    """

    @override
    def invoke(  # type: ignore[override]
        self, v: _T, **kwargs
    ) -> Union[_T, Multiple[_T]]:
        """
        Apply the operator while preserving QTen's symbolic output invariants.

        Parameters
        ----------
        v : _T
            Input object to transform.
        **kwargs : Any
            Extra keyword arguments forwarded to the resolved registration.

        Returns
        -------
        _T | Multiple[_T]
            Transformed object, or a factored result carrying an explicit
            scalar coefficient.

        Raises
        ------
        AssertionError
            If a registered implementation returns a value outside the expected
            same-type / `Multiple[same-type]` contract.
        """
        if type(v) is Multiple:
            result = super().invoke(v.base, **kwargs)
            if type(result) is Multiple:
                return Multiple((v.coef * result.coef).simplify(), result.base)
            return Multiple(v.coef, result)
        result = super().invoke(v, **kwargs)
        if isinstance(v, (U1Basis, U1Span, HilbertSpace)):
            assert type(result) is not Multiple, (
                f"Operator {type(self)} acting on {type(v).__name__} should not yield a Multiple!"
            )
        assert isinstance(result, type(v)) or (
            type(result) is Multiple and isinstance(result.base, type(v))
        ), (
            f"Operator {type(self)} acting on {type(v).__name__} should yield same typed object"
            f"or Multiple[{type(v).__name__}]"
        )
        return result


@Operable.__matmul__.register
def _(o: Opr, v: Operable):
    return o(v)


@Opr.register(U1Basis)
def _(o: Opr, psi: U1Basis) -> U1Basis:
    new_coef: sy.Expr = psi.coef
    new_base: Tuple[Any, ...] = tuple()
    for rep in psi.base:
        ret = o(rep) if o.allows(rep) else rep
        if isinstance(ret, Multiple):
            new_coef *= ret.coef
            rep = ret.base
        else:
            rep = ret
        new_base += (rep,)
    return U1Basis(new_coef, new_base)


@Opr.register(U1Span)
def _(o: Opr, v: U1Span) -> U1Span:
    new_span: Tuple[U1Basis, ...] = tuple(cast(U1Basis, o @ psi) for psi in v.span)
    return U1Span(new_span)


@Opr.register(HilbertSpace)
def _(o: Opr, h: HilbertSpace) -> HilbertSpace:
    new_h = HilbertSpace.new(cast(U1Basis, o @ el) for el in h)
    return new_h


@same_rays.register
def _(a: U1Basis, b: U1Basis) -> bool:
    """Check if two [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] define the same ray."""
    return a.rays() == b.rays()


@dataclass(frozen=True)
class FuncOpr(Generic[_IrrepType], Opr):
    """
    Symbolic operator induced by applying a Python callable to one irrep type.

    [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] lifts a plain callable acting on a single irrep/component into an
    [`Opr`][qten.symbolics.hilbert_space.Opr] that acts on the symbolic Hilbert-space objects defined in this module.
    The operator is parameterized by:

    `T` is the concrete runtime type of the irrep/component to target. `func`
    is a callable mapping `T -> T | Multiple[T]`.

    Registered actions
    ------------------
    [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] defines a specialized registration on:

    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]. The handler finds the
    unique irrep in `psi.base` whose exact runtime type is `T`, applies `func`
    to it, and rebuilds the basis state with that component replaced.

    For [`U1Span`][qten.symbolics.hilbert_space.U1Span] and [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace], [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] relies on the inherited generic
    [`Opr`][qten.symbolics.hilbert_space.Opr] lifting in this module, which maps the operator over contained
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] elements.

    Semantics
    ---------
    This class is intended for structure-preserving symbolic rewrites such as
    "replace each [`Offset`][qten.geometries.spatials.Offset] by `Offset.fractional()`" or "apply a transformation
    to each irrep of a basis state". It is not a general-purpose map over
    arbitrary Python objects: dispatch exists only for the symbolic container
    types registered below.

    The callable `func` is expected to return either a transformed object of
    the same concrete type `T`, or
    [`Multiple(coef, value)`][qten.symbolics.Multiple] where `value` has type
    `T`.

    Return plain `T` when the transformation only changes the irrep/component
    itself. Return [`Multiple`][qten.symbolics.Multiple] when the transformation also contributes an
    explicit scalar factor, such as a phase or gauge coefficient, that should
    be accumulated symbolically rather than absorbed into the object.

    [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] does not itself validate the semantic correctness of `func`, but
    the surrounding symbolic code assumes the transformation stays within the
    same representation family.

    Interaction with `@`
    --------------------
    Because [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] is an [`Opr`][qten.symbolics.hilbert_space.Opr], it participates in the overloaded `@`
    syntax:

    `f @ x` applies the operator to an object `x`. `f @ g` forms a
    [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr].

    With the current [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr] semantics, operator composition follows the
    standard algebraic order:

    `(f @ g) @ x == f(g(x))`.

    So `f @ g` means "apply `g`, then apply `f`".

    Interaction with [`Multiple`][qten.symbolics.Multiple]
    ---------------------------
    If `v` is [`Multiple(coef, base)`][qten.symbolics.Multiple], then `f(v)` applies `f` to `base`.

    If the result is a plain object `base'`, the output is
    [`Multiple(coef, base')`][qten.symbolics.Multiple]. If the result is
    [`Multiple(coef', base')`][qten.symbolics.Multiple], the coefficients are
    multiplied and simplified, producing
    [`Multiple(simplify(coef * coef'), base')`][qten.symbolics.Multiple].

    This behavior is provided centrally by `Opr.invoke`, so [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] inherits
    the same coefficient-lifting semantics as every other operator subclass.

    Interaction with Symbolic Containers
    ------------------------------------
    [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] treats [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] specially: it targets the unique irrep of type
    `T` via `psi.irrep_of(T)` rather than iterating over all components with
    [`allows(...)`][qten.abstracts.Functional.allows]. This is why [`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] keeps a dedicated [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]
    registration even though [`U1Span`][qten.symbolics.hilbert_space.U1Span] and [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] can be handled by the
    generic [`Opr`][qten.symbolics.hilbert_space.Opr] lifting.

    Examples
    --------
    Use [`FuncOpr(Offset, Offset.fractional)`][qten.symbolics.hilbert_space.FuncOpr] to normalize every [`Offset`][qten.geometries.spatials.Offset]
    appearing inside a [`U1Basis`][qten.symbolics.hilbert_space.U1Basis], [`U1Span`][qten.symbolics.hilbert_space.U1Span], or [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].

    In particular:

    `fractional @ psi` rewrites one basis state. `fractional @ space` rewrites
    every basis state in a Hilbert space. `fractional @ t @ space` means
    `fractional(t(space))`.

    Attributes
    ----------
    T : Type[_IrrepType]
        Exact runtime type of the irrep/component targeted by this operator.
    func : Callable[[_IrrepType], _IrrepType | Multiple[_IrrepType]]
        Callable applied to the targeted irrep.
    """

    T: Type[_IrrepType]
    """
    Exact runtime type of the irrep/component targeted by this operator when
    acting on symbolic basis states.
    """
    func: Callable[[_IrrepType], Union[_IrrepType, Multiple[_IrrepType]]]
    """
    Callable applied to the targeted irrep. It may return either a transformed
    irrep directly or a [`Multiple`][qten.symbolics.Multiple] carrying an
    additional scalar factor.
    """


@FuncOpr.register(U1Basis)
def _(f: FuncOpr, psi: U1Basis) -> U1Basis:
    irrep = psi.irrep_of(f.T)
    ret = f.func(irrep)
    if type(ret) is Multiple:
        new_psi = psi.replace(ret.base)
        return replace(new_psi, coef=(new_psi.coef * ret.coef).simplify())
    new_irrep = ret
    new_psi = psi.replace(new_irrep)
    return new_psi


@same_rays.register
def _(a: U1Span, b: U1Span) -> bool:
    return set(a.rays().span) == set(b.rays().span)


@Operable.__eq__.register
def _(a: U1Basis, b: U1Basis) -> bool:
    return a.coef == b.coef and a.base == b.base


@Operable.__matmul__.register
def _(a: U1Basis, b: U1Basis) -> U1Basis:
    """
    Combine two basis states by symbolic tensor product.

    The tensor product concatenates the ordered irrep tuples and multiplies the
    symbolic U(1) coefficients. Concretely, non-empty operands produce
    `U1Basis(simplify(a.coef * b.coef), a.base + b.base)`. If `a.base` is
    empty, `b` is returned unchanged. If `b.base` is empty, `a` is returned
    unchanged.

    The concatenated `base` tuple preserves tensor-product construction order:
    all labels from `a` come before all labels from `b`. During construction,
    `U1Basis.__post_init__` separately computes `rep` by sorting that `base`
    tuple by fully-qualified runtime type name (`module.qualname`). Therefore
    `base` records how the tensor product was built, while `rep` is the
    canonical type-sorted representation used by type-based operations and
    comparisons.

    The constructed result is subject to the normal
    [`U1Basis`][qten.symbolics.hilbert_space.U1Basis] validation rules. In
    particular, the combined basis may not contain two irreps of the same
    concrete runtime type.

    Parameters
    ----------
    a : U1Basis
        Left tensor-product factor. Its irreps appear first in the output
        `base` tuple.
    b : U1Basis
        Right tensor-product factor. Its irreps appear after `a.base` in the
        output `base` tuple.

    Returns
    -------
    U1Basis
        Tensor-product basis state with multiplied coefficient and concatenated
        irrep labels.

    Raises
    ------
    ValueError
        If the concatenated irrep labels violate the `U1Basis` uniqueness
        invariant.
    """
    if not a.base:
        return b
    if not b.base:
        return a
    return U1Basis((a.coef * b.coef).simplify(), a.base + b.base)


@Operable.__or__.register
def _(a: U1Basis, b: U1Basis) -> U1Span:
    if a == b:
        return U1Span((a,))
    return U1Span((a, b))


@Operable.__or__.register
def _(span: U1Span, state: U1Basis) -> U1Span:
    if state in span.span:
        return span
    return U1Span(span.span + (state,))


@Operable.__or__.register
def _(state: U1Basis, span: U1Span) -> U1Span:
    if state in span.span:
        return span
    return U1Span((state,) + span.span)


@Operable.__or__.register
def _(a: U1Span, b: U1Span) -> U1Span:
    existing = set(a.span)
    new_states = tuple(s for s in b.span if s not in existing)
    return U1Span(a.span + new_states)


@Operable.__sub__.register
def _(a: U1Span, b: U1Span) -> U1Span:
    b_elements = set(b.span)
    new_states = tuple(s for s in a.span if s not in b_elements)
    return U1Span(new_states)


@Operable.__and__.register
def _(a: U1Span, b: U1Span) -> U1Span:
    b_elements = set(b.span)
    new_states = tuple(s for s in a.span if s in b_elements)
    return U1Span(new_states)


@Operable.__or__.register
def _(space: HilbertSpace, state: U1Basis) -> HilbertSpace:
    if state in space.structure:
        return space
    return HilbertSpace.new((*space.elements(), state))


@Operable.__or__.register
def _(state: U1Basis, space: HilbertSpace) -> HilbertSpace:
    if state in space.structure:
        return space
    return HilbertSpace.new((state, *space.elements()))


@dataclass(frozen=True)
class ComposedOpr(Opr):
    """
    Finite composition of symbolic operators.

    [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr] stores an ordered tuple of [`Opr`][qten.symbolics.hilbert_space.Opr] instances and represents
    their algebraic composition. It is produced automatically by operator
    multiplication between operators:

    `a @ b` returns
    [`ComposedOpr((a, b))`][qten.symbolics.hilbert_space.ComposedOpr].

    Application order
    -----------------
    Although the tuple is stored as `(a, b)`, application follows standard
    operator-composition order rather than pipeline order:

    `(a @ b) @ x == a(b(x))`.

    More generally, if `ops == (o1, o2, ..., on)`, then applying the composed
    operator to `x` yields:

    `o1(o2(...(on(x))...))`.

    Operationally, this is implemented by applying the stored operators in
    reverse order at invocation time.

    This matches the usual reading of matrix multiplication and function
    composition, where the rightmost operator acts first.

    Flattening behavior
    -------------------
    Repeated composition is flattened structurally:

    `(a @ b) @ c` stores `(a, b, c)`, and `a @ (b @ c)` stores
    `(a, b, c)`.

    so nested [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr] objects are normalized into a single tuple of
    operators instead of building a deep binary tree. This keeps composition
    associative at the representation level as well as the semantic level.

    Interaction with object application
    ----------------------------------
    As with every [`Opr`][qten.symbolics.hilbert_space.Opr], `composed @ x` applies the operator to `x`. Each step
    is fed into the next one according to the order above.

    For plain objects `x`, the intermediate values are passed directly from one
    operator to the next.

    Interaction with [`Multiple`][qten.symbolics.Multiple]
    ---------------------------
    [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr] also supports [`Multiple(base, coef)`][qten.symbolics.Multiple] inputs. In that case the
    composition acts on `base`, while scalar coefficients returned by
    intermediate operators are accumulated multiplicatively.

    Concretely, if an intermediate step returns a plain object `y`,
    composition continues with `y`. If an intermediate step returns
    [`Multiple(c, y)`][qten.symbolics.Multiple], composition continues with `y`
    and multiplies the running coefficient by `c`.

    The final result is returned as one [`Multiple(total_coef, final_base)`][qten.symbolics.Multiple].

    This is essential for symbolic transformations where some operators carry
    phase factors or representation coefficients.

    Algebraic scope
    ---------------
    [`ComposedOpr`][qten.symbolics.hilbert_space.ComposedOpr] is intended for endomorphism-like symbolic operators in this
    module: operators that map supported symbolic objects back into the same
    symbolic universe. It assumes that composing the stored operators is valid
    on the given input type; if some step has no registered action for the
    intermediate object, dispatch will fail with `NotImplementedError`.

    Examples
    --------
    If `fractional = FuncOpr(Offset, Offset.fractional)` and `t` is an
    [`AbelianOpr`][qten.pointgroups.abelian.AbelianOpr], then:

    `fractional @ t @ space`

    means:

    `fractional(t(space))`

    not:

    `t(fractional(space))`

    Attributes
    ----------
    ops : Tuple[Opr, ...]
        Operators stored in algebraic composition order. The rightmost
        operator is applied first.
    """

    ops: Tuple[Opr, ...]
    """
    Operators stored in algebraic composition order. The rightmost operator is
    applied first when the composed operator is invoked on an object.
    """

    @override
    def invoke(self, v: _T, **kwargs) -> Union[_T, Multiple[_T]]:
        """
        Apply the composed operators in algebraic composition order.

        Parameters
        ----------
        v : _T
            Input object to transform.
        **kwargs : Any
            Extra keyword arguments forwarded to each operator application.

        Returns
        -------
        _T | Multiple[_T]
            Result after applying the stored operators from right to left.
        """
        result = v
        for op in reversed(self.ops):
            result = op(result, **kwargs)
        return result


@Operable.__matmul__.register
def _(a: Opr, b: Opr) -> ComposedOpr:
    return ComposedOpr((a, b))


@Operable.__matmul__.register
def _(a: ComposedOpr, b: Opr) -> ComposedOpr:
    return ComposedOpr((*a.ops, b))


@Operable.__matmul__.register
def _(a: Opr, b: ComposedOpr) -> ComposedOpr:
    return ComposedOpr((a, *b.ops))


@Operable.__matmul__.register
def _(a: ComposedOpr, b: ComposedOpr) -> ComposedOpr:
    return ComposedOpr((*a.ops, *b.ops))
