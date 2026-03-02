from abc import ABC
from copy import copy
from dataclasses import dataclass, replace
from typing import (
    Any,
    Callable,
    Dict,
    Self,
    Tuple,
    Type,
    TypeVar,
    Generic,
    Union,
    NamedTuple,
)
from typing import cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import product

import numpy as np
import torch
import sympy as sy
from multipledispatch import dispatch  # type: ignore[import-untyped]

from .utils import FrozenDict, full_typename
from .abstracts import AbstractKet, Convertible, Operable, Functional, Span, HasUnit
from .spatials import Spatial
from .state_space import StateSpace, StateSpaceFactorization, restructure
from .tensors import Tensor
from .precision import get_precision_config


_IrrepType = TypeVar("_IrrepType")
"""Defines a irreducible representation type."""


@dataclass(frozen=True)  # eq=False, Skipping Operable.__eq__
class Ket(Generic[_IrrepType], AbstractKet[int], Operable, Convertible):
    """
    A single basis label in the Hilbert construction.

    A `Ket` wraps one irreducible-representation object (`irrep`). It does not
    store amplitudes; it is only the symbolic building block used to form larger
    tensor-product states with `@`.

    Notes
    -----
    `Ket` can be converted to `U1Basis` via `ket.convert(U1Basis)`.
    """

    # TODO: In the future if we replace @dispatch operator_xxx with Opeator.register, check if this type defines __lt__ or __gt__
    irrep: _IrrepType

    @override
    def ket(self, another: "Ket[_IrrepType]") -> int:
        """
        Return the overlap of this ket with another ket.

        Parameters
        ----------
        `another` : `Ket[_IrrepType]`
            The ket to compute the overlap with.

        Returns
        -------
        `int`
            `1` if the irreps of this ket and `another` are equal, `0` otherwise.
        """
        return int(self.irrep == another.irrep)


@dispatch(Ket, Ket)
def operator_lt(a: Ket[_IrrepType], b: Ket[_IrrepType]) -> bool:
    return a.irrep < b.irrep  # type: ignore[operator]


@dispatch(Ket, Ket)
def operator_gt(a: Ket[_IrrepType], b: Ket[_IrrepType]) -> bool:
    return a.irrep > b.irrep  # type: ignore[operator]


@dataclass(frozen=True)
class U1Basis(Spatial, AbstractKet[sy.Expr], HasUnit, Convertible):
    """
    Immutable single-particle basis state built from typed irreps.

    `U1Basis` is a symbolic tensor-product state with U(1) irreducible representation
    presented as an ordered tuple of `Ket` objects. Each ket contributes one
    irreducible representation label (`ket.irrep`) to the state. This object is
    intentionally symbolic: it stores basis labels only and does not store amplitudes
    or coefficients.

    A key invariant is enforced at construction time: for each concrete irrep
    type, the state must contain exactly one ket of that type. In other words,
    irrep multiplicities must be unity. This guarantees that type-based updates
    via `replace` are unambiguous and fast.

    Parameters
    ----------
    `irrep`: `sy.Expr`
        The irrep of this state under an recent operation.
    `kets` : `Tuple[Ket[Any], ...]`
        Tuple of kets that defines the state. Input order is canonicalized in
        `__post_init__` by sorting on `full_typename(type(ket.irrep))`.

    Attributes
    ----------
    `irrep`: `sy.Expr`
        The irrep of this state under an recent operation.
    `kets` : `Tuple[Ket[Any], ...]`
        Immutable canonical ket order sorted by concrete irrep type name
        (`module.qualname`).

    Notes
    -----
    - `dim` is always `1`; this type represents one basis vector.
    - Ket order is canonicalized at construction; permutations of the same
      typed irreps produce the same internal `kets` tuple.
    - `replace(irrep)` substitutes the unique ket whose irrep has the same
      concrete runtime type as `irrep`.
    - `@` dispatch overloads combine kets/states into a new `U1Basis`.
    - `|` dispatch overloads build a `U1Span` of distinct `U1Basis` values.
    - Ordering (`<`, `>`) compares, in order: number of irreps, tuple of
      fully-qualified irrep type names (`module.qualname`) from
      `canonical_repr()`, then the canonical irrep-value tuple itself.
      If irrep values of matching types are not orderable, the comparison
      raises from the underlying irrep objects.

    Raises
    ------
    `ValueError`
        Raised in `__post_init__` when any irrep type appears with multiplicity
        different from `1`.
    """

    irrep: sy.Expr
    kets: Tuple[Ket[Any], ...]

    def __post_init__(self) -> None:
        counts: Dict[Type, int] = {}
        for ket in self.kets:
            irrep_type = type(ket.irrep)
            counts[irrep_type] = counts.get(irrep_type, 0) + 1
        non_singletons = {t: c for t, c in counts.items() if c != 1}
        if non_singletons:
            detail = ", ".join(f"{t.__name__}:{c}" for t, c in non_singletons.items())
            raise ValueError(
                "U1Basis allows only irrep with unity multiplicity; "
                f"got multiple non-singleton types ({detail})."
            )
        object.__setattr__(
            self,
            "kets",
            tuple(
                sorted(
                    self.kets,
                    key=lambda ket: full_typename(type(ket.irrep)),
                )
            ),
        )

    @property
    def dim(self) -> int:
        """The dimension of a single particle state is always `1`."""
        return 1

    def replace(self, irrep: Any) -> "U1Basis":
        """
        Return a new state where the irrep of the same concrete type is replaced.

        The method searches this state's ket tuple for the unique ket whose irrep has
        the same *exact* runtime type as `irrep` (using `type(x) is type(y)`), then
        returns a new `U1Basis` with that ket substituted. The original instance is
        not modified.

        Parameters
        ----------
        `irrep` : `Any`
            Replacement irrep instance. Its concrete type must already exist in this
            state exactly once (enforced by `U1Basis.__post_init__`).

        Returns
        -------
        `U1Basis`
            A new `U1Basis` with one ket replaced.

        Raises
        ------
        `ValueError`
            If this state does not contain any ket whose irrep has the same concrete
            type as `irrep`.
        """
        target_type = type(irrep)
        kets = self.kets
        for i, ket in enumerate(kets):
            if type(ket.irrep) is target_type:
                return replace(self, kets=kets[:i] + (Ket(irrep),) + kets[i + 1 :])
        raise ValueError(
            f"U1Basis has no irrep of type {target_type.__name__} to replace."
        )

    def irrep_of(self, T: Type[_IrrepType]) -> _IrrepType:
        """
        Return the unique irrep in this state whose concrete type is `T`.

        This method performs a direct scan over `self.kets` and returns the
        first irrep satisfying `type(ket.irrep) is T`. Because
        `U1Basis.__post_init__` enforces unity multiplicity for each irrep
        type, the match is unique whenever it exists.

        Parameters
        ----------
        `T` : `Type[_IrrepType]`
            Concrete irrep type to retrieve.

        Returns
        -------
        `_IrrepType`
            The irrep instance of type `T` contained in this state.

        Raises
        ------
        `ValueError`
            If no irrep with concrete type `T` exists in this state.

        Notes
        -----
        - Matching uses exact runtime type identity (`type(x) is T`), not
          subclass checks.
        - Runtime is linear in the number of kets (`O(n)`), with no temporary
          mapping allocations.
        """
        for ket in self.kets:
            irrep = ket.irrep
            if type(irrep) is T:
                return cast(_IrrepType, irrep)
        raise ValueError(f"U1Basis {self} has no irrep of type {T.__name__}.")

    @override
    def ket(self, psi: "U1Basis") -> sy.Expr:
        """
        Return the overlap of this state with another state.

        Parameters
        ----------
        `psi` : `U1Basis`
            The state to compute the overlap with.

        Returns
        -------
        `sy.Expr`
            The symbolic overlap of this state with `psi`. If the kets of the
            two states do not match, the overlap is `0`. If the kets match, the
            overlap is the product of this state's irrep and the conjugate of
            `psi`'s irrep.
        """
        if not self.kets == psi.kets:
            return sy.Integer(0)
        return cast(sy.Expr, (sy.conjugate(self.irrep) * psi.irrep).simplify())

    def __str__(self) -> str:
        ket_repr = "⊗".join(
            f"|{irrep_repr}⟩"
            if len(irrep_repr := repr(ket.irrep)) <= 32
            else f"|{type(ket.irrep).__name__}⟩"
            for ket in self.kets
        )
        if self.irrep != sy.Integer(1):
            ket_repr = f"({self.irrep}) * " + ket_repr
        return ket_repr

    def __repr__(self) -> str:
        return self.__str__()

    @override
    def unit(self) -> "U1Basis":
        """Get a new copy from this `U1Basis` with the U(1) irrep being `1`."""
        return replace(self, irrep=sy.Integer(1))

    @lru_cache
    def canonical_repr(self) -> Tuple[Any, ...]:
        """
        Get a canonical representation of this `U1Basis` for hashing and comparison.

        Kets are canonicalized in `U1Basis` init with sorting on
        `full_typename(type(ket.irrep))`, so this returns the ordered irrep tuple.

        Returns
        -------
        `Tuple[Any, ...]`
            Canonical irrep tuple in deterministic type-name order.
        """
        return tuple(ket.irrep for ket in self.kets)

    def canonical_repr_types(self) -> Tuple[Type, ...]:
        """
        Get a tuple of concrete irrep types in this `U1Basis` in canonical order.

        This is the same order as `canonical_repr()`, which is determined by
        sorting on `full_typename(type(ket.irrep))`.

        Returns
        -------
        `Tuple[Type, ...]`
            Tuple of concrete irrep types in deterministic type-name order.
        """
        return tuple(type(ket.irrep) for ket in self.kets)


@Ket.add_conversion(U1Basis)
def ket_to_u1basis(ket: Ket[_IrrepType]) -> U1Basis:
    """Convert a `Ket` to a `U1Basis` with the ket as its only element."""
    return U1Basis(irrep=sy.Integer(1), kets=(ket,))


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_lt(a: U1Basis, b: U1Basis) -> bool:
    rep_a = a.canonical_repr()
    rep_b = b.canonical_repr()
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


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_gt(a: U1Basis, b: U1Basis) -> bool:
    rep_a = a.canonical_repr()
    rep_b = b.canonical_repr()
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
class U1Span(Span[U1Basis, sy.ImmutableDenseMatrix], Spatial, HasUnit):
    """
    Finite span of distinct single-particle basis states.

    `U1Span` is the additive container used by `U1Basis`'s `*` operator. It
    stores an ordered tuple of basis states and represents the symbolic span
    generated by those states. The object is immutable (`frozen=True`) and
    preserves insertion order.

    This type is intentionally lightweight: it does not store amplitudes,
    coefficients, or perform linear-algebra simplification. Duplicate handling
    is implemented in `operator_add` overloads, which keep only one copy of an
    existing state when building a span.

    Parameters
    ----------
    `span` : `Tuple[U1Basis, ...]`
        Ordered tuple of `U1Basis` elements contained in this span.

    Attributes
    ----------
    `span` : `Tuple[U1Basis, ...]`
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
        return self.span

    @override
    def unit(self) -> "U1Span":
        """Return the actual span without any basis scaling by a irrep."""
        return U1Span(tuple(m.unit() for m in self.span))

    @override
    def gram(self, ket: "U1Span") -> sy.ImmutableDenseMatrix:
        tbl: Dict["U1Basis", Tuple[int, "U1Basis"]] = {
            psi.unit(): (n, psi) for n, psi in enumerate(ket.span)
        }
        out = sy.zeros(self.dim, ket.dim)
        for n, psi in enumerate(self.span):
            unit = psi.unit()
            if unit not in tbl:
                continue
            m, kpsi = tbl[unit]
            out[n, m] = psi.ket(kpsi)
        return sy.ImmutableDenseMatrix(out)


U1Elements = Union[U1Basis, U1Span]
"""Union of valid element types in a `HilbertSpace`."""


@dataclass(frozen=True)
class HilbertSpace(HasUnit, StateSpace[U1Elements], Span[U1Elements, Tensor]):
    """
    Composite local Hilbert space built from states and state spans.

    `HilbertSpace` is the symbolic basis container used by tensor-network style
    operators in this module. It extends `StateSpace` by allowing each sector
    key to be either a single `U1Basis` or a grouped `U1Span`. The inherited
    `structure` mapping stores each sector together with its contiguous slice in
    the flattened basis.

    The class provides helpers for common basis-management workflows:
    - `flatten()` expands all `U1Span` sectors into explicit `U1Basis` sectors
      while preserving first-seen order and returns a normalized structure.
    - `elements()` returns the flattened basis states as `Tuple[U1Basis, ...]`.
    - `lookup(query)` retrieves a unique basis state by exact typed-irrep match.
    - `group(**groups)` partitions flattened elements into labeled `U1Span`
      sectors and returns both the grouped spans and the basis-change mapping.

    As a `Span`, `HilbertSpace` supports overlap/mapping computations through
    `gram`, which builds a `Tensor` map between two spaces using flattened
    `U1Basis` overlap (`U1Span.gram`). As a `HasUnit`, `unit()` keeps basis
    structure while replacing each element by its unit-normalized counterpart.

    Parameters
    ----------
    `structure` : `OrderedDict[U1Elements, slice]`
        Ordered sector mapping inherited from `StateSpace`, where each key is a
        `U1Basis` or `U1Span` and each value is the sector slice in flattened
        coordinates.

    Notes
    -----
    - `dim` is inherited from `StateSpace` and equals the total flattened basis
      size implied by the last slice stop.
    - `flatten()` is `@lru_cache`-backed, so repeated flattening of the same
      immutable instance is O(1) after the first call.
    - `group()` requires disjoint selectors; overlapping grouped spans raise
      `ValueError`.
    - Even if two `HilbertSpace` objects have the same flattened `U1Basis`
      elements, `permutation_order`, `flat_permutation_order`, and
      `embedding_order` may fail when `U1Span` sectors are present in one of
      the spaces, because these utilities operate on current `structure` keys
      rather than flattened basis elements.
    """

    __hash__ = StateSpace.__hash__

    def __str__(self) -> str:
        return f"HilbertSpace(dim={self.dim}, sectors={len(self.structure)})"

    def __repr__(self) -> str:
        if not self.structure:
            return f"{self}: <empty>"

        def _format_el(el: U1Elements) -> str:
            if isinstance(el, U1Basis):
                return str(el)
            span = cast(U1Span, el)
            if span.dim <= 3:
                preview = ", ".join(str(state) for state in span.span)
            else:
                preview = f"{span.span[0]}, {span.span[1]}, ..., {span.span[-1]}"
            return f"U1Span(dim={span.dim})[{preview}]"

        body = "\n".join(
            f"\t{n}: {s.start}:{s.stop} {_format_el(el)}"
            for n, (el, s) in enumerate(self.structure.items())
        )
        return f"{self}:\n{body}"

    def lookup(self, query: Dict[Type[Any], Any]) -> U1Basis:
        """
        Return the unique element that exactly matches all typed-irrep query entries.

        Parameters
        ----------
        `query` : `Dict[Type[Any], Any]`
            Mapping from irrep runtime type to expected irrep value.
            A candidate element matches only if, for every `(T, value)` pair,
            it contains an irrep of exact type `T` and `irrep == value`.

        Returns
        -------
        `U1Basis`
            The unique matching element.

        Raises
        ------
        `ValueError`
            If no element matches, if multiple elements match, or if `query` is empty.
        """
        if not query:
            raise ValueError("lookup query cannot be empty.")

        matches: list[U1Basis] = []
        for el in self.elements():
            if not isinstance(el, U1Basis):
                continue

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

    @lru_cache
    def flatten(self) -> "HilbertSpace":
        """
        Return a new instance of `HilbertSpace` with all `HilbertSpan` flattened to its elements
        respecting the original ordering.

        Returns
        -------
        `HilbertSpace`
            A new instance of `HilbertSpace` with all `HilbertSpan` flattened.
        """
        flattened_elements: OrderedDict[U1Elements, slice] = OrderedDict()
        for el in self.structure.keys():
            if issubclass(type(el), U1Span):
                for m in cast(U1Span, el).elements():
                    flattened_elements[m] = slice(0, m.dim)
                continue
            flattened_elements[el] = slice(0, el.dim)
        return HilbertSpace(restructure(flattened_elements))

    def elements(self) -> Tuple[U1Basis, ...]:
        """Get the flattened elements of this `HilbertSpace`."""
        return cast(Tuple[U1Basis, ...], tuple(self.flatten().structure.keys()))

    class GroupResult(NamedTuple):
        """
        Result payload returned by `HilbertSpace.group`.

        Attributes
        ----------
        `spans` : `FrozenDict[str, U1Span]`
            Mapping from user-provided group label to the generated span.
        `mapping` : `Tensor`
            Basis-change map from the original space to the regrouped
            `HilbertSpace`.
        """

        spans: FrozenDict[str, U1Span]
        """Mapping from user-provided group label to the generated span."""
        mapping: Tensor
        """Basis-change map from the original space to the regrouped `HilbertSpace`."""

    def group(
        self, **groups: Union[Callable[[U1Basis], bool], Any]
    ) -> "HilbertSpace.GroupResult":
        """
        Group flattened basis elements into labeled spans.

        Parameters
        ----------
        `**groups` : `Union[Callable[[U1Basis], bool], Any]`
            Label-to-selector mapping. A selector may be:
            - `Callable[[U1Basis], bool]`: include states where predicate is `True`.
            - an irrep object: converted to a predicate selecting states where
              `state.irrep_of(type(selector)) == selector`.

        Returns
        -------
        `HilbertSpace.GroupResult`
            - `spans`: frozen mapping from labels to generated `U1Span`.
              Each generated span is sorted in ascending `U1Basis` order.
            - `mapping`: tensor with dims `(self, new_hilbert_space)` mapping
              original flattened elements to a regrouped space with structure
              `(ungrouped elements) + (generated spans)`.
        """
        elements = self.elements()
        all_span = U1Span(elements)

        spans_by_label: OrderedDict[str, U1Span] = OrderedDict()
        grouped_union = U1Span(())

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
            span = U1Span(tuple(sorted(selected)))
            overlap = grouped_union & span
            if overlap.dim > 0:
                raise ValueError(
                    f"grouped spans overlap: state {overlap.span[0]!r} appears in multiple groups (including {label!r})."
                )

            grouped_union = grouped_union | span
            spans_by_label[label] = span

        ungrouped = (all_span - grouped_union).span
        new_hilbert = hilbert((*ungrouped, *spans_by_label.values()))

        return HilbertSpace.GroupResult(
            spans=FrozenDict(spans_by_label),
            mapping=self.gram(new_hilbert),
        )

    class GroupByResult(NamedTuple):
        """
        Result payload returned by `HilbertSpace.group_by`.

        Attributes
        ----------
        `groups` : `Tuple[U1Span, ...]`
            Grouped spans in the order their irrep keys first appear in the
            flattened basis.
        `mapping` : `Tensor`
            Basis-change map from the original space to the grouped
            `HilbertSpace`.
        """

        groups: Tuple[U1Span, ...]
        """Grouped spans in the order their irrep keys first appear in the flattened basis."""
        mapping: Tensor
        """Basis-change map from the original space to the grouped `HilbertSpace`."""

    def group_by(self, *T: Type) -> "HilbertSpace.GroupByResult":
        """
        Form groups by matching irrep types keyed by the irreps.

        This function will try to group the basis by the irrep in order specified by the types in `T`.

        Parameters
        ----------
        `*T` : `Type`
            The irrep types to group by. The grouping will be performed in the order of the types specified.
            For example `group_by(A, B)` will group the basis by `(A, B)`, the basis with the same irrep of `A` and `B`
            will be within the same `U1Span`.

        Returns
        -------
        `HilbertSpace.GroupByResult`
            The result of grouping the basis elements by the specified irrep types.
        """
        keys: OrderedDict[Tuple[Any, ...], None] = OrderedDict()
        for el in self.elements():
            keys[tuple(el.irrep_of(t) for t in T)] = None

        selectors: OrderedDict[str, Callable[[U1Basis], bool]] = OrderedDict()
        for i, key in enumerate(keys):
            selectors[f"_{i}"] = lambda el, key=key: all(
                el.irrep_of(t) == target for t, target in zip(T, key)
            )

        result = self.group(**selectors)
        return HilbertSpace.GroupByResult(
            groups=tuple(result.spans.values()),
            mapping=result.mapping,
        )

    def is_homogeneous(self) -> bool:
        """
        Check if this `HilbertSpace` is homogeneous, i.e., all basis states share the same irrep types.

        Returns
        -------
        `bool`
            `True` if all basis states in this `HilbertSpace` have the same set of irrep types, `False` otherwise.
        """
        elements = self.elements()
        if not elements:  # An empty Hilbert space is considered homogeneous.
            return True
        irrep_types = elements[0].canonical_repr_types()
        for el in elements[1:]:
            if el.canonical_repr_types() != irrep_types:
                return False
        return True

    def factorize(self, *irrep_types: Tuple[Type, ...]) -> StateSpaceFactorization:
        """
        Factorize this homogeneous `HilbertSpace` into tensor factors grouped by irrep type.

        Each argument in `irrep_types` defines one output factor as a tuple of irrep
        types to group together. Across all groups, every irrep type in this space must
        appear exactly once. The result is a `StateSpaceFactorization` describing the factor
        spaces and the basis reindexing needed to move between the original basis order
        and the factorized tensor-product structure.

        Example
        -------
        For basis states labeled by `(int, str)`:
        `(|1⟩|'a'⟩, |1⟩|'b'⟩, |2⟩|'a'⟩, |2⟩|'b'⟩)`,
        `factorize((int,), (str,))` produces two factors:
        `(|1⟩, |2⟩)` and `(|'a'⟩, |'b'⟩)`.

        Failed examples
        ---------------
        Assume a homogeneous space with canonical irrep-type order `(int, str, float)`.

        - If the space mixes different irrep-type layouts across basis states, it is not
          homogeneous and factorization fails.
        - If a factorization leaves out one of the space's irrep types (for example,
          dropping `float`), factorization fails.
        - If a factorization introduces an irrep type not present in the space (for
          example `bool`), factorization fails.
        - Incomplete Cartesian-product basis: for basis
          `(|1⟩|'a'⟩, |1⟩|'b'⟩, |2⟩|'a'⟩, |2⟩|'b'⟩, |2⟩|'c'⟩)`,
          factorization by `(int,)` and `(str,)` fails because the factorized product
          would require `|1⟩|'c'⟩` as well.

        Parameters
        ----------
        `*irrep_types` : `Tuple[Type, ...]`
            Factor specification. Each tuple is one factor, containing the irrep
            types assigned to that factor.

        Returns
        -------
        `StateSpaceFactorization`
            Factorization metadata: factor spaces and basis-index mapping.

        Raises
        ------
        `ValueError`
            Raised when:
            - this space is not homogeneous;
            - some irrep type in the space is missing from `irrep_types`;
            - `irrep_types` contains a type not present in the space;
            - the basis is not factorizable for the requested groups (incomplete
              Cartesian-product structure).
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

        canonical_types = elements[0].canonical_repr_types()
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
        for group in irrep_types:
            keys: OrderedDict[Tuple[Any, ...], None] = OrderedDict()
            for el in elements:
                keys[tuple(el.irrep_of(T) for T in group)] = None
            grouped_basis = tuple(
                U1Basis(sy.Integer(1), tuple(Ket(irrep) for irrep in key))
                for key in keys
            )
            factor_keys.append(tuple(keys.keys()))
            factorized.append(hilbert(grouped_basis))

        # Map each basis state to its grouped-irrep key tuple.
        combo_to_element: OrderedDict[Tuple[Tuple[Any, ...], ...], U1Basis] = (
            OrderedDict()
        )
        for el in elements:
            combo = tuple(tuple(el.irrep_of(T) for T in group) for group in irrep_types)
            combo_to_element[combo] = el

        expected_size = 1
        for keys in factor_keys:
            expected_size *= len(keys)
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
            align_dim=hilbert(align_elements),
        )

    @override
    def tensor_product(self, other: "HilbertSpace") -> "HilbertSpace":
        """
        Build the tensor-product space of this space with another `HilbertSpace`.

        The resulting basis is generated by taking every ordered pair
        `(a, b)` from `self.elements()` and `other.elements()`, then forming the
        basis-level tensor product `a @ b` for each pair.

        Ordering follows `itertools.product(self.elements(), other.elements())`:
        basis elements from `self` vary slowest, and basis elements from `other`
        vary fastest. This deterministic order is important for tensor reshape
        and alignment logic that depends on stable axis conventions.

        Parameters
        ----------
        `other` : `HilbertSpace`
            Right-hand tensor factor.

        Returns
        -------
        `HilbertSpace`
            A new space whose basis spans `self ⊗ other`.
        """
        elements = []
        for a, b in product(self.elements(), other.elements()):
            elements.append(a @ b)
        return hilbert(elements)

    @override
    def unit(self) -> "HilbertSpace":
        return hilbert(el.unit() for el in self)

    @override
    def gram(self, another: "HilbertSpace") -> Tensor:
        span = U1Span(cast(Tuple[U1Basis, ...], self.elements()))
        new_span = U1Span(cast(Tuple[U1Basis, ...], another.elements()))
        irrep = span.gram(new_span)
        precision = get_precision_config()
        data = torch.from_numpy(
            np.asarray(irrep.tolist(), dtype=precision.np_complex)
        ).to(dtype=precision.torch_complex)
        return Tensor(data=data, dims=(self, another))


def hilbert(itr: Iterable[U1Elements]) -> HilbertSpace:
    structure: OrderedDict[U1Elements, slice] = OrderedDict()
    base = 0
    for el in itr:
        structure[el] = slice(base, base + el.dim)
        base += el.dim
    return HilbertSpace(structure=structure)


@U1Basis.add_conversion(StateSpace)
def u1basis_to_hilbertspace(basis: U1Basis) -> StateSpace:
    """Convert a `U1Basis` to a `HilbertSpace` containing only that basis state."""
    return hilbert((basis,))


# Support conversion to HilbertSpace using `basis.convert(HilbertSpace)`.
U1Basis.add_conversion(HilbertSpace)(u1basis_to_hilbertspace)


@dispatch(HilbertSpace, HilbertSpace)  # type: ignore[no-redef]
def same_span(a: HilbertSpace, b: HilbertSpace) -> bool:
    return set(m.unit() for m in a.structure.keys()) == set(
        m.unit() for m in b.structure.keys()
    )


_ObservableType = TypeVar("_ObservableType")


class U1Operator(Generic[_ObservableType], Functional, Operable, ABC):
    """
    A composable operator that acts on observable-compatible objects.

    `Operator` combines two core behaviors:

    1. `Functional` dispatch and chaining
       Implementations are registered via `Functional.register` for pairs of
       `(input_type, operator_subclass)`. At runtime, :meth:`apply` resolves the
       function chain for the concrete input object and executes each function in
       order.

    2. `Operable` matrix-application syntax
       Because `Operator` is `Operable`, it participates in the overloaded `@`
       operator. This module defines `operator_matmul(Operator, U1Basis)`,
       so `op @ value` applies the operator and returns only the transformed
       value component.

    Conceptually, an operator application returns two outputs:

    - an observation/measurement-like payload (`_ObservableType`)
    - the transformed object (same runtime type as the input)

    The observation payload allows operator execution to report auxiliary
    information while still producing an updated value.

    Type Parameters
    ---------------
    `_ObservableType`
        The type of the first element returned by an operator application
        (e.g. expectation value, coefficient, metadata, or any domain-specific
        observable artifact).

    Implementation Guidelines
    -------------------------
    - The registered callable chain for an operator must produce a
      2-tuple `(observable, transformed_value)`.
    - The transformed value must have the same runtime type as the input object.
      This invariant is validated by :meth:`apply` using assertions.
    - Closure validation in :meth:`apply` has two branches:
      - for `U1Basis` inputs, closure is span-based
        (`same_span(input, transformed_value)`).
      - for non-`U1Basis` inputs, closure is value-based
        (`input == transformed_value`).
      In either branch, if closure fails, the observable must be `None`.
    - If no registration exists for `(type(input), type(operator))`,
      :class:`NotImplementedError` is raised by `Functional.apply`.

    Usage Pattern
    -------------
    1. Define an `Operator` subclass.
    2. Register behavior with `@YourOperatorSubclass.register(InputType)`.
    3. Apply by either:
       - `obs, out = op(input_obj)` to receive both outputs, or
       - `out = op @ input_obj` to receive only the transformed value.
    """

    @override
    def apply(  # type: ignore[override]
        self, v: U1Basis, **kwargs
    ) -> Tuple[_ObservableType, U1Basis]:
        result = super().apply(v, **kwargs)
        assert isinstance(result, tuple), (
            f"Operator {type(self)} acting on {type(v).__name__} should yield a Tuple[Any, Any]!"
        )
        o, ov = result
        assert isinstance(ov, type(v)), (
            f"Operator {type(self)} acting on {type(v).__name__} should yield a Tuple[Any, {type(v).__name__}]!"
        )
        if type(v) is type(ov):
            try:
                needs_none = not same_span(v, ov)  # type: ignore[arg-type]
            except Exception:
                needs_none = v != ov
        else:
            needs_none = v != ov
        if needs_none:
            assert o is None, (
                f"Un-closed operator action should have undefined irrep (None), got {o}!"
            )

        return o, ov

    def eigen_opr(self) -> Self:
        """
        Return an eigenstate-validating copy of this operator.

        This helper creates a shallow copy of the operator and wraps its
        :meth:`apply` method with a post-condition check: the transformed output
        must be equal to the input value. If the value changes, the wrapped
        operator raises :class:`ValueError`.

        Use this when downstream logic assumes that the input is already an
        eigenstate of the operator and should therefore remain unchanged by the
        operator action.

        Returns
        -------
        Self
            A copy of the current operator whose application enforces
            the same closure condition as :meth:`apply`.

        Raises
        ------
        ValueError
            Raised at call time if the wrapped operator is applied to a value
            that is not closure-preserving under :meth:`apply` semantics.

        Examples
        --------
        A common use case is validating basis assumptions in algorithms that
        require eigen-aligned states:

        - Build `checked = op.eigen_opr()`.
        - Apply `checked(state)` before a specialized computation.
        - Fail fast with `ValueError` if `state` is not an eigenstate.
        """
        op = copy(self)
        apply = op.apply

        def eigen_apply(v: U1Basis, **kwargs) -> Tuple[_ObservableType, U1Basis]:
            """
            Apply the copied operator and enforce closure-preserving output.

            This wrapper delegates to the copied operator's original
            :meth:`apply` implementation, then validates that the transformed
            output remains closed with the input under the same semantics used by
            :meth:`Operator.apply`:

            - If both input and output are `U1Basis`, closure is checked
              by span membership with `same_span(v, ov)`.
            - Otherwise, closure is checked by strict value equality `ov == v`.

            Parameters
            ----------
            `v` : `U1Basis`
                Input object to transform.
            `**kwargs`
                Keyword arguments forwarded to the wrapped :meth:`apply` call.

            Returns
            -------
            `Tuple[_ObservableType, U1Basis]`
                The observable payload and transformed output from the wrapped
                operator call, when closure is preserved.

            Raises
            ------
            `ValueError`
                If the transformed output is not closure-preserving with respect
                to `v` under the branch-specific rule above.
            """
            o, ov = apply(v, **kwargs)
            if isinstance(v, U1Basis) and isinstance(ov, U1Basis):
                is_closed = same_span(v, ov)
            else:
                is_closed = ov == v
            if not is_closed:
                raise ValueError(
                    f"{type(op).__name__} expected a closure-preserving state, but output "
                    "{ov!r} is not closed with input {v!r}."
                )
            return o, ov

        object.__setattr__(op, "apply", eigen_apply)
        return op


@dispatch(U1Operator, Operable)
def operator_matmul(o: U1Operator, v: Operable):
    _, v = o(v)
    return v


@dataclass(frozen=True)
class FuncOpr(Generic[_IrrepType], U1Operator[sy.Integer]):
    T: Type[_IrrepType]
    func: Callable


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def same_span(a: U1Basis, b: U1Basis) -> bool:
    """Check if the unit basis of two `U1Basis` are the same."""
    return a.unit() == b.unit()


@FuncOpr.register(U1Basis)
def _func_opr_u1_state(f: FuncOpr, psi: U1Basis) -> Tuple[sy.Integer | None, U1Basis]:
    irrep = psi.irrep_of(f.T)
    new_irrep = f.func(irrep)
    func_irrep = sy.Integer(1) if irrep == new_irrep else None
    new_psi = psi.replace(new_irrep)
    return func_irrep, new_psi


@FuncOpr.register(U1Span)
def _func_opr_u1_span(f: FuncOpr, s: U1Span) -> Tuple[sy.Integer | None, U1Span]:
    new_s: U1Span = replace(s, span=tuple(f @ psi for psi in s.span))
    func_irrep = sy.Integer(1) if same_span(s, new_s) else None
    return func_irrep, new_s


@FuncOpr.register(HilbertSpace)
def _func_opr_hilbert(
    f: FuncOpr, h: HilbertSpace
) -> Tuple[sy.Integer | None, HilbertSpace]:
    new_h = hilbert(f @ el for el in h)
    func_irrep = sy.Integer(1) if same_span(h, new_h) else None
    return func_irrep, new_h


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def same_span(a: U1Span, b: U1Span) -> bool:
    return set(a.unit().span) == set(b.unit().span)


@dispatch(Ket, Ket)
def operator_eq(a: Ket, b: Ket) -> bool:
    return a.irrep == b.irrep


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_eq(a: U1Basis, b: U1Basis) -> bool:
    return a.irrep == b.irrep and a.kets == b.kets


@dispatch(Ket, Ket)  # type: ignore[no-redef]
def operator_matmul(a: Ket, b: Ket) -> U1Basis:
    return U1Basis(sy.Integer(1), (a, b))


@dispatch(U1Basis, Ket)  # type: ignore[no-redef]
def operator_matmul(psi: U1Basis, ket: Ket) -> U1Basis:
    return U1Basis(psi.irrep, psi.kets + (ket,))


@dispatch(Ket, U1Basis)  # type: ignore[no-redef]
def operator_matmul(ket: Ket, psi: U1Basis) -> U1Basis:
    return U1Basis(psi.irrep, (ket,) + psi.kets)


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_matmul(a: U1Basis, b: U1Basis) -> U1Basis:
    if not a.kets:
        return b
    if not b.kets:
        return a
    return U1Basis((a.irrep * b.irrep).simplify(), a.kets + b.kets)


@dispatch(U1Basis, U1Basis)
def operator_or(a: U1Basis, b: U1Basis) -> U1Span:
    if a == b:
        return U1Span((a,))
    return U1Span((a, b))


@dispatch(U1Span, U1Basis)  # type: ignore[no-redef]
def operator_or(span: U1Span, state: U1Basis) -> U1Span:
    if state in span.span:
        return span
    return U1Span(span.span + (state,))


@dispatch(U1Basis, U1Span)  # type: ignore[no-redef]
def operator_or(state: U1Basis, span: U1Span) -> U1Span:
    if state in span.span:
        return span
    return U1Span((state,) + span.span)


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def operator_or(a: U1Span, b: U1Span) -> U1Span:
    existing = set(a.span)
    new_states = tuple(s for s in b.span if s not in existing)
    return U1Span(a.span + new_states)


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def operator_sub(a: U1Span, b: U1Span) -> U1Span:
    b_elements = set(b.span)
    new_states = tuple(s for s in a.span if s not in b_elements)
    return U1Span(new_states)


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def operator_and(a: U1Span, b: U1Span) -> U1Span:
    b_elements = set(b.span)
    new_states = tuple(s for s in a.span if s in b_elements)
    return U1Span(new_states)
