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
from multipledispatch import dispatch  # type: ignore[import-untyped]

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
from .state_space import StateSpace, StateSpaceFactorization
from ..linalg.tensors import Tensor
from ..precision import get_precision_config
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
    """
    Immutable single-particle basis state built from typed irreps.

    `U1Basis` is a symbolic tensor-product state with U(1) irreducible representation
    presented as an ordered tuple of irreps. This object is
    intentionally symbolic: it stores basis labels only and does not store amplitudes
    or coefficients.

    A key invariant is enforced at construction time: for each concrete irrep
    type, the state must contain exactly one ket of that type. In other words,
    irrep multiplicities must be unity. This guarantees that type-based updates
    via `replace` are unambiguous and fast.

    Parameters
    ----------
    `u1`: `sy.Expr`
        The irrep of this state under an recent operation.
    `rep` : `Tuple[Any, ...]`
        Tuple of irreps that defines the state. Input order is canonicalized in
        `__post_init__` by sorting on `full_typename(type(irrep))`.

    Attributes
    ----------
    `u1`: `sy.Expr`
        The irrep of this state under an recent operation.
    `rep` : `Tuple[Any, ...]`
        Immutable canonical irrep order sorted by concrete irrep type name
        (`module.qualname`).

    Notes
    -----
    - `dim` is always `1`; this type represents one basis vector.
    - Irrep order is canonicalized at construction; permutations of the same
      typed irreps produce the same internal `rep` tuple.
    - `replace(irrep)` substitutes the unique irrep with the same
      concrete runtime type as `irrep`.
    - `@` dispatch overloads combine reps/states into a new `U1Basis`.
    - `|` dispatch overloads build a `U1Span` of distinct `U1Basis` values.
    - Ordering (`<`, `>`) compares, in order: number of irreps, tuple of
      fully-qualified irrep type names (`module.qualname`) from
      `self.rep`, then the canonical irrep-value tuple itself.
      If irrep values of matching types are not orderable, the comparison
      raises from the underlying irrep objects.

    Raises
    ------
    `ValueError`
        Raised in `__post_init__` when any irrep type appears with multiplicity
        different from `1`.
    """

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rep",
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
        Build a `U1Basis` with the given reps and a default U(1) value of `1`.

        Parameters
        ----------
        `rep` : `Any`
            Irreps to build the state from. Input order is canonicalized in
            `__post_init__` by sorting on `full_typename(type(irrep))`.

        Returns
        -------
        `U1Basis`
            A `U1Basis` instance with the given reps and a default U(1) value of `1`.
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
        returns a new `U1Basis` with that irrep substituted. The original instance is
        not modified.

        Parameters
        ----------
        `irrep` : `Any`
            Replacement irrep instance. Its concrete type must already exist in this
            state exactly once (enforced by `U1Basis.__post_init__`).

        Returns
        -------
        `U1Basis`
            A new `U1Basis` with one irrep replaced.

        Raises
        ------
        `ValueError`
            If this state does not contain any irrep with the same concrete
            type as `irrep`.
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
        `*T` : `Type[Any]`
            Concrete irrep types to remove. Matching uses exact runtime type
            identity (`type(x) is T`), not subclass checks.

        Returns
        -------
        `U1Basis`
            A new `U1Basis` with all irreps whose concrete types are in `T`
            removed. If none of the requested types are present, `self` is
            returned unchanged.
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

        This method performs a direct scan over `self.rep` and returns the
        first irrep satisfying `type(irrep) is T`. Because
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
        - Runtime is linear in the number of irreps (`O(n)`), with no temporary
          mapping allocations.
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
        `psi` : `U1Basis`
            The state to compute the overlap with.

        Returns
        -------
        `sy.Expr`
            The symbolic overlap of this state with `psi`. If the irreps of the
            two states do not match, the overlap is `0`. If the irreps match, the
            overlap is the product of this state's U(1) value and the conjugate of
            `psi`'s U(1) value.
        """
        if self.base != psi.base:
            return sy.Integer(0)
        return cast(sy.Expr, (sy.conjugate(self.coef) * psi.coef).simplify())

    def __str__(self) -> str:
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
        return self.__str__()

    @override
    def rays(self) -> "U1Basis":
        """Return the canonical ray representative with U(1) coefficient `1`."""
        return replace(self, coef=sy.Integer(1))

    def repr_types(self) -> Tuple[Type, ...]:
        """
        Get a tuple of concrete irrep types in this `U1Basis` in canonical order.

        This is the same order as `self.rep`, which is determined by
        sorting on `full_typename(type(irrep))`.

        Returns
        -------
        `Tuple[Type, ...]`
            Tuple of concrete irrep types in deterministic type-name order.
        """
        return tuple(type(irrep) for irrep in self.base)


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_lt(a: U1Basis, b: U1Basis) -> bool:
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


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_gt(a: U1Basis, b: U1Basis) -> bool:
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
    def rays(self) -> "U1Span":
        """Return the span obtained by replacing each basis state by its ray representative."""
        return U1Span(tuple(m.rays() for m in self.span))

    def cross_gram(self, ket: "U1Span") -> sy.ImmutableDenseMatrix:
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
    """Convert a `U1Basis` to a `U1Span` containing just that basis state."""
    return U1Span((basis,))


@need_validation()
@dataclass(frozen=True)
class HilbertSpace(HasRays, StateSpace[U1Basis], Span[U1Basis]):
    """
    Composite local Hilbert space built from states and state spans.

    `HilbertSpace` is the symbolic basis container used by tensor-network style
    operators in this module. It extends `StateSpace` with `U1Basis` sectors.
    The inherited `structure` mapping stores each sector together with its
    integer index in basis order.

    The class provides helpers for common basis-management workflows:
    - `elements()` returns the flattened basis states as `Tuple[U1Basis, ...]`.
    - `lookup(query)` retrieves a unique basis state by exact typed-irrep match.
    - `group(**groups)` partitions basis elements into labeled grouped
      `HilbertSpace` values.

    As a `Span`, `HilbertSpace` supports overlap/mapping computations through
    `cross_gram`, which builds a `Tensor` map between two spaces using `U1Basis`
    overlap (`U1Span.cross_gram`). As a `HasRays`, `rays()` keeps basis
    structure while replacing each element by its canonical ray representative.

    Parameters
    ----------
    `structure` : `OrderedDict[U1Basis, int]`
        Ordered sector mapping inherited from `StateSpace`, where each key is a
        `U1Basis` and each value is the sector index in basis coordinates.

    Notes
    -----
    - `dim` is inherited from `StateSpace` and equals the total flattened basis
      size (the number of indexed basis sectors).
    - `group()` requires disjoint selectors; overlapping grouped spans raise
      `ValueError`.
    - `permutation_order` and `embedding_order` operate on current
      `structure` keys.
    """

    __hash__ = StateSpace.__hash__

    @staticmethod
    def new(itr: Iterable[U1Basis]) -> "HilbertSpace":
        """
        Build a `HilbertSpace` from an ordered basis iterable.

        Parameters
        ----------
        `itr` : `Iterable[U1Basis]`
            Basis elements to insert into the space in iteration order. Each
            element is assigned its zero-based sector index according to its
            position in `itr`.

        Returns
        -------
        `HilbertSpace`
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
        return f"HilbertSpace(dim={self.dim}, sectors={len(self.structure)})"

    def __repr__(self) -> str:
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

        Parameters
        ----------
        `**groups` : `Union[Callable[[U1Basis], bool], Any]`
            Label-to-selector mapping. A selector may be:
            - `Callable[[U1Basis], bool]`: include states where predicate is `True`.
            - an irrep object: converted to a predicate selecting states where
              `state.irrep_of(type(selector)) == selector`.

        Returns
        -------
        `FrozenDict[str, HilbertSpace]`
            Frozen mapping from labels to grouped `HilbertSpace` values.
            Each grouped subspace is sorted in ascending `U1Basis` order.
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
        `*T` : `Type`
            The irrep types to group by. The grouping will be performed in the order of the types specified.
            For example `group_by(A, B)` will group the basis by `(A, B)`, and
            basis states with the same irrep of `A` and `B` will be in the same
            grouped subspace.

        Returns
        -------
        `Tuple[HilbertSpace, ...]`
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
        Check if this `HilbertSpace` is homogeneous, i.e., all basis states share the same irrep types.

        Returns
        -------
        `bool`
            `True` if all basis states in this `HilbertSpace` have the same set of irrep types, `False` otherwise.
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
        Return the canonical irrep-type order shared by this `HilbertSpace` basis.

        A homogeneous Hilbert space has one consistent irrep-type layout across all
        basis states (for example `(int, str)` for every element). This method
        returns that layout in canonical order.

        For an empty space, the result is the empty tuple `()`.

        Returns
        -------
        `Tuple[Type[Any], ...]`
            The canonical irrep-type sequence of the basis representation.

        Raises
        ------
        `ValueError`
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

    def factorize(
        self, *irrep_types: Tuple[Type, ...], coef_on: Optional[int] = None
    ) -> StateSpaceFactorization:
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
        `coef_on` : `Optional[int]`
            Index of the factor that inherits the original `U1Basis.coef`.
            `None` defaults to the leftmost factor (`0`). Negative indices are
            interpreted using normal Python indexing. All other factors are
            built with coefficient `1`.

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
            - `coef_on` is out of range for the requested factors;
            - the requested `coef_on` assignment is not well-defined because the
              same grouped basis key appears with different coefficients;
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
        return HilbertSpace.new(elements)

    @override
    def rays(self) -> "HilbertSpace":
        """Return the Hilbert space obtained by replacing each basis state by its ray representative."""
        return HilbertSpace.new(el.rays() for el in self)

    def cross_gram(self, another: "HilbertSpace") -> Tensor:
        """
        Build the cross-Gram overlap matrix between this basis and another basis.

        Matrix entries are computed from concrete basis overlaps
        `G_{ij} = <self_i | another_j>`, so any nontrivial U(1) irrep phase in
        basis vectors is encoded in `data`.

        Output dimension convention
        ---------------------------
        The returned tensor uses dims `(self, another.rays())`.
        The target (column) dimension is intentionally replaced by its canonical
        ray representative (phase removed)
        so the codomain metadata is gauge-fixed, while phase information remains
        in matrix elements.

        This is equivalent to using a representative of the same ray space
        (projective Hilbert space) for the target basis labels.
        """
        span = U1Span(cast(Tuple[U1Basis, ...], self.elements()))
        new_span = U1Span(cast(Tuple[U1Basis, ...], another.elements()))
        irrep = span.cross_gram(new_span)
        precision = get_precision_config()
        data = torch.from_numpy(
            np.asarray(irrep.tolist(), dtype=precision.np_complex)
        ).to(dtype=precision.torch_complex)
        return Tensor(data=data, dims=(self, another.rays()))


@U1Basis.add_conversion(StateSpace)
def _u1basis_to_hilbertspace(basis: U1Basis) -> StateSpace:
    """Convert a `U1Basis` to a `HilbertSpace` containing only that basis state."""
    return HilbertSpace.new((basis,))


# Support conversion to HilbertSpace using `basis.convert(HilbertSpace)`.
U1Basis.add_conversion(HilbertSpace)(
    cast(Callable[[U1Basis], HilbertSpace], _u1basis_to_hilbertspace)
)


@U1Span.add_conversion(StateSpace)
def _u1span_to_hilbertspace(span: U1Span) -> StateSpace:
    """Convert a `U1Span` to a `HilbertSpace` containing the span's basis states."""
    return HilbertSpace.new(span.span)


U1Span.add_conversion(HilbertSpace)(
    cast(Callable[[U1Span], HilbertSpace], _u1span_to_hilbertspace)
)


@HilbertSpace.add_conversion(HilbertSpace)
def _(v: HilbertSpace) -> HilbertSpace:
    """Identity conversion for `HilbertSpace`."""
    return v


@dispatch(HilbertSpace, HilbertSpace)  # type: ignore[no-redef]
def same_rays(a: HilbertSpace, b: HilbertSpace) -> bool:
    return set(m.rays() for m in a.structure.keys()) == set(
        m.rays() for m in b.structure.keys()
    )


_T = TypeVar("_T")


class Opr(Functional, Operable, ABC):
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
    - a transformed object of the same runtime type as the input, or
    - `Multiple(coef, transformed_object)` when the action also contributes a
      scalar prefactor

    The `Multiple` form is used when the operator naturally decomposes into a
    scalar factor times an object in the same representation family. Typical
    examples include phase factors, characters, gauge coefficients, and other
    symbolic amplitudes that should stay factored instead of being folded
    directly into the transformed object.

    Type Parameters
    ---------------
    `_ObservableType`
        Historical name from an earlier design. In the current implementation,
        operators return either the transformed object itself or
        `Multiple[transformed_object]`.

    Implementation Guidelines
    -------------------------
    - A registered handler should return a plain transformed object when the
      operator acts without producing an extra scalar factor.
    - A registered handler should return `Multiple(coef, value)` when the
      action produces a scalar prefactor that should remain explicit.
    - In both cases, the underlying transformed value must stay in the same
      representation family as the input. More precisely:
      - plain return: `type(result) is type(input)` or a compatible subclass
      - multiple return: `type(result.base) is type(input)` or a compatible
        subclass
    - `Opr.invoke` automatically lifts operators over `Multiple`: if
      `op(base) -> y`, then `op(Multiple(c, base)) -> Multiple(c, y)`; if
      `op(base) -> Multiple(c2, y)`, the coefficients are multiplied to produce
      `Multiple(simplify(c * c2), y)`.
    - For symbolic container inputs `U1Basis`, `U1Span`, and `HilbertSpace`,
      the generic `Opr` lifting in this module expects the final result to stay
      in-kind and not return `Multiple` at the top level. Scalar factors should
      instead be attached to the contained irreps/components and accumulated
      into the container's internal coefficient structure.
    - This module provides generic `Opr` registrations for `U1Basis`,
      `U1Span`, and `HilbertSpace`. Subclasses inherit those handlers through
      `Functional` MRO fallback unless they define more specific registrations.
    - If no registration exists for `(type(input), type(operator))` after MRO
      fallback on both the input type and the operator type,
      :class:`NotImplementedError` is raised by `Functional.invoke`.

    Usage Pattern
    -------------
    1. Define an `Operator` subclass.
    2. Register behavior with `@YourOperatorSubclass.register(InputType)`.
    3. Apply by either:
       - `out = op(input_obj)` to receive the transformed value, or
       - `out = op @ input_obj` using infix syntax.

    Return-value Guidance
    ---------------------
    As a rule of thumb:

    - return `value` when the operator only changes the representation/object
      itself
    - return `Multiple(coef, value)` when the operator contributes a symbolic
      scalar prefactor that should remain factored

    In particular, for atomic irreps or basis-function-like objects, returning
    `Multiple` is often appropriate. For higher-level symbolic containers such
    as `U1Basis`, `U1Span`, and `HilbertSpace`, prefer returning the same
    container type directly and let the generic lifting logic accumulate scalar
    factors internally.
    """

    @override
    def invoke(  # type: ignore[override]
        self, v: _T, **kwargs
    ) -> Union[_T, Multiple[_T]]:
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


@dispatch(Opr, Operable)
def operator_matmul(o: Opr, v: Operable):
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


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def same_rays(a: U1Basis, b: U1Basis) -> bool:
    """Check if two `U1Basis` define the same ray."""
    return a.rays() == b.rays()


@dataclass(frozen=True)
class FuncOpr(Generic[_IrrepType], Opr):
    """
    Symbolic operator induced by applying a Python callable to one irrep type.

    `FuncOpr` lifts a plain callable acting on a single irrep/component into an
    `Opr` that acts on the symbolic Hilbert-space objects defined in this module.
    The operator is parameterized by:

    - `T`: the concrete runtime type of the irrep/component to target
    - `func`: a callable mapping `T -> T | Multiple[T]`

    Registered actions
    ------------------
    `FuncOpr` defines a specialized registration on:

    - `U1Basis`
      Finds the unique irrep in `psi.base` whose exact runtime type is `T`,
      applies `func` to it, and rebuilds the basis state with that component
      replaced.

    For `U1Span` and `HilbertSpace`, `FuncOpr` relies on the inherited generic
    `Opr` lifting in this module, which maps the operator over contained
    `U1Basis` elements.

    Semantics
    ---------
    This class is intended for structure-preserving symbolic rewrites such as
    "replace each `Offset` by `Offset.fractional()`" or "apply a transformation
    to each irrep of a basis state". It is not a general-purpose map over
    arbitrary Python objects: dispatch exists only for the symbolic container
    types registered below.

    The callable `func` is expected to return either:

    - a transformed object of the same concrete type `T`, or
    - `Multiple(coef, value)` where `value` has type `T`

    Return plain `T` when the transformation only changes the irrep/component
    itself. Return `Multiple` when the transformation also contributes an
    explicit scalar factor, such as a phase or gauge coefficient, that should
    be accumulated symbolically rather than absorbed into the object.

    `FuncOpr` does not itself validate the semantic correctness of `func`, but
    the surrounding symbolic code assumes the transformation stays within the
    same representation family.

    Interaction with `@`
    --------------------
    Because `FuncOpr` is an `Opr`, it participates in the overloaded `@`
    syntax:

    - `f @ x` applies the operator to an object `x`
    - `f @ g` forms a `ComposedOpr`

    With the current `ComposedOpr` semantics, operator composition follows the
    standard algebraic order:

    - `(f @ g) @ x == f(g(x))`

    So `f @ g` means "apply `g`, then apply `f`".

    Interaction with `Multiple`
    ---------------------------
    If `v` is `Multiple(coef, base)`, then `f(v)` applies `f` to `base`.

    - If the result is a plain object `base'`, the output is
      `Multiple(coef, base')`.
    - If the result is `Multiple(coef', base')`, the coefficients are
      multiplied and simplified, producing
      `Multiple(simplify(coef * coef'), base')`.

    This behavior is provided centrally by `Opr.invoke`, so `FuncOpr` inherits
    the same coefficient-lifting semantics as every other operator subclass.

    Interaction with Symbolic Containers
    ------------------------------------
    `FuncOpr` treats `U1Basis` specially: it targets the unique irrep of type
    `T` via `psi.irrep_of(T)` rather than iterating over all components with
    `allows(...)`. This is why `FuncOpr` keeps a dedicated `U1Basis`
    registration even though `U1Span` and `HilbertSpace` can be handled by the
    generic `Opr` lifting.

    Examples
    --------
    Use `FuncOpr(Offset, Offset.fractional)` to normalize every `Offset`
    appearing inside a `U1Basis`, `U1Span`, or `HilbertSpace`.

    In particular:

    - `fractional @ psi` rewrites one basis state
    - `fractional @ space` rewrites every basis state in a Hilbert space
    - `fractional @ t @ space` means `fractional(t(space))`
    """

    T: Type[_IrrepType]
    func: Callable[[_IrrepType], Union[_IrrepType, Multiple[_IrrepType]]]


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


@dispatch(U1Span, U1Span)  # type: ignore[no-redef]
def same_rays(a: U1Span, b: U1Span) -> bool:
    return set(a.rays().span) == set(b.rays().span)


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_eq(a: U1Basis, b: U1Basis) -> bool:
    return a.coef == b.coef and a.base == b.base


@dispatch(U1Basis, U1Basis)  # type: ignore[no-redef]
def operator_matmul(a: U1Basis, b: U1Basis) -> U1Basis:
    if not a.base:
        return b
    if not b.base:
        return a
    return U1Basis((a.coef * b.coef).simplify(), a.base + b.base)


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


@dispatch(HilbertSpace, U1Basis)  # type: ignore[no-redef]
def operator_or(space: HilbertSpace, state: U1Basis) -> HilbertSpace:
    if state in space.structure:
        return space
    return HilbertSpace.new((*space.elements(), state))


@dispatch(U1Basis, HilbertSpace)  # type: ignore[no-redef]
def operator_or(state: U1Basis, space: HilbertSpace) -> HilbertSpace:
    if state in space.structure:
        return space
    return HilbertSpace.new((state, *space.elements()))


@dataclass(frozen=True)
class ComposedOpr(Opr):
    """
    Finite composition of symbolic operators.

    `ComposedOpr` stores an ordered tuple of `Opr` instances and represents
    their algebraic composition. It is produced automatically by operator
    multiplication between operators:

    - `a @ b` returns `ComposedOpr((a, b))`

    Application order
    -----------------
    Although the tuple is stored as `(a, b)`, application follows standard
    operator-composition order rather than pipeline order:

    - `(a @ b) @ x == a(b(x))`

    More generally, if `ops == (o1, o2, ..., on)`, then applying the composed
    operator to `x` yields:

    - `o1(o2(...(on(x))...))`

    Operationally, this is implemented by applying the stored operators in
    reverse order at invocation time.

    This matches the usual reading of matrix multiplication and function
    composition, where the rightmost operator acts first.

    Flattening behavior
    -------------------
    Repeated composition is flattened structurally:

    - `(a @ b) @ c` stores `(a, b, c)`
    - `a @ (b @ c)` stores `(a, b, c)`

    so nested `ComposedOpr` objects are normalized into a single tuple of
    operators instead of building a deep binary tree. This keeps composition
    associative at the representation level as well as the semantic level.

    Interaction with object application
    ----------------------------------
    As with every `Opr`, `composed @ x` applies the operator to `x`. Each step
    is fed into the next one according to the order above.

    For plain objects `x`, the intermediate values are passed directly from one
    operator to the next.

    Interaction with `Multiple`
    ---------------------------
    `ComposedOpr` also supports `Multiple(base, coef)` inputs. In that case the
    composition acts on `base`, while scalar coefficients returned by
    intermediate operators are accumulated multiplicatively.

    Concretely, if an intermediate step returns:

    - a plain object `y`, composition continues with `y`
    - `Multiple(c, y)`, composition continues with `y` and multiplies the
      running coefficient by `c`

    The final result is returned as one `Multiple(total_coef, final_base)`.

    This is essential for symbolic transformations where some operators carry
    phase factors or representation coefficients.

    Algebraic scope
    ---------------
    `ComposedOpr` is intended for endomorphism-like symbolic operators in this
    module: operators that map supported symbolic objects back into the same
    symbolic universe. It assumes that composing the stored operators is valid
    on the given input type; if some step has no registered action for the
    intermediate object, dispatch will fail with `NotImplementedError`.

    Examples
    --------
    If `fractional = FuncOpr(Offset, Offset.fractional)` and `t` is an
    `AbelianOpr`, then:

    - `fractional @ t @ space`

    means:

    - `fractional(t(space))`

    not:

    - `t(fractional(space))`
    """

    ops: Tuple[Opr, ...]

    @override
    def invoke(self, v: _T, **kwargs) -> Union[_T, Multiple[_T]]:
        result = v
        for op in reversed(self.ops):
            result = op(result, **kwargs)
        return result


@dispatch(Opr, Opr)  # type: ignore[no-redef]
def operator_matmul(a: Opr, b: Opr) -> ComposedOpr:
    return ComposedOpr((a, b))


@dispatch(ComposedOpr, Opr)  # type: ignore[no-redef]
def operator_matmul(a: ComposedOpr, b: Opr) -> ComposedOpr:
    return ComposedOpr((*a.ops, b))


@dispatch(Opr, ComposedOpr)  # type: ignore[no-redef]
def operator_matmul(a: Opr, b: ComposedOpr) -> ComposedOpr:
    return ComposedOpr((a, *b.ops))


@dispatch(ComposedOpr, ComposedOpr)  # type: ignore[no-redef]
def operator_matmul(a: ComposedOpr, b: ComposedOpr) -> ComposedOpr:
    return ComposedOpr((*a.ops, *b.ops))
