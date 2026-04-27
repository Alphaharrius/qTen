"""
Indexed symbolic state-space containers.

This module defines ordered finite state spaces used as tensor dimensions in
QTen. [`StateSpace`][qten.symbolics.state_space.StateSpace] stores arbitrary
spatial or symbolic elements with stable integer indices, while specialized
spaces such as [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and
[`BzPath`][qten.symbolics.state_space.BzPath] describe reciprocal-space grids
and paths.

Repository usage
----------------
Use state spaces as labelled tensor dimensions and as the finite index sets
that connect symbolic geometry to numeric tensor data. Hilbert-space basis
objects live in [`qten.symbolics.hilbert_space`][qten.symbolics.hilbert_space].
"""

from dataclasses import dataclass, replace, field
from typing import Any, Callable, NamedTuple, Tuple, TypeVar, Generic, Union, Self, cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from functools import lru_cache
from itertools import islice

from multimethod import multimethod

from ..abstracts import Convertible, Operable, Span
from ..validations import need_validation
from ..geometries import Momentum, ReciprocalLattice
from ..geometries.spatials import Spatial


T = TypeVar("T")


def _check_contiguous_indices(s: "StateSpace") -> None:
    """
    Validator function to check that a state space's structure indices are contiguous.

    This is a standalone function version of `ValidateContiguousIndices.validate`
    for use in `@need_validation` without needing to define a separate class.
    """
    values = tuple(s.structure.values())
    n = len(values)
    if values != tuple(range(n)):
        raise ValueError(
            "StateSpace.structure values must match insertion order as contiguous "
            "indices 0..n-1."
        )


@need_validation(_check_contiguous_indices)
@dataclass(frozen=True)
class StateSpace(Spatial, Convertible, Generic[T], Span[T]):
    """
    [`StateSpace`][qten.symbolics.state_space.StateSpace] is a collection of indices with additional information attached to the elements,
    for the case of TNS there are only two types of state spaces: [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] and [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace].
    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] is needed because some tensors are better represented in momentum space, e.g. Hamiltonians
    with translational symmetry, while [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace] is needed to represent local degrees of freedom, e.g. spin or fermionic modes.

    Attributes
    ----------
    structure : OrderedDict[Spatial, int]
        An ordered dictionary mapping each spatial component (e.g., [`Offset`][qten.geometries.spatials.Offset],
        Momentum) to its single flattened index.

    dim : int
        The total dimension of the state space, calculated as the count of elements regardless of their lengths.
    """

    structure: OrderedDict[T, int]
    """
    An ordered dictionary mapping each spatial component (e.g., `Offset`,
    `Momentum`) to its single flattened index.
    """

    @property
    def dim(self) -> int:
        """The total size of the vector space."""
        return len(self.structure)

    @override
    def elements(self) -> Tuple[T, ...]:
        """Return the spatial elements as a tuple."""
        return tuple(self.structure.keys())

    def __len__(self) -> int:
        """Return the number of spatial elements."""
        return len(self.structure)

    def __iter__(self) -> Iterator[T]:
        """Iterate over spatial elements."""
        return iter(self.structure.keys())

    def __hash__(self) -> int:
        """
        Return a hash derived from the ordered state-space structure.

        The hash is computed from `tuple(self.structure.items())`, so it
        depends on both the spatial elements and their stored integer indices in
        insertion order. Two state spaces with the same elements but a different
        order, or with the same elements mapped to different integer positions,
        intentionally produce different hashes.

        This custom implementation is required because `structure` is an
        `OrderedDict`, which is mutable and unhashable by itself. The state-space
        dataclasses are frozen, so the tuple snapshot is stable as long as the
        contained spatial elements are themselves hashable and immutable.

        Returns
        -------
        int
            Hash value for the ordered `(element, index)` mapping.
        """
        return hash(tuple(self.structure.items()))

    def __getitem__(
        self, key: Union[int, slice, range, Sequence[int]]
    ) -> Union[T, "StateSpace[T]"]:
        """
        Index into the state-space by element position.

        Supported forms
        ---------------
        `space[i]` returns a single spatial element by position, including
        negative indices. `space[start:stop:step]` returns a new space with the
        selected elements in slice order. `space[range(...)]` returns a new
        space with the range-selected elements. `space[[i, j, ...]]` returns a
        new space with elements in explicit index order.

        Parameters
        ----------
        key : Union[int, slice, range, Sequence[int]]
            Index selector. Sequences must contain unique integers; negative
            integers are normalized relative to `len(self)`.

        Returns
        -------
        Spatial or StateSpace
            A spatial element for `int` indexing, otherwise a new instance of the
            same class containing the selected elements. Any extra dataclass fields
            on subclasses are preserved in the returned instance.

        Raises
        ------
        IndexError
            If an integer index is out of bounds.
        TypeError
            If `key` is not an `int`, `slice`, `range`, or sequence of integers.
        ValueError
            If a sequence selector contains duplicate indices.
        """
        if isinstance(key, int):
            if key < 0:
                key += len(self.structure)
            if key < 0 or key >= len(self.structure):
                raise IndexError("StateSpace index out of range")
            return next(islice(self.structure.keys(), key, None))
        if isinstance(key, slice):
            keys = tuple(self.structure.keys())[key]
            new_structure = OrderedDict((k, self.structure[k]) for k in keys)
            return replace(self, structure=restructure(new_structure))
        if isinstance(key, range):
            keys = tuple(self.structure.keys())
            new_structure = OrderedDict((keys[i], self.structure[keys[i]]) for i in key)
            return replace(self, structure=restructure(new_structure))
        if isinstance(key, Sequence) and not isinstance(key, (str, bytes)):
            keys = tuple(self.structure.keys())
            indices: list[int] = []
            seen = set()
            for idx in key:
                if isinstance(idx, bool) or not isinstance(idx, int):
                    raise TypeError(
                        "StateSpace sequence indices must contain only integers"
                    )
                normalized = idx
                if normalized < 0:
                    normalized += len(keys)
                if normalized < 0 or normalized >= len(keys):
                    raise IndexError("StateSpace index out of range")
                if normalized in seen:
                    raise ValueError("StateSpace sequence indices must be unique")
                seen.add(normalized)
                indices.append(normalized)
            new_structure = OrderedDict(
                (keys[i], self.structure[keys[i]]) for i in indices
            )
            return replace(self, structure=restructure(new_structure))
        raise TypeError(
            "StateSpace indices must be int, slice, range, or a sequence of ints, "
            f"not {type(key)}"
        )

    def same_rays(self, other: "StateSpace") -> bool:
        """
        Check if this state space has the same rays as another, i.e., they have the same set of spatial keys regardless of order.

        Parameters
        ----------
        other : StateSpace
            The other state space to compare against.

        Returns
        -------
        bool
            True if both state spaces have the same rays, `False` otherwise.
        """
        return same_rays(self, other)

    def map(self, func: Callable[[T], T]) -> "Self":
        """
        Map the spatial elements of this state space using a provided function.

        Parameters
        ----------
        func : Callable[[T], T]
            A function that takes a spatial element and returns a transformed spatial element.

        Returns
        -------
        Self
            A new state space with the transformed spatial elements.
        """
        new_structure = OrderedDict()
        for k, s in self.structure.items():
            new_k = func(k)
            new_structure[new_k] = s
        return replace(self, structure=restructure(new_structure))

    def filter(self, pred: Callable[[T], bool]) -> "Self":
        """
        Return the subspace containing elements where `pred(element)` is `True`.

        Parameters
        ----------
        pred : Callable[[T], bool]
            Predicate applied to each element in basis order.

        Returns
        -------
        Self
            A new state space of the same concrete type containing only the
            selected elements, with indices repacked contiguously.
        """
        new_structure = OrderedDict(
            (k, s) for k, s in self.structure.items() if pred(k)
        )
        return replace(self, structure=restructure(new_structure))

    def tensor_product(self, other: Self) -> Self:
        """
        Return the tensor-product state space of this space and another space.

        This method defines the protocol used by the `@` operator on
        [`StateSpace`][qten.symbolics.state_space.StateSpace] instances. The
        base class cannot construct a generic product because it does not know
        how to combine two elements of type `T`. Concrete subclasses must
        implement the element-level product and rebuild a contiguous
        `structure` for the resulting basis.

        Implementations should preserve deterministic product ordering. The
        convention used by concrete tensor-product spaces in QTen is the
        Cartesian product order of `self.elements()` and `other.elements()`,
        where elements from `self` vary slowest and elements from `other` vary
        fastest. The returned space should be the same concrete state-space
        family when the operation is closed over that family.

        Parameters
        ----------
        other : StateSpace
            Right-hand tensor factor. Implementations may require `other` to be
            the same concrete state-space type as `self`.

        Returns
        -------
        StateSpace
            New state space representing `self ⊗ other`, with contiguous
            integer indices in the implementation-defined product order.

        Raises
        ------
        NotImplementedError
            Always raised by the base class. Subclasses that support tensor
            products must override this method.
        """
        raise NotImplementedError(f"Tensor product not implemented for {type(self)}!")

    @multimethod
    def extract(self, info_type: type[Any]) -> Any:
        """
        Extract an object implied by the elements of this state space.

        Subclasses may register specialized implementations for supported
        target types.

        Parameters
        ----------
        info_type : type[Any]
            Type of metadata or object to extract from this state space.

        Returns
        -------
        Any
            Extracted object produced by a registered specialization.

        Raises
        ------
        NotImplementedError
            If no extraction rule is registered for `info_type`.
        """
        raise NotImplementedError(
            f"Extraction of {info_type} from {type(self)} is not supported!"
        )


@StateSpace.add_conversion(StateSpace)
def _(s: StateSpace) -> StateSpace:
    """Identity conversion to allow mapping between different StateSpace subclasses."""
    return s


@Operable.__matmul__.register
def _(a: StateSpace, b: StateSpace):
    return a.tensor_product(b)


def restructure(
    structure: OrderedDict[T, int],
) -> OrderedDict[T, int]:
    """
    Return a new `OrderedDict` with contiguous, ordered integer indices.

    Parameters
    ----------
    structure : OrderedDict[Spatial, int]
        The original structure with possibly non-contiguous indices.

    Returns
    -------
    OrderedDict[Spatial, int]
        The restructured `OrderedDict` with contiguous, ordered indices.
    """
    return OrderedDict((k, i) for i, k in enumerate(structure.keys()))


@lru_cache
def permutation_order(src: "StateSpace", dest: "StateSpace") -> Tuple[int, ...]:
    """
    Return the permutation of `src` sectors needed to match `dest` sector order.

    This returns a per-sector permutation: each entry corresponds to a key in
    `dest.structure` and gives the index of the same key in `src.structure`.
    The mapping is directly index-based on the current [`StateSpace`][qten.symbolics.state_space.StateSpace] structure
    and can be used to reorder sector-aligned data.

    Parameters
    ----------
    src : StateSpace
        The source state space defining the original ordering.
    dest : StateSpace
        The destination state space defining the target ordering.

    Returns
    -------
    Tuple[int, ...]
        Sector indices mapping each key in `dest` to its position in `src`.

    Raises
    ------
    ValueError
        If a destination sector key is missing from the source state space.
    """
    src_indices = src.structure
    dest_keys = dest.structure.keys()
    try:
        return tuple(src_indices[k] for k in dest_keys)
    except KeyError:
        missing = [k for k in dest_keys if k not in src_indices]
        raise ValueError(
            "Cannot build permutation order: destination contains keys not present "
            f"in source: {missing}"
        ) from None


@lru_cache
def embedding_order(sub: StateSpace, sup: StateSpace) -> Tuple[int, ...]:
    """
    Return the positions of `sub` elements inside `sup`.

    The returned tuple has one integer for each element of `sub`, in
    `sub.elements()` order. Each integer is the stored basis index of that same
    element in `sup.structure`. This is useful when a tensor axis represents a
    smaller state space and its data must be scattered into, gathered from, or
    aligned against a larger state space.

    For example, if `sup` contains elements `(a, b, c)` with indices
    `(0, 1, 2)` and `sub` contains `(c, a)`, then this function returns
    `(2, 0)`. The result follows the order of `sub`, not the order of `sup`.

    Parameters
    ----------
    sub : StateSpace
        State space whose elements should be located in `sup`.
    sup : StateSpace
        State space providing the reference element-to-index mapping.

    Returns
    -------
    Tuple[int, ...]
        Indices in `sup` corresponding to each element of `sub`, ordered like
        `sub.elements()`.

    Raises
    ------
    ValueError
        If any element of `sub` is not present as a key in `sup.structure`.
    """
    indices: list[int] = []
    sup_indices = sup.structure
    for key, _ in sub.structure.items():
        if key not in sup_indices:
            raise ValueError(f"Key {key} not found in superspace")
        indices.append(sup_indices[key])
    return tuple(indices)


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
@multimethod
def same_rays(a: StateSpace, b: StateSpace) -> bool:
    """
    Check whether two state spaces span the same ray representatives.

    Parameters
    ----------
    a : StateSpace
        First state space.
    b : StateSpace
        Second state space.

    Returns
    -------
    bool
        `True` when the two state spaces contain the same set of basis keys,
        ignoring ordering and stored integer indices.
    """
    return set(a.structure.keys()) == set(b.structure.keys())


@Operable.__add__.register
def _(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        raise ValueError(
            f"Cannot add StateSpaces of different types: {type(a)} and {type(b)}!"
        )
    new_structure = OrderedDict(
        (
            *a.structure.items(),
            *((k, v) for k, v in b.structure.items() if k not in a.structure),
        )
    )
    return type(a)(structure=restructure(new_structure))


@Operable.__sub__.register
def _(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        raise ValueError(
            f"Cannot subtract StateSpaces of different types: {type(a)} and {type(b)}!"
        )
    new_structure = OrderedDict(
        ((k, v) for k, v in a.structure.items() if k not in b.structure)
    )
    return type(a)(structure=restructure(new_structure))


@Operable.__or__.register
def _(a: StateSpace, b: StateSpace):
    return a + b


@Operable.__and__.register
def _(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        raise ValueError(
            f"Cannot intersect StateSpaces of different types: {type(a)} and {type(b)}!"
        )
    new_structure = OrderedDict(
        ((k, v) for k, v in a.structure.items() if k in b.structure)
    )
    return type(a)(structure=restructure(new_structure))


@Operable.__eq__.register
def _(a: StateSpace, b: StateSpace):
    return a.structure == b.structure


@need_validation()
@dataclass(frozen=True)
class MomentumSpace(StateSpace[Momentum]):
    """
    Ordered state space of momentum points over one reciprocal lattice.

    [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] is the
    reciprocal-space specialization of [`StateSpace`][qten.symbolics.state_space.StateSpace]. Its `structure`
    keys are [`Momentum`][qten.geometries.spatials.Momentum] objects and its
    values are the corresponding integer basis indices.

    Notes
    -----
    The class inherits the custom hashing behavior from
    [`StateSpace`][qten.symbolics.state_space.StateSpace] so instances remain
    hashable despite storing an `OrderedDict`.
    """

    def __hash__(self) -> int:
        """
        Return a hash derived from the ordered momentum structure.

        [`MomentumSpace`][qten.symbolics.state_space.MomentumSpace] uses the
        same structure-based hash as
        [`StateSpace`][qten.symbolics.state_space.StateSpace]. The hash includes
        every [`Momentum`][qten.geometries.spatials.Momentum] key and its stored
        integer basis index in insertion order. This means two momentum spaces
        with the same momentum set but different basis order hash differently.

        Returns
        -------
        int
            Hash value for the ordered `(Momentum, index)` mapping.
        """
        return StateSpace.__hash__(self)

    def __str__(self):
        """
        Return a compact momentum-space summary.

        Returns
        -------
        str
            Summary containing the momentum-space dimension.
        """
        return f"MomentumSpace({self.dim})"

    def __repr__(self):
        """
        Return a multiline representation with indexed momentum elements.

        Returns
        -------
        str
            Header plus one line per momentum element in basis order.
        """
        header = f"{str(self)}:\n"
        body = "\t" + "\n\t".join(
            [f"{n}: {k}" for n, k in enumerate(self.structure.keys())]
        )
        return header + body


@dataclass(frozen=True)
class BzPath:
    """
    A Brillouin-zone path through high-symmetry waypoints.

    Attributes
    ----------
    k_space : MomentumSpace
        Unique momentum points sampled along the path.
    labels : tuple
        Labels for the waypoints in path order.
    waypoint_indices : tuple
        Indices into the dense path where each waypoint occurs.
    path_order : tuple
        For each dense path sample, index of the corresponding unique momentum
        in `k_space`.
    path_positions : tuple
        Cumulative Cartesian arc-length coordinate for each dense path sample.
    """

    k_space: MomentumSpace
    """
    Unique momentum points sampled along the path, with duplicates across
    segments removed while preserving the path's effective traversal order.
    """
    labels: tuple
    """
    Labels for the waypoints in path order, typically high-symmetry point
    names used for plotting.
    """
    waypoint_indices: tuple
    """
    Indices into the dense path where each waypoint occurs, suitable for axis
    ticks or segment markers in band-structure plots.
    """
    path_order: tuple
    """
    For each dense path sample, index of the corresponding unique momentum in
    `k_space`. This maps the full piecewise-linear path back onto the unique
    momentum list.
    """
    path_positions: tuple
    """
    Cumulative Cartesian arc-length coordinate for each dense path sample,
    used as the continuous x-axis parameter along the path.
    """


@Momentum.add_conversion(StateSpace)
def momentum_to_momentumspace(k: Momentum) -> StateSpace:
    """
    Convert one momentum point to a one-element momentum space.

    Parameters
    ----------
    k : Momentum
        Momentum point to wrap.

    Returns
    -------
    MomentumSpace
        One-element momentum space containing `k`.
    """
    structure = OrderedDict({k: 0})
    return MomentumSpace(structure=structure)


# Register the conversion from `Momentum` to `MomentumSpace`.
Momentum.add_conversion(MomentumSpace)(
    cast(Callable[[Momentum], MomentumSpace], momentum_to_momentumspace)
)


@StateSpace.extract.register
def _(space: MomentumSpace, _: type[ReciprocalLattice]) -> ReciprocalLattice:
    """
    Extract the unique reciprocal lattice shared by all momenta.

    Parameters
    ----------
    space : MomentumSpace
        Momentum space whose elements should share one reciprocal lattice.
    _ : type[ReciprocalLattice]
        Extraction target type.

    Returns
    -------
    ReciprocalLattice
        The unique reciprocal lattice used by every momentum in `space`.

    Raises
    ------
    ValueError
        If the space is empty or contains momenta from multiple reciprocal
        lattices.
    """
    if not space.elements():
        raise ValueError("MomentumSpace is empty")

    reciprocal_lattices = {k.space for k in space}
    if len(reciprocal_lattices) != 1:
        raise ValueError("MomentumSpace does not have a unique ReciprocalLattice")

    return reciprocal_lattices.pop()


@lru_cache
def brillouin_zone(lattice: ReciprocalLattice) -> MomentumSpace:
    """
    Enumerate the discrete Brillouin-zone momenta of a reciprocal lattice.

    Parameters
    ----------
    lattice : ReciprocalLattice
        Reciprocal lattice whose Cartesian momentum samples should be
        collected.

    Returns
    -------
    MomentumSpace
        Momentum space whose ordering follows
        [`ReciprocalLattice.cartes()`][qten.geometries.spatials.ReciprocalLattice.cartes].
    """
    elements = lattice.cartes()
    structure = OrderedDict((el, n) for n, el in enumerate(elements))
    return MomentumSpace(structure=structure)


@dataclass(frozen=True)
class _BAxis:
    pass


@need_validation()
@dataclass(frozen=True)
class BroadcastSpace(StateSpace[_BAxis]):
    """
    Metadata marker for singleton/broadcast tensor axes.

    Design intent
    -------------
    [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] represents an axis that behaves like a size-1 axis under
    tensor broadcasting. It is used to model dimensions introduced by
    `unsqueeze`, `None` indexing, or other operations where data may be
    expanded without introducing a concrete physical basis.

    Structure semantics
    -------------------
    [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] stores a private singleton marker in `structure`:
    `OrderedDict({_BAxis(): 0})`.
    This keeps the axis dimension at `1` while still providing a stable
    coordinate for structure-based index mapping helpers.

    Implication for index mapping
    -----------------------------
    For `BroadcastSpace -> BroadcastSpace`, [`embedding_order(...)`][qten.symbolics.state_space.embedding_order] resolves to
    `(0,)`. This is intentional and allows consumers that build runtime index
    coordinates from structure mappings to treat broadcast axes as a singleton
    axis at position `0`.

    The `_BAxis` marker is internal implementation detail. It is not a physical
    basis element and should not be relied on outside broadcast-axis plumbing.

    Compatibility rules
    -------------------
    Multimethod rules in this module treat [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] as compatible
    with any [`StateSpace`][qten.symbolics.state_space.StateSpace] in [`same_rays(...)`][qten.symbolics.state_space.same_rays], and as neutral in
    `__add__(...)`. The neutral-addition behavior is `BroadcastSpace + X -> X`,
    `X + BroadcastSpace -> X`, and
    `BroadcastSpace + BroadcastSpace -> BroadcastSpace`.

    This makes it suitable as a placeholder axis that can be promoted to a
    concrete state space during alignment/broadcast operations.

    Attributes
    ----------
    structure : OrderedDict
        Private singleton mapping `OrderedDict({_BAxis(): 0})` used to encode a
        size-1 broadcast axis.
    """

    # Internal singleton marker so structure-based mappings can emit index 0.
    structure: OrderedDict = field(
        default_factory=lambda: OrderedDict({_BAxis(): 0}), init=False
    )
    """
    Private singleton mapping `OrderedDict({_BAxis(): 0})` used to encode a
    size-1 broadcast axis. The stored marker is internal and exists only so
    structure-based helpers can consistently refer to the unique broadcast slot.
    """

    def __hash__(self) -> int:
        """
        Return a hash for the singleton broadcast-axis structure.

        [`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace] hashes the
        same ordered structure snapshot as
        [`StateSpace`][qten.symbolics.state_space.StateSpace]. Because the
        structure always contains one private `_BAxis` marker at index `0`, the
        hash represents this singleton broadcast dimension rather than a
        physical basis element.

        Returns
        -------
        int
            Hash value for the ordered singleton `(_BAxis(), 0)` mapping.
        """
        return StateSpace.__hash__(self)

    def __repr__(self):
        """
        Return the broadcast-axis marker representation.

        Returns
        -------
        str
            The literal string `"BroadcastSpace"`.
        """
        return "BroadcastSpace"

    __str__ = __repr__


@same_rays.register
def _(a: BroadcastSpace, b: BroadcastSpace) -> bool:
    return True


@same_rays.register
def _(a: StateSpace, b: BroadcastSpace) -> bool:
    return True


@same_rays.register
def _(a: BroadcastSpace, b: StateSpace) -> bool:
    return True


# The set union of any StateSpace with a BroadcastSpace is a BroadcastSpace
@Operable.__add__.register
def _(a: BroadcastSpace, b: BroadcastSpace):
    return BroadcastSpace()


@Operable.__add__.register
def _(a: StateSpace, b: BroadcastSpace):
    return a


@Operable.__add__.register
def _(a: BroadcastSpace, b: StateSpace):
    return b


@need_validation()
@dataclass(frozen=True)
class IndexSpace(StateSpace[int]):
    """
    A simple state space where the spatial elements are just integer indices.
    This can be useful for representing generic tensor dimensions that don't
    have a specific physical interpretation, such as the virtual bond dimension in a TNS.
    """

    def __hash__(self) -> int:
        """
        Return a hash derived from the ordered integer-index structure.

        [`IndexSpace`][qten.symbolics.state_space.IndexSpace] uses the same
        structure-based hash as
        [`StateSpace`][qten.symbolics.state_space.StateSpace]. The hash includes
        each integer element and its stored basis index in order. For spaces
        produced by [`linear(size)`][qten.symbolics.state_space.IndexSpace.linear],
        this is the hash of the canonical mapping `0 -> 0`, `1 -> 1`, and so on.

        Returns
        -------
        int
            Hash value for the ordered `(int, index)` mapping.
        """
        return StateSpace.__hash__(self)

    @staticmethod
    def linear(size: int) -> "IndexSpace":
        """
        Build a contiguous index space of length [`size`][qten.geometries.spatials.ReciprocalLattice.size].

        The resulting space contains integer keys `0..size-1`, each mapped to
        the same integer index.

        Parameters
        ----------
        size : int
            Number of indices in the space.

        Returns
        -------
        IndexSpace
            A contiguous [`IndexSpace`][qten.symbolics.state_space.IndexSpace] with canonical linear ordering.

        Raises
        ------
        ValueError
            If [`size`][qten.geometries.spatials.ReciprocalLattice.size] is negative.
        """
        if size < 0:
            raise ValueError("IndexSpace size must be non-negative.")
        return IndexSpace(OrderedDict((i, i) for i in range(size)))

    def __str__(self):
        """
        Return a compact index-space summary.

        Returns
        -------
        str
            Summary containing the index-space size.
        """
        return f"IndexSpace(size={self.dim})"

    def __repr__(self):
        """
        Return the developer representation of this index space.

        Returns
        -------
        str
            Same value as `str(self)`.
        """
        return str(self)


class StateSpaceFactorization(NamedTuple):
    """
    Ruleset for factorizing one [`StateSpace`][qten.symbolics.state_space.StateSpace]-like tensor dimension.

    Attributes
    ----------
    factorized : Tuple[StateSpace, ...]
        Target factor spaces in `torch.Tensor.reshape` ordering.
    align_dim : StateSpace
        A permutation of the original dimension whose flattened order is
        compatible with reshaping into `factorized`.
    """

    factorized: Tuple[StateSpace, ...]
    align_dim: StateSpace
