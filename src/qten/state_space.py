from dataclasses import dataclass, replace, field
from typing import Callable, NamedTuple, Tuple, TypeVar, Generic, Union, Self, cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterator
from functools import lru_cache
from itertools import islice

from multipledispatch import dispatch  # type: ignore[import-untyped]

from .abstracts import Convertible, Span
from .validations import need_validation
from .geometries.spatials import (
    Spatial,
    ReciprocalLattice,
    Momentum,
    cartes,
)


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
    `StateSpace` is a collection of indices with additional information attached to the elements,
    for the case of TNS there are only two types of state spaces: `MomentumSpace` and `HilbertSpace`.
    `MomentumSpace` is needed because some tensors are better represented in momentum space, e.g. Hamiltonians
    with translational symmetry, while `HilbertSpace` is needed to represent local degrees of freedom, e.g. spin or fermionic modes.

    Attributes
    ----------
    structure : OrderedDict[Spatial, int]
        An ordered dictionary mapping each spatial component (e.g., `Offset`,
        `Momentum`) to its single flattened index.

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

    def __hash__(self):
        # TODO: Do we need to consider the order of the structure?
        return hash(tuple(self.structure.items()))

    def __getitem__(self, v: Union[int, slice, range]) -> Union[T, "StateSpace[T]"]:
        """
        Index into the state-space by element position.

        Parameters
        ----------
        `v` : `Union[int, slice, range]`
            - `int` returns a single spatial element by position (supports negative
              indices).
            - `slice` returns a new instance containing the selected elements in
              order, with slices re-packed to be contiguous.
            - `range` returns a new instance containing the elements at the given
              indices (in the range order), with slices re-packed to be contiguous.

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
            If `v` is not an `int`, `slice`, or `range`.
        """
        if isinstance(v, int):
            if v < 0:
                v += len(self.structure)
            if v < 0 or v >= len(self.structure):
                raise IndexError("StateSpace index out of range")
            return next(islice(self.structure.keys(), v, None))
        if isinstance(v, slice):
            keys = tuple(self.structure.keys())[v]
            new_structure = OrderedDict((k, self.structure[k]) for k in keys)
            return replace(self, structure=restructure(new_structure))
        if isinstance(v, range):
            keys = tuple(self.structure.keys())
            new_structure = OrderedDict((keys[i], self.structure[keys[i]]) for i in v)
            return replace(self, structure=restructure(new_structure))
        raise TypeError(
            f"StateSpace indices must be int, slice, or range, not {type(v)}"
        )

    def same_rays(self, other: "StateSpace") -> bool:
        """
        Check if this state space has the same rays as another, i.e., they have the same set of spatial keys regardless of order.

        Parameters
        ----------
        `other` : `StateSpace`
            The other state space to compare against.

        Returns
        -------
        `bool`
            `True` if both state spaces have the same rays, `False` otherwise.
        """
        return same_rays(self, other)

    def map(self, func: Callable[[T], T]) -> "StateSpace[T]":
        """
        Map the spatial elements of this state space using a provided function.

        Parameters
        ----------
        `func` : `Callable[[T], T]`
            A function that takes a spatial element and returns a transformed spatial element.

        Returns
        -------
        `StateSpace[T]`
            A new state space with the transformed spatial elements.
        """
        new_structure = OrderedDict()
        for k, s in self.structure.items():
            new_k = func(k)
            new_structure[new_k] = s
        return replace(self, structure=restructure(new_structure))

    def tensor_product(self, other: Self) -> Self:
        """
        Return the tensor product of this state space with another.

        Parameters
        ----------
        `other` : `StateSpace`
            The other state space to tensor with.

        Returns
        -------
        `StateSpace`
            A new state space representing the tensor product of the two.
        """
        raise NotImplementedError(f"Tensor product not implemented for {type(self)}!")


@StateSpace.add_conversion(StateSpace)
def state_space_to_state_space(s: StateSpace) -> StateSpace:
    """Identity conversion to allow mapping between different StateSpace subclasses."""
    return s


@dispatch(StateSpace, StateSpace)
def operator_matmul(a: StateSpace, b: StateSpace):
    return a.tensor_product(b)


def restructure(
    structure: OrderedDict[T, int],
) -> OrderedDict[T, int]:
    """
    Return a new `OrderedDict` with contiguous, ordered integer indices.

    Parameters
    ----------
    structure : `OrderedDict[Spatial, int]`
        The original structure with possibly non-contiguous indices.

    Returns
    -------
    `OrderedDict[Spatial, int]`
        The restructured `OrderedDict` with contiguous, ordered indices.
    """
    return OrderedDict((k, i) for i, k in enumerate(structure.keys()))


@lru_cache
def permutation_order(src: "StateSpace", dest: "StateSpace") -> Tuple[int, ...]:
    """
    Return the permutation of `src` sectors needed to match `dest` sector order.

    This returns a per-sector permutation: each entry corresponds to a key in
    `dest.structure` and gives the index of the same key in `src.structure`.
    The mapping is directly index-based on the current `StateSpace` structure
    and can be used to reorder sector-aligned data.

    Parameters
    ----------
    src : `StateSpace`
        The source state space defining the original ordering.
    dest : `StateSpace`
        The destination state space defining the target ordering.

    Returns
    -------
    `Tuple[int, ...]`
        Sector indices mapping each key in `dest` to its position in `src`.

    Raises
    ------
    `ValueError`
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
    Return indices mapping `sub` into `sup` (assumes `sub` ⊆ `sup`).

    Parameters
    ----------
    sub : `StateSpace`
        The subspace whose indices are to be embedded.
    sup : `StateSpace`
        The superspace providing the full index set.

    Returns
    -------
    `Tuple[int, ...]`
        Flattened indices mapping `sub` into `sup`.
    """
    indices: list[int] = []
    sup_indices = sup.structure
    for key, _ in sub.structure.items():
        if key not in sup_indices:
            raise ValueError(f"Key {key} not found in superspace")
        indices.append(sup_indices[key])
    return tuple(indices)


# TODO: We can put @lru_cache if the hashing of StateSpace is well defined
@dispatch(StateSpace, StateSpace)
def same_rays(a: StateSpace, b: StateSpace) -> bool:
    return set(a.structure.keys()) == set(b.structure.keys())


@dispatch(StateSpace, StateSpace)
def operator_add(a: StateSpace, b: StateSpace):
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


@dispatch(StateSpace, StateSpace)
def operator_sub(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        raise ValueError(
            f"Cannot subtract StateSpaces of different types: {type(a)} and {type(b)}!"
        )
    new_structure = OrderedDict(
        ((k, v) for k, v in a.structure.items() if k not in b.structure)
    )
    return type(a)(structure=restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_or(a: StateSpace, b: StateSpace):
    return a + b


@dispatch(StateSpace, StateSpace)
def operator_and(a: StateSpace, b: StateSpace):
    if type(a) is not type(b):
        raise ValueError(
            f"Cannot intersect StateSpaces of different types: {type(a)} and {type(b)}!"
        )
    new_structure = OrderedDict(
        ((k, v) for k, v in a.structure.items() if k in b.structure)
    )
    return type(a)(structure=restructure(new_structure))


@dispatch(StateSpace, StateSpace)
def operator_eq(a: StateSpace, b: StateSpace):
    return a.structure == b.structure


@need_validation()
@dataclass(frozen=True)
class MomentumSpace(StateSpace[Momentum]):
    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__

    def __str__(self):
        return f"MomentumSpace({self.dim})"

    def __repr__(self):
        header = f"{str(self)}:\n"
        body = "\t" + "\n\t".join(
            [f"{n}: {k}" for n, k in enumerate(self.structure.keys())]
        )
        return header + body


@Momentum.add_conversion(StateSpace)
def momentum_to_momentumspace(k: Momentum) -> StateSpace:
    """Convert a `Momentum` to a `MomentumSpace` containing only that momentum."""
    structure = OrderedDict({k: 0})
    return MomentumSpace(structure=structure)


# Register the conversion from `Momentum` to `MomentumSpace`.
Momentum.add_conversion(MomentumSpace)(
    cast(Callable[[Momentum], MomentumSpace], momentum_to_momentumspace)
)


@lru_cache
def brillouin_zone(lattice: ReciprocalLattice) -> MomentumSpace:
    elements = cartes(lattice)
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
    `BroadcastSpace` represents an axis that behaves like a size-1 axis under
    tensor broadcasting. It is used to model dimensions introduced by
    `unsqueeze`, `None` indexing, or other operations where data may be
    expanded without introducing a concrete physical basis.

    Structure semantics
    -------------------
    `BroadcastSpace` stores a private singleton marker in `structure`:
    `OrderedDict({_BAxis(): 0})`.
    This keeps the axis dimension at `1` while still providing a stable
    coordinate for structure-based index mapping helpers.

    Implication for index mapping
    -----------------------------
    For `BroadcastSpace -> BroadcastSpace`, `embedding_order(...)` resolves to
    `(0,)`. This is intentional and allows consumers that build runtime index
    coordinates from structure mappings to treat broadcast axes as a singleton
    axis at position `0`.

    The `_BAxis` marker is internal implementation detail. It is not a physical
    basis element and should not be relied on outside broadcast-axis plumbing.

    Compatibility rules
    -------------------
    Multipledispatch rules in this module treat `BroadcastSpace` as compatible
    with any `StateSpace` in `same_rays(...)`, and as neutral in
    `operator_add(...)`:
    - `BroadcastSpace + X -> X`
    - `X + BroadcastSpace -> X`
    - `BroadcastSpace + BroadcastSpace -> BroadcastSpace`

    This makes it suitable as a placeholder axis that can be promoted to a
    concrete state space during alignment/broadcast operations.
    """

    # Internal singleton marker so structure-based mappings can emit index 0.
    structure: OrderedDict = field(
        default_factory=lambda: OrderedDict({_BAxis(): 0}), init=False
    )

    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__

    def __repr__(self):
        return "BroadcastSpace"

    __str__ = __repr__


@dispatch(BroadcastSpace, BroadcastSpace)  # type: ignore[no-redef]
def same_rays(a: BroadcastSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(StateSpace, BroadcastSpace)  # type: ignore[no-redef]
def same_rays(a: StateSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(BroadcastSpace, StateSpace)  # type: ignore[no-redef]
def same_rays(a: BroadcastSpace, b: StateSpace) -> bool:
    return True


# The set union of any StateSpace with a BroadcastSpace is a BroadcastSpace
@dispatch(BroadcastSpace, BroadcastSpace)  # type: ignore[no-redef]
def operator_add(a: BroadcastSpace, b: BroadcastSpace):
    return BroadcastSpace()


@dispatch(StateSpace, BroadcastSpace)  # type: ignore[no-redef]
def operator_add(a: StateSpace, b: BroadcastSpace):
    return a


@dispatch(BroadcastSpace, StateSpace)  # type: ignore[no-redef]
def operator_add(a: BroadcastSpace, b: StateSpace):
    return b


@need_validation()
@dataclass(frozen=True)
class IndexSpace(StateSpace[int]):
    """
    A simple state space where the spatial elements are just integer indices.
    This can be useful for representing generic tensor dimensions that don't
    have a specific physical interpretation, such as the virtual bond dimension in a TNS.
    """

    __hash__ = StateSpace.__hash__

    @staticmethod
    def linear(size: int) -> "IndexSpace":
        """
        Build a contiguous index space of length `size`.

        The resulting space contains integer keys `0..size-1`, each mapped to
        the same integer index.

        Parameters
        ----------
        `size` : `int`
            Number of indices in the space.

        Returns
        -------
        `IndexSpace`
            A contiguous `IndexSpace` with canonical linear ordering.

        Raises
        ------
        `ValueError`
            If `size` is negative.
        """
        if size < 0:
            raise ValueError("IndexSpace size must be non-negative.")
        return IndexSpace(OrderedDict((i, i) for i in range(size)))

    def __str__(self):
        return f"IndexSpace(size={self.dim})"

    def __repr__(self):
        return str(self)


class StateSpaceFactorization(NamedTuple):
    """
    Ruleset for factorizing one `StateSpace`-like tensor dimension.

    Attributes
    ----------
    `factorized` : `Tuple[StateSpace, ...]`
        Target factor spaces in `torch.Tensor.reshape` ordering.
    `align_dim` : `StateSpace`
        A permutation of the original dimension whose flattened order is
        compatible with reshaping into `factorized`.
    """

    factorized: Tuple[StateSpace, ...]
    align_dim: StateSpace
