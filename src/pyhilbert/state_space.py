from dataclasses import dataclass, replace, field
from typing import Callable, NamedTuple, Tuple, TypeVar, Generic, Union, Self, cast
from typing_extensions import override
from collections import OrderedDict
from collections.abc import Iterator
from functools import lru_cache
from itertools import islice

from multipledispatch import dispatch  # type: ignore[import-untyped]

from .abstracts import Convertible
from .spatials import (
    Spatial,
    ReciprocalLattice,
    Momentum,
    cartes,
)


T = TypeVar("T")


@dataclass(frozen=True)
class StateSpace(Spatial, Convertible, Generic[T]):
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

    def __post_init__(self) -> None:
        values = tuple(self.structure.values())
        if any((not isinstance(v, int)) or isinstance(v, bool) for v in values):
            raise TypeError("StateSpace.structure values must be integer indices.")
        n = len(values)
        if set(values) != set(range(n)):
            raise ValueError(
                "StateSpace.structure values must form a contiguous index set 0..n-1."
            )

    @property
    def dim(self) -> int:
        """The total size of the vector space."""
        return len(self.structure)

    def elements(self) -> Tuple[T, ...]:
        """Return the spatial elements as a tuple."""
        return tuple(k for k in self.structure.keys())

    def __len__(self) -> int:
        """Return the number of spatial elements."""
        return len(self.structure)

    def __iter__(self) -> Iterator[T]:
        """Iterate over spatial elements."""
        return iter(k for k, _ in self.structure.items())

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

    def same_span(self, other: "StateSpace") -> bool:
        """
        Check if this state space has the same span as another, i.e., they have the same set of spatial keys regardless of order.

        Parameters
        ----------
        `other` : `StateSpace`
            The other state space to compare against.

        Returns
        -------
        `bool`
            `True` if both state spaces have the same span, `False` otherwise.
        """
        return same_span(self, other)

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
    It does not expand slices; use `flat_permutation_order` to get element-wise
    indices for reordering a flattened tensor.

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
    order_table = {k: n for n, k in enumerate(src.structure.keys())}
    missing = [k for k in dest.structure.keys() if k not in order_table]
    if missing:
        raise ValueError(
            "Cannot build permutation order: destination contains keys not present "
            f"in source: {missing}"
        )
    return tuple(order_table[k] for k in dest.structure.keys())


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
def same_span(a: StateSpace, b: StateSpace) -> bool:
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
class BroadcastSpace(StateSpace[Spatial]):
    """
    Metadata marker for singleton/broadcast tensor axes.

    Design intent
    -------------
    `BroadcastSpace` represents an axis that behaves like a size-1 axis under
    tensor broadcasting. It is used to model dimensions introduced by
    `unsqueeze`, `None` indexing, or other operations where data may be
    expanded without introducing a concrete physical basis.

    `dim` behavior
    --------------
    `BroadcastSpace.dim` is intentionally defined as `1`.
    This means code that derives expected tensor shapes from
    `tuple(dim.dim for dim in dims)` will naturally treat broadcast axes as
    singleton axes.

    Structure semantics
    -------------------
    The inherited `structure` remains empty (`OrderedDict()`), because a
    broadcast axis has no concrete basis elements of its own. In other words:
    - `structure` encodes basis content (none for broadcast axes)
    - `dim` encodes broadcast shape semantics (singleton axis => `1`)

    Compatibility rules
    -------------------
    Multipledispatch rules in this module treat `BroadcastSpace` as compatible
    with any `StateSpace` in `same_span(...)`, and as neutral in
    `operator_add(...)`:
    - `BroadcastSpace + X -> X`
    - `X + BroadcastSpace -> X`
    - `BroadcastSpace + BroadcastSpace -> BroadcastSpace`

    This makes it suitable as a placeholder axis that can be promoted to a
    concrete state space during alignment/broadcast operations.
    """

    structure: OrderedDict = field(default_factory=OrderedDict)

    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__

    @override
    @property
    def dim(self) -> int:
        # BroadcastSpace is modeled as a singleton axis for shape semantics.
        return 1

    def __repr__(self):
        return "BroadcastSpace"

    __str__ = __repr__


@dispatch(BroadcastSpace, BroadcastSpace)  # type: ignore[no-redef]
def same_span(a: BroadcastSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(StateSpace, BroadcastSpace)  # type: ignore[no-redef]
def same_span(a: StateSpace, b: BroadcastSpace) -> bool:
    return True


@dispatch(BroadcastSpace, StateSpace)  # type: ignore[no-redef]
def same_span(a: BroadcastSpace, b: StateSpace) -> bool:
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


class IndexSpace(StateSpace[int]):
    """
    A simple state space where the spatial elements are just integer indices.
    This can be useful for representing generic tensor dimensions that don't
    have a specific physical interpretation, such as the virtual bond dimension in a TNS.
    """

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
        return f"IndexSpace({self.dim})"

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
