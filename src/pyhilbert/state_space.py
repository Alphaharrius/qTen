from dataclasses import dataclass, replace, field
from typing import Callable, NamedTuple, Tuple, TypeVar, Generic, Union, Self, cast
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from functools import lru_cache
from itertools import chain, islice

from multipledispatch import dispatch  # type: ignore[import-untyped]

from .abstracts import Convertible
from .spatials import (
    Spatial,
    ReciprocalLattice,
    Momentum,
    cartes,
)


TSpatial = TypeVar("TSpatial", bound=Spatial)


@dataclass(frozen=True)
class StateSpace(Spatial, Convertible, Generic[TSpatial]):
    """
    `StateSpace` is a collection of indices with additional information attached to the elements,
    for the case of TNS there are only two types of state spaces: `MomentumSpace` and `HilbertSpace`.
    `MomentumSpace` is needed because some tensors are better represented in momentum space, e.g. Hamiltonians
    with translational symmetry, while `HilbertSpace` is needed to represent local degrees of freedom, e.g. spin or fermionic modes.

    Attributes
    ----------
    structure : OrderedDict[Spatial, slice]
        An ordered dictionary mapping each spatial component (e.g., `Offset`, `Momentum`) to a slice object that defines its
        position and the range in the tensor. The slices should be contiguous and ordered.

    dim : int
        The total dimension of the state space, calculated as the count of elements regardless of their lengths.
    """

    structure: OrderedDict[TSpatial, slice]
    """
    An ordered dictionary mapping each spatial component (e.g., `Offset`, `Momentum`) to a slice object that defines its 
    position and the range in the tensor. The slices should be contiguous and ordered.
    """

    @property
    def dim(self) -> int:
        """The total size of the vector space (sum of all sector dimensions)."""
        if not self.structure:
            return 0
        return self.structure[next(reversed(self.structure))].stop

    def elements(self) -> Tuple[TSpatial, ...]:
        """Return the spatial elements as a tuple."""
        return tuple(k for k in self.structure.keys())

    def get_slice(self, key: TSpatial) -> slice:
        """Get the slice associated with a given spatial key."""
        return self.structure[key]

    def __len__(self) -> int:
        """Return the number of spatial elements."""
        return len(self.structure)

    def __iter__(self) -> Iterator[TSpatial]:
        """Iterate over spatial elements."""
        return iter(k for k, _ in self.structure.items())

    def __hash__(self):
        # TODO: Do we need to consider the order of the structure?
        return hash(tuple((k, s.start, s.stop) for k, s in self.structure.items()))

    def __getitem__(
        self, v: Union[int, slice, range]
    ) -> Union[TSpatial, "StateSpace[TSpatial]"]:
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

    def map(self, func: Callable[[TSpatial], TSpatial]) -> "StateSpace[TSpatial]":
        """
        Map the spatial elements of this state space using a provided function.

        Parameters
        ----------
        `func` : `Callable[[TSpatial], TSpatial]`
            A function that takes a spatial element and returns a transformed spatial element.

        Returns
        -------
        `StateSpace[TSpatial]`
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
    structure: OrderedDict[TSpatial, slice],
) -> OrderedDict[TSpatial, slice]:
    """
    Return a new `OrderedDict` with contiguous, ordered slices preserving lengths.

    Parameters
    ----------
    structure : `OrderedDict[Spatial, slice]`
        The original structure with possibly non-contiguous slices.

    Returns
    -------
    `OrderedDict[Spatial, slice]`
        The restructured `OrderedDict` with contiguous, ordered slices.
    """
    new_structure: OrderedDict[TSpatial, slice] = OrderedDict()
    base = 0
    for k, s in structure.items():
        L = s.stop - s.start
        new_structure[k] = slice(base, base + L)
        base += L
    return new_structure


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
def flat_permutation_order(src: "StateSpace", dest: "StateSpace") -> Tuple[int, ...]:
    """
    Return the flattened index permutation that reorders `src` to match `dest`.

    This expands each sector slice in `src` into its element indices, then
    concatenates those groups according to `permutation_order(src, dest)`.
    The result can be used to permute a flat vector or tensor axis from `src`
    ordering into `dest` ordering.

    Parameters
    ----------
    src : `StateSpace`
        The source state space defining the original ordering.
    dest : `StateSpace`
        The destination state space defining the target ordering.

    Returns
    -------
    `Tuple[int, ...]`
        Flattened indices that map element positions in `src` to `dest`.
    """
    index_groups = [tuple(range(s.start, s.stop)) for s in src.structure.values()]
    ordered_groups = (index_groups[i] for i in permutation_order(src, dest))
    return tuple(chain.from_iterable(ordered_groups))


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
    indices = []
    sup_slices = sup.structure
    for key, _ in sub.structure.items():
        if key not in sup_slices:
            raise ValueError(f"Key {key} not found in superspace")
        sup_slice = sup_slices[key]
        indices.append(range(sup_slice.start, sup_slice.stop))
    return tuple(chain.from_iterable(indices))


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
    structure = OrderedDict({k: slice(0, 1)})
    return MomentumSpace(structure=structure)


# Register the conversion from `Momentum` to `MomentumSpace`.
Momentum.add_conversion(MomentumSpace)(
    cast(Callable[[Momentum], MomentumSpace], momentum_to_momentumspace)
)


@lru_cache
def brillouin_zone(lattice: ReciprocalLattice) -> MomentumSpace:
    elements = cartes(lattice)
    structure = OrderedDict((el, slice(n, n + 1)) for n, el in enumerate(elements))
    return MomentumSpace(structure=structure)


@dataclass(frozen=True)
class BroadcastSpace(StateSpace[Spatial]):
    structure: OrderedDict = field(default_factory=OrderedDict)

    # Ensure that __hash__ is inherited from StateSpace since the hash of StateSpace is specifically
    # designed to account for the structure attribute which is an un-hashable type OrderedDict.
    __hash__ = StateSpace.__hash__

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


@dataclass(frozen=True)
class FactorBand(Spatial, Convertible):
    """
    A spectral band in an eigenvalue spectrum.

    Attributes
    ----------
    idx : int
        Zero-based band index.
    count : int
        Number of eigenvalues in the band.
    """

    idx: int
    count: int

    @property
    def dim(self) -> int:
        """Return the band dimension (number of eigenvalues in the band)."""
        return self.count


@dataclass(frozen=True)
class FactorSpace(StateSpace[FactorBand]):
    """
    State space describing a spectrum partitioned into spectral bands.

    Each band corresponds to a contiguous block of eigenvalues, and the total
    dimension equals the sum of all band sizes.
    """

    __hash__ = StateSpace.__hash__

    def __str__(self):
        band_count_repr = ", ".join([str(band.dim) for band in self])
        return f"FactorSpace({band_count_repr})"

    @classmethod
    def from_band_counts(cls, band_counts: Iterable[int]) -> "FactorSpace":
        """
        Construct a `FactorSpace` from per-band eigenvalue counts.

        Parameters
        ----------
        band_counts : Iterable[int]
            Sizes of each band in order.
        """
        structure = OrderedDict()
        base = 0
        for idx, count in enumerate(band_counts):
            band = FactorBand(idx=idx, count=count)
            structure[band] = slice(base, base + count)
            base += count
        return cls(structure=structure)


@FactorBand.add_conversion(StateSpace)
def factorband_to_factorspace(band: FactorBand) -> StateSpace:
    """Convert a `FactorBand` to a `FactorSpace` containing only that band."""
    structure = OrderedDict({band: slice(0, band.dim)})
    return FactorSpace(structure=structure)


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
