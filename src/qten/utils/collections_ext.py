from collections.abc import Mapping
from typing import (
    Iterator,
    Iterable,
    Any,
    Generic,
    Tuple,
    TypeVar,
    cast,
    Callable,
    Dict,
)


_K = TypeVar("_K")
_V = TypeVar("_V")


class FrozenDict(Mapping[_K, _V], Generic[_K, _V]):
    __slots__ = ("__items", "__hash")

    def __init__(self, *args: Any, **kwargs: Any):
        data = dict(*args, **kwargs)
        try:
            fitems = frozenset(data.items())  # ensures all keys/vals are hashable
        except TypeError as e:
            raise TypeError(
                "All keys and values must be hashable. "
                "Use deep_freeze() for nested mutables."
            ) from e
        object.__setattr__(
            self, "_FrozenDict__items", tuple(fitems)
        )  # hidden, immutable
        object.__setattr__(self, "_FrozenDict__hash", hash(fitems))

    # internal accessor that bypasses the guard
    def _items(self) -> Tuple[Tuple[_K, _V], ...]:
        return cast(
            Tuple[Tuple[_K, _V], ...],
            object.__getattribute__(self, "_FrozenDict__items"),
        )

    # --- Mapping interface ---
    def __len__(self) -> int:
        return len(self._items())

    def __iter__(self) -> Iterator[_K]:
        for k, _ in self._items():
            yield k

    def __getitem__(self, key: _K) -> _V:
        for k, v in self._items():
            if k == key:
                return v
        raise KeyError(key)

    # --- equality & hash ---
    def __hash__(self) -> int:
        return object.__getattribute__(self, "_FrozenDict__hash")

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if isinstance(other, Mapping):
            try:
                return frozenset(self._items()) == frozenset(other.items())
            except TypeError:
                return False
        return NotImplemented

    def __str__(self) -> str:
        return str(dict(self._items()))

    def __repr__(self) -> str:
        return repr(dict(self._items()))

    def __getattribute__(self, name: str):
        if name in {"_FrozenDict__items"}:
            raise AttributeError("Private storage is hidden")
        return super().__getattribute__(name)


def matchby(
    source: Iterable[Any], dest: Iterable[Any], base_func: Callable[[Any], Any]
) -> Dict[Any, Any]:
    """
    Map elements from source to destination using a provided mapping function.
    Parameters
    ----------
    `source` : `Iterable[Any]`
        The source elements to be mapped.
    `dest` : `Iterable[Any]`
        The destination elements to map to.
    `base_func` : `Callable[[Any], Any]`
        A function that defines the comparison baseline.

    Returns
    -------
    `Dict[Any, Any]`
        A dictionary mapping each source element to its corresponding destination element `source -> dest`.
    """
    mapping: Dict[Any, Any] = {}

    source_base: Dict[Any, Any] = {m: base_func(m) for m in source}
    dest_base: Dict[Any, Any] = {base_func(m): m for m in dest}

    if len(dest_base) != len(tuple(dest)):
        raise ValueError("Destination elements have non-unique base values!")

    for sm, sb in source_base.items():
        if sb not in dest_base:
            raise ValueError(
                f"Source element {sm} with base {sb} has no match in destination!"
            )
        mapping[sm] = dest_base[sb]

    return mapping
