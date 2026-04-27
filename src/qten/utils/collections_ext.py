"""
Collection helpers used across QTen.

This module provides small extensions around Python collection protocols. The
main public type, [`FrozenDict`][qten.utils.collections_ext.FrozenDict], is an
immutable, hashable mapping used where configuration dictionaries need stable
identity in caches or exported records.
"""

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
    """
    Immutable hashable dictionary-like mapping.

    [`FrozenDict`][qten.utils.collections_ext.FrozenDict] stores its items as an
    immutable frozenset-backed payload so instances can be hashed and used as
    dictionary keys or cache entries, provided all keys and values are
    hashable.

    Parameters
    ----------
    *args : Any
        Positional arguments accepted by the built-in `dict` constructor.
    **kwargs : Any
        Keyword arguments accepted by the built-in `dict` constructor.

    Raises
    ------
    TypeError
        If any key or value is not hashable.

    Examples
    --------
    ```python
    options = FrozenDict({"method": "exact", "maxiter": 100})
    cache = {options: "result"}
    ```
    """

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

    For each element in `source`, `base_func` is evaluated and matched against
    exactly one element in `dest` with the same baseline value. This is useful
    when two collections contain distinct objects that should be paired by a
    derived key rather than by object identity.

    Parameters
    ----------
    source : Iterable[Any]
        Source elements to map from.
    dest : Iterable[Any]
        Destination elements to map onto.
    base_func : Callable[[Any], Any]
        Function that extracts the comparison key from both source and
        destination elements.

    Returns
    -------
    Dict[Any, Any]
        Mapping from each source element to its corresponding destination
        element.

    Raises
    ------
    ValueError
        If two destination elements produce the same baseline value.
    ValueError
        If a source element has no destination element with a matching baseline
        value.

    Examples
    --------
    ```python
    source = ["x0", "y0"]
    dest = ["x1", "y1"]
    matchby(source, dest, lambda item: item[0])
    ```
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
