from dataclasses import dataclass, field
from typing import FrozenSet, Iterable

from qten.geometries.spatials import Offset
from qten.plottings import Plottable


@dataclass(frozen=True)
class PointCloud(Plottable):
    """
    Plottable collection of spatial offsets with an optional display color.
    """

    offsets: FrozenSet[Offset] = field(default_factory=frozenset)
    color: str | None = None

    def __post_init__(self):
        from qten.geometries.spatials import Offset

        normalized = frozenset(self.offsets)
        if not all(isinstance(offset, Offset) for offset in normalized):
            raise TypeError("PointCloud offsets must all be Offset instances.")
        object.__setattr__(self, "offsets", normalized)

    @classmethod
    def of(cls, offsets: Iterable[Offset], color: str | None = None) -> "PointCloud":
        """Construct a `PointCloud` from any iterable of offsets."""
        return cls(offsets=frozenset(offsets), color=color)
