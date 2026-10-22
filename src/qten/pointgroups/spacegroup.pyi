from dataclasses import dataclass
from typing import Sequence
from .elements import PointGroupOpr as PointGroupOpr
from ..geometries.spatials import AffineSpace as AffineSpace, Momentum as Momentum

@dataclass(frozen=True)
class SpaceGroup:
    elements: tuple[PointGroupOpr, ...]
    @property
    def order(self) -> int: ...
    @classmethod
    def from_generators(
        cls,
        generators: Sequence[PointGroupOpr],
        *,
        max_order: int = 512,
        modulo: AffineSpace | None = None,
    ) -> SpaceGroup: ...
    def little_group(self, k: Momentum) -> tuple[PointGroupOpr, ...]: ...
