from dataclasses import dataclass
from typing import Tuple

from ..symbolics import Multiple
from ..symbolics import U1Basis


@dataclass(frozen=True)
class Bond(Multiple[Tuple[U1Basis, U1Basis]]):
    pass
