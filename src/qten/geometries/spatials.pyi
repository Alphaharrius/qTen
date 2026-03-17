import abc
import numpy as np
import torch
from ..abstracts import (
    Convertible as Convertible,
    HasBase as HasBase,
    HasDual as HasDual,
    Operable as Operable,
)
from ..plottings import Plottable as Plottable
from ..precision import get_precision_config as get_precision_config
from ..utils.collections_ext import FrozenDict as FrozenDict
from ..validations import need_validation as need_validation
from ..validations.symbolics import (
    check_invertibility as check_invertibility,
    check_numerical as check_numerical,
)
from .boundary import (
    BoundaryCondition as BoundaryCondition,
    PeriodicBoundary as PeriodicBoundary,
)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from sympy import ImmutableDenseMatrix
from typing import Any, Generic, Mapping, Sequence, TypeVar
from typing_extensions import override

_O = TypeVar("_O", bound="Offset[Any]")
_VecType = TypeVar("_VecType", bound=np.ndarray | ImmutableDenseMatrix)

@dataclass(frozen=True)
class Spatial(Operable, Plottable, ABC, metaclass=abc.ABCMeta):
    @property
    @abstractmethod
    def dim(self) -> int: ...

@dataclass(frozen=True)
class AffineSpace(Spatial):
    basis: ImmutableDenseMatrix
    @property
    def dim(self) -> int: ...

@dataclass(frozen=True)
class AbstractLattice(AffineSpace, HasDual, Generic[_O], metaclass=abc.ABCMeta):
    @property
    def affine(self) -> AffineSpace: ...
    @abstractmethod
    def cartes(self) -> tuple[_O, ...]: ...

@dataclass(frozen=True)
class Lattice(AbstractLattice["Offset"]):
    boundaries: BoundaryCondition
    @property
    @lru_cache
    def shape(self) -> tuple[int, ...]: ...
    def __init__(
        self,
        basis: ImmutableDenseMatrix,
        boundaries: BoundaryCondition | None = None,
        unit_cell: Mapping[str, ImmutableDenseMatrix] | None = None,
        shape: Sequence[int] | None = None,
    ) -> None: ...
    @property
    @lru_cache
    def unit_cell(self) -> FrozenDict: ...
    @property
    @lru_cache
    def dual(self) -> ReciprocalLattice: ...
    @lru_cache
    @override
    def cartes(self) -> tuple["Offset", ...]: ...
    def coords(self) -> torch.Tensor: ...

@dataclass(frozen=True)
class ReciprocalLattice(AbstractLattice["Momentum"]):
    lattice: Lattice
    @property
    @lru_cache
    def shape(self) -> tuple[int, ...]: ...
    @property
    @lru_cache
    def size(self) -> int: ...
    @property
    @lru_cache
    def dual(self) -> Lattice: ...
    @lru_cache
    @override
    def cartes(self) -> tuple["Momentum", ...]: ...

S = TypeVar("S", bound=AffineSpace)

@dataclass(frozen=True)
class Offset(Spatial, HasBase[S], Generic[S]):
    rep: ImmutableDenseMatrix
    space: S
    def __post_init__(self) -> None: ...
    @property
    def dim(self) -> int: ...
    def __add__(self, other: Offset[Any]) -> Offset[S]: ...
    def __neg__(self) -> Offset[S]: ...
    def __sub__(self, other: Offset[Any]) -> Offset[S]: ...
    def __lt__(self, other: Offset[Any]) -> bool: ...
    def __gt__(self, other: Offset[Any]) -> bool: ...
    def fractional(self) -> Offset[S]: ...
    def base(self) -> S: ...
    def rebase(self, space: S) -> Offset[S]: ...
    def to_vec(self, T: type[_VecType]) -> _VecType: ...

@dataclass(frozen=True)
class Momentum(Offset[ReciprocalLattice], Convertible):
    def __add__(self, other: Momentum) -> Momentum: ...  # type: ignore[override]
    def __neg__(self) -> Momentum: ...
    def __sub__(self, other: Momentum) -> Momentum: ...  # type: ignore[override]
    @override
    def fractional(self) -> Momentum: ...
    def base(self) -> ReciprocalLattice: ...
    def rebase(self, space: ReciprocalLattice) -> Momentum: ...
