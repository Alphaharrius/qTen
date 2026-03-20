import sympy as sy
from ..abstracts import HasBase as HasBase
from ..geometries.spatials import (
    AffineSpace as AffineSpace,
    Momentum as Momentum,
    Offset as Offset,
    Spatial as Spatial,
)
from ..symbolics import Multiple as Multiple
from ..symbolics.hilbert_space import (
    HilbertSpace as HilbertSpace,
    Opr as Opr,
    U1Basis as U1Basis,
    U1Span as U1Span,
)
from ..utils.collections_ext import FrozenDict as FrozenDict
from ..validations import need_validation as need_validation
from ..validations.symbolics import (
    check_invertibility as check_invertibility,
    check_numerical as check_numerical,
)
from dataclasses import dataclass
from typing import TypeVar, overload

_T = TypeVar("_T")

@dataclass(frozen=True)
class AbelianBasis(Spatial):
    expr: sy.Expr
    axes: tuple[sy.Symbol, ...]
    order: int
    rep: sy.ImmutableDenseMatrix
    @property
    def dim(self) -> int: ...
    def __lt__(self, other: AbelianBasis) -> bool: ...
    def __gt__(self, other: AbelianBasis) -> bool: ...

@dataclass(frozen=True)
class AbelianGroup(Opr):
    irrep: sy.ImmutableDenseMatrix
    axes: tuple[sy.Symbol, ...]
    def euclidean_repr(self, order: int) -> sy.ImmutableDenseMatrix: ...
    def group_order(self, max_order: int = 128) -> int: ...
    def basis(self, order: int) -> FrozenDict[sy.Expr, AbelianBasis]: ...
    @property
    def basis_table(self) -> FrozenDict[sy.Expr, AbelianBasis]: ...

@dataclass(frozen=True)
class AbelianOpr(Opr, HasBase[AffineSpace]):
    g: AbelianGroup
    offset: Offset
    def __init__(
        self,
        g: AbelianGroup,
    ) -> None: ...
    def base(self) -> AffineSpace: ...
    def rebase(self, new_base: AffineSpace) -> AbelianOpr: ...
    def fixpoint_at(self, r: Offset, rebase: bool = False) -> AbelianOpr: ...
    @overload
    def invoke(self, obj: AbelianBasis, **kwargs) -> Multiple[AbelianBasis]: ...
    @overload
    def invoke(self, obj: Offset, **kwargs) -> Offset: ...
    @overload
    def invoke(self, obj: Momentum, **kwargs) -> Momentum: ...
    @overload
    def invoke(self, obj: U1Basis, **kwargs) -> U1Basis: ...
    @overload
    def invoke(self, obj: U1Span, **kwargs) -> U1Span: ...
    @overload
    def invoke(self, obj: HilbertSpace, **kwargs) -> HilbertSpace: ...
    @overload
    def invoke(self, obj: Multiple[_T], **kwargs) -> Multiple[_T]: ...
    @overload
    def invoke(self, obj: _T, **kwargs) -> _T | Multiple[_T]: ...  # type: ignore[override]
