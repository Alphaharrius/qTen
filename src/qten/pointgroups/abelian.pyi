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
from functools import lru_cache
from typing import Any

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
class AffineTransform(Opr, HasBase[AffineSpace]):
    irrep: sy.ImmutableDenseMatrix
    axes: tuple[sy.Symbol, ...]
    offset: Offset
    basis_function_order: int
    @property
    @lru_cache
    def euclidean_basis(self) -> sy.ImmutableDenseMatrix: ...
    @property
    @lru_cache
    def full_rep(self) -> sy.MatrixBase: ...
    @property
    @lru_cache
    def rep(self) -> sy.ImmutableDenseMatrix: ...
    @property
    @lru_cache
    def affine_rep(self) -> sy.ImmutableDenseMatrix: ...
    @property
    @lru_cache
    def basis(self) -> FrozenDict[sy.Expr, AbelianBasis]: ...
    def base(self) -> AffineSpace: ...
    def rebase(self, new_base: AffineSpace) -> AffineTransform: ...
    def with_origin(self, origin: Offset) -> AffineTransform: ...
    @lru_cache
    def group_elements(self, max_order: int = 128) -> tuple["AffineTransform", ...]: ...
    def irreps(self) -> FrozenDict[sy.Expr, AbelianBasis]: ...
    def __matmul__(self, other) -> Any: ...
