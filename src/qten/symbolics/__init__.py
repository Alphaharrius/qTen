"""
Symbolic basis, operator, and state-space layer for QTen.

This package hosts the symbolic objects that give QTen tensors their semantic
axis metadata: basis states, Hilbert spaces, abstract operators, state spaces,
and helper constructors acting on those structures.

Core symbolic values
--------------------
[`Multiple`][qten.symbolics.Multiple] is the scalar-times-object wrapper used
for factored symbolic coefficients. [`U1Basis`][qten.symbolics.hilbert_space.U1Basis]
is the atomic symbolic basis state, [`U1Span`][qten.symbolics.hilbert_space.U1Span]
is an ordered span of basis states, and [`HilbertSpace`][qten.symbolics.hilbert_space.HilbertSpace]
stores symbolic basis sectors as a state space.

Operators
---------
[`Opr`][qten.symbolics.hilbert_space.Opr] is the base symbolic operator class.
[`FuncOpr`][qten.symbolics.hilbert_space.FuncOpr] lifts a Python callable on one
irrep type. Convenience constructors include
[`translate_opr`][qten.symbolics.ops.translate_opr],
[`rebase_opr`][qten.symbolics.ops.rebase_opr], and
[`fractional_opr`][qten.symbolics.ops.fractional_opr].

State-space utilities
---------------------
State-space types include [`StateSpace`][qten.symbolics.state_space.StateSpace],
[`MomentumSpace`][qten.symbolics.state_space.MomentumSpace],
[`BroadcastSpace`][qten.symbolics.state_space.BroadcastSpace],
[`IndexSpace`][qten.symbolics.state_space.IndexSpace],
[`StateSpaceFactorization`][qten.symbolics.state_space.StateSpaceFactorization],
and [`BzPath`][qten.symbolics.state_space.BzPath]. Helper functions include
[`brillouin_zone`][qten.symbolics.state_space.brillouin_zone],
[`embedding_order`][qten.symbolics.state_space.embedding_order],
[`permutation_order`][qten.symbolics.state_space.permutation_order],
[`restructure`][qten.symbolics.state_space.restructure], and
[`same_rays`][qten.symbolics.state_space.same_rays].

Hilbert-space helpers
---------------------
Hilbert-space helper functions include
[`region_hilbert`][qten.symbolics.ops.region_hilbert],
[`hilbert_opr_repr`][qten.symbolics.ops.hilbert_opr_repr],
[`match_indices`][qten.symbolics.ops.match_indices], and
[`interpolate_reciprocal_path`][qten.symbolics.ops.interpolate_reciprocal_path].
"""

from ._base import Multiple as Multiple

from .state_space import (
    BzPath as BzPath,
    BroadcastSpace as BroadcastSpace,
    IndexSpace as IndexSpace,
    MomentumSpace as MomentumSpace,
    StateSpace as StateSpace,
    StateSpaceFactorization as StateSpaceFactorization,
    brillouin_zone as brillouin_zone,
    embedding_order as embedding_order,
    permutation_order as permutation_order,
    restructure as restructure,
    same_rays as same_rays,
)

from .hilbert_space import (
    FuncOpr as FuncOpr,
    HilbertSpace as HilbertSpace,
    Opr as Opr,
    U1Basis as U1Basis,
    U1Span as U1Span,
)

from .ops import (
    fractional_opr as fractional_opr,
    hilbert_opr_repr as hilbert_opr_repr,
    interpolate_reciprocal_path as interpolate_reciprocal_path,
    match_indices as match_indices,
    rebase_opr as rebase_opr,
    region_hilbert as region_hilbert,
    translate_opr as translate_opr,
)
