from collections import OrderedDict

import pytest
from sympy import ImmutableDenseMatrix

from pyhilbert.hilbert_space import U1Basis, HilbertSpace
from pyhilbert.spatials import AffineSpace, Momentum
from pyhilbert.state_space import IndexSpace, MomentumSpace


def test_momentum_space_rejects_non_contiguous_structure_values():
    space = AffineSpace(basis=ImmutableDenseMatrix([[1]]))
    k0 = Momentum(rep=ImmutableDenseMatrix([0]), space=space)
    k1 = Momentum(rep=ImmutableDenseMatrix([1]), space=space)

    with pytest.raises(ValueError, match="contiguous indices 0..n-1"):
        MomentumSpace(structure=OrderedDict(((k0, 0), (k1, 2))))


def test_index_space_rejects_non_contiguous_structure_values():
    with pytest.raises(ValueError, match="contiguous indices 0..n-1"):
        IndexSpace(structure=OrderedDict(((3, 0), (5, 3))))


def test_hilbert_space_rejects_non_contiguous_structure_values():
    space = AffineSpace(basis=ImmutableDenseMatrix([[1]]))
    r0 = Momentum(rep=ImmutableDenseMatrix([0]), space=space)
    r1 = Momentum(rep=ImmutableDenseMatrix([1]), space=space)
    psi0 = U1Basis(coef=1, base=(r0,))
    psi1 = U1Basis(coef=1, base=(r1,))

    with pytest.raises(ValueError, match="contiguous indices 0..n-1"):
        HilbertSpace(structure=OrderedDict(((psi0, 0), (psi1, 2))))
