import pytest
import sympy as sy
import torch

from qten.linalg.orthogonalize import lowdin_orthonormalize
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace


def _space(name: str, n: int):
    return HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=((name, i),)) for i in range(n)
    )


def test_lowdin_orthonormalize_preserves_dims_and_orthonormalizes_columns():
    torch.manual_seed(0)

    row_space = _space("row", 4)
    col_space = _space("col", 2)
    tensor = Tensor(
        data=torch.randn(4, 2, dtype=torch.float64),
        dims=(row_space, col_space),
    )

    result = lowdin_orthonormalize(tensor)

    assert result.dims == tensor.dims
    assert torch.allclose(
        result.data.mH @ result.data,
        torch.eye(2, dtype=torch.float64),
        atol=1e-12,
        rtol=1e-12,
    )


def test_lowdin_orthonormalize_handles_complex_batches():
    torch.manual_seed(0)

    batch_space = IndexSpace.linear(3)
    row_space = _space("row", 4)
    col_space = _space("col", 2)
    tensor = Tensor(
        data=torch.randn(3, 4, 2, dtype=torch.complex128),
        dims=(batch_space, row_space, col_space),
    )

    result = lowdin_orthonormalize(tensor)
    identity = torch.eye(2, dtype=torch.complex128).expand(3, -1, -1)

    assert torch.allclose(result.data.mH @ result.data, identity, atol=1e-12)


def test_lowdin_orthonormalize_rejects_linearly_dependent_columns():
    space = _space("row", 3)
    columns = _space("col", 2)
    data = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="linearly dependent columns"):
        lowdin_orthonormalize(Tensor(data=data, dims=(space, columns)))
