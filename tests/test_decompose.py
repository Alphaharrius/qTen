import torch
import sympy as sy

from pyhilbert.decompose import eig, eigh, eigvals, qr, svd
from pyhilbert.state_space import IndexSpace
from pyhilbert.hilbert_space import U1Basis, hilbert
from pyhilbert.tensors import Tensor


def _space(name: str, n: int):
    return hilbert(U1Basis(coef=sy.Integer(1), base=((name, i),)) for i in range(n))


def test_eigh_reconstructs_hermitian_matrix():
    torch.manual_seed(0)

    space = _space("m", 3)

    data = torch.randn(3, 3, dtype=torch.complex64)
    hermitian = data + data.conj().transpose(-2, -1)
    tensor = Tensor(data=hermitian, dims=(space, space))

    eigvals, eigvecs = eigh(tensor)

    assert isinstance(eigvals.dims[-1], IndexSpace)
    assert eigvecs.dims[-2] == space
    assert eigvecs.dims[-1] is eigvals.dims[-1]

    diag = torch.diag(eigvals.data).to(eigvecs.data.dtype)
    recon = eigvecs.data @ diag @ eigvecs.data.conj().transpose(-2, -1)
    assert torch.allclose(recon, hermitian, atol=1e-5, rtol=1e-5)


def test_eig_reconstructs_general_matrix():
    torch.manual_seed(0)

    space = _space("m", 3)

    data = torch.randn(3, 3, dtype=torch.complex64)
    tensor = Tensor(data=data, dims=(space, space))

    eigvals, eigvecs = eig(tensor)

    assert isinstance(eigvals.dims[-1], IndexSpace)
    assert eigvecs.dims[-2] == space
    assert eigvecs.dims[-1] is eigvals.dims[-1]

    diag = torch.diag(eigvals.data)
    recon = eigvecs.data @ diag @ torch.linalg.inv(eigvecs.data)
    assert torch.allclose(recon, data, atol=1e-5, rtol=1e-5)


def test_eigvals_returns_index_space_without_band_grouping():
    space = _space("m", 4)

    values = torch.tensor(
        [1.0 + 1.0e-4j, 1.0 + 2.0e-4j, 2.0 + 0.0j, 2.0 + 1.0e-6j],
        dtype=torch.complex64,
    )
    data = torch.diag(values)
    tensor = Tensor(data=data, dims=(space, space))

    vals = eigvals(tensor)

    assert isinstance(vals.dims[-1], IndexSpace)
    assert vals.dims[-1].dim == 4
    assert torch.allclose(vals.data, values)


def test_qr_reconstructs_tall_matrix():
    torch.manual_seed(0)

    row_space = _space("row", 4)
    col_space = _space("col", 3)

    data = torch.randn(4, 3, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    q, r = qr(tensor)

    assert q.dims[-2] == row_space
    assert isinstance(q.dims[-1], IndexSpace)
    assert r.dims[-1] == col_space
    assert r.dims[-2] is q.dims[-1]

    recon = q.data @ r.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)


def test_qr_reconstructs_wide_matrix():
    torch.manual_seed(0)

    row_space = _space("row", 3)
    col_space = _space("col", 5)

    data = torch.randn(3, 5, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    q, r = qr(tensor)

    assert q.dims[-2] == row_space
    assert isinstance(q.dims[-1], IndexSpace)
    assert r.dims[-1] == col_space
    assert r.dims[-2] is q.dims[-1]

    recon = (q @ r).data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)


def test_svd_reconstructs_tall_matrix():
    torch.manual_seed(0)

    row_space = _space("row", 4)
    col_space = _space("col", 3)

    data = torch.randn(4, 3, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    u, s, vh = svd(tensor)

    assert u.dims[-2] == row_space
    assert isinstance(u.dims[-1], IndexSpace)
    assert s.dims[-1] is u.dims[-1]
    assert vh.dims[-1] == col_space
    assert vh.dims[-2] is u.dims[-1]

    diag = torch.diag_embed(s.data).to(u.data.dtype)
    recon = u.data @ diag @ vh.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)


def test_svd_matrix_values_no_band_grouping():
    row_space = _space("row", 3)
    col_space = _space("col", 3)

    singular_values = torch.tensor([3.0, 3.0, 1.0], dtype=torch.float64)
    data = torch.diag(singular_values)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    u, s, vh = svd(tensor, values_as_matrix=True)

    assert isinstance(s.dims[-1], IndexSpace)
    assert s.dims[-2] is s.dims[-1]
    assert s.dims[-1].dim == 3

    recon = u.data @ s.data.to(u.data.dtype) @ vh.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)


def test_svd_full_matrices_reconstructs():
    torch.manual_seed(0)

    row_space = _space("row", 4)
    col_space = _space("col", 3)

    data = torch.randn(4, 3, dtype=torch.float64)
    tensor = Tensor(data=data, dims=(row_space, col_space))

    u, s, vh = svd(tensor, full_matrices=True, values_as_matrix=True)

    assert u.dims[-2] == row_space
    assert isinstance(u.dims[-1], IndexSpace)
    assert isinstance(s.dims[-2], IndexSpace)
    assert isinstance(s.dims[-1], IndexSpace)
    assert isinstance(vh.dims[-2], IndexSpace)
    assert vh.dims[-1] == col_space

    recon = u.data @ s.data.to(u.data.dtype) @ vh.data
    assert torch.allclose(recon, data, atol=1e-6, rtol=1e-6)
