import torch
import sympy as sy
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
import pytest

from pyhilbert.spatials import (
    Lattice,
    Offset,
    AffineSpace,
    ReciprocalLattice,
    Momentum,
)
from pyhilbert.state_space import brillouin_zone
from pyhilbert.hilbert_space import U1Basis, hilbert
from pyhilbert.tensors import Tensor
from pyhilbert.basis_transform import bandfold, BasisTransform
from pyhilbert.boundary import PeriodicBoundary


@dataclass(frozen=True)
class Orb:
    name: str


def _mode(r: Offset, orb: str = "s") -> U1Basis:
    return U1Basis(coef=sy.Integer(1), base=(r, Orb(orb)))


def test_bandfold_1d():
    # 1. Setup
    # 1a. Define a 1D lattice with 4 k-points
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    assert k_space.dim == 4

    # 1b. Define a simple 1-dim Hilbert space
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    h_space = hilbert([_mode(r_offset)])
    assert h_space.dim == 1

    # 1c. Create an input tensor (4, 1, 1)
    # Data is just a sequence of numbers for easy tracking
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double the unit cell)
    M = ImmutableDenseMatrix([[2]])

    # 2. Execute
    tensor_out = bandfold(M, tensor_in)

    # 3. Assert
    # 3a. Check new dimensions
    scaled_k_space = tensor_out.dims[0]
    new_h_space = tensor_out.dims[1]

    assert scaled_k_space.dim == 2  # 4 / det(M) = 4 / 2 = 2
    assert new_h_space.dim == 2  # 1 * det(M) = 1 * 2 = 2
    assert tensor_out.dims[2].dim == 2

    # 3b. Check the data
    # k=0 folds to k=0. k=1/2 folds to k=0.
    # k=1/4 folds to k=1/4. k=3/4 folds to k=1/4.
    # Original k-points: 0, 1/4, 1/2, 3/4
    # New k-points: 0, 1/2.

    # Check data for k_new=0 (index 0)
    # Maps k=0 (val 0) and k=1/2 (val 2)
    # Expected matrix: [[1, -1], [-1, 1]]
    expected_k0 = torch.tensor([[1, -1], [-1, 1]], dtype=torch.complex128)
    assert torch.allclose(tensor_out.data[0], expected_k0)

    # Check data for k_new=1/2 (index 1)
    # Maps k=1/4 (val 1) and k=3/4 (val 3)
    # Expected matrix: [[2, i], [-i, 2]]
    expected_k1 = torch.tensor([[2, 1j], [-1j, 2]], dtype=torch.complex128)
    assert torch.allclose(tensor_out.data[1], expected_k1)


def test_bandfold_2d():
    # 1. Setup
    # 1a. Define a 2D lattice with 4 k-points (2x2)
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)
    assert k_space.dim == 4

    # 1b. Define a simple Hilbert space
    r_offset = Offset(rep=ImmutableDenseMatrix([0, 0]), space=lattice)
    h_space = hilbert([_mode(r_offset, "s")])
    assert h_space.dim == 1

    # 1c. Create input tensor (4, 1, 1)
    # Data: 0, 1, 2, 3
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double in both directions)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])

    # 2. Execute
    tensor_out = bandfold(M, tensor_in)

    # 3. Assert
    # 3a. Check dimensions
    # New lattice shape: (2//2, 2//2) = (1, 1) -> 1 k-point
    # New Hilbert dim: 1 * det(M) = 4
    scaled_k_space = tensor_out.dims[0]
    new_h_space = tensor_out.dims[1]

    assert scaled_k_space.dim == 1
    assert new_h_space.dim == 4
    assert tensor_out.dims[2].dim == 4

    # 3b. Check data
    # All 4 k-points fold to the single Gamma point.
    # Expected matrix derived from folding 0, 1, 2, 3
    # Basis order: (0,0), (0,1), (1,0), (1,1)
    expected_matrix = torch.tensor(
        [
            [1.5, -0.5, -1.0, 0.0],
            [-0.5, 1.5, 0.0, -1.0],
            [-1.0, 0.0, 1.5, -0.5],
            [0.0, -1.0, -0.5, 1.5],
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(tensor_out.data[0].real, expected_matrix)
    assert torch.allclose(tensor_out.data[0].imag, torch.zeros_like(expected_matrix))


def test_affine_space_transform():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])
    t = BasisTransform(M)

    new_space = t(space)
    assert isinstance(new_space, AffineSpace)
    assert new_space.basis == M @ basis


def test_basis_transform_rejects_non_invertible_matrix():
    with pytest.raises(ValueError, match="positive determinant"):
        BasisTransform(ImmutableDenseMatrix([[1, 0], [0, 0]]))


def test_lattice_transform():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    M = ImmutableDenseMatrix([[2]])
    t = BasisTransform(M)

    new_lat = t(lat)
    assert isinstance(new_lat, Lattice)
    # Basis scaled by 2
    assert new_lat.basis == ImmutableDenseMatrix([[2]])
    # Shape halved
    assert new_lat.shape == (2,)
    # Unit cell populated (det(M)=2 atoms)
    assert len(new_lat.unit_cell) == 2
    # Check keys and positions
    # Default key is "r", so we expect "r_0", "r_1"
    assert "r_0" in new_lat.unit_cell
    assert "r_1" in new_lat.unit_cell
    # Positions: 0 and 0.5
    assert new_lat.unit_cell["r_0"].rep == ImmutableDenseMatrix([0])
    assert new_lat.unit_cell["r_1"].rep == ImmutableDenseMatrix([sy.Rational(1, 2)])


def test_reciprocal_lattice_transform():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual  # Basis is [2pi]

    M = ImmutableDenseMatrix([[2]])
    t = BasisTransform(M)

    new_recip = t(recip)
    assert isinstance(new_recip, ReciprocalLattice)
    # New recip basis should be old_recip_basis * M^-T = [2pi] * [1/2] = [pi]
    assert new_recip.basis == ImmutableDenseMatrix([[sy.pi]])


def test_offset_transform():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    offset = Offset(rep=ImmutableDenseMatrix([1]), space=lattice)

    M = ImmutableDenseMatrix([[2]])
    t = BasisTransform(M)

    new_offset = t(offset)
    # Physical position is 1. New basis is 2. New rep should be 0.5.
    assert new_offset.rep == ImmutableDenseMatrix([sy.Rational(1, 2)])
    assert new_offset.space.basis == ImmutableDenseMatrix([[2]])


def test_momentum_transform():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual  # Basis [2pi]
    # Momentum at 0.5 (fractional) -> physical pi
    k = Momentum(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=recip)

    M = ImmutableDenseMatrix([[2]])
    t = BasisTransform(M)

    new_k = t(k)
    # New recip basis is [pi]. Physical momentum pi. New rep should be 1.
    # Formula: new_rep = M^T @ old_rep = [2] @ [0.5] = [1]
    assert new_k.rep == ImmutableDenseMatrix([1])
    assert new_k.space.basis == ImmutableDenseMatrix([[sy.pi]])
