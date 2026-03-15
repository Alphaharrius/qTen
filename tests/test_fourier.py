import torch
import numpy as np
import sympy as sy
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
from qten.geometries.spatials import Lattice, Offset, Momentum
from qten.symbolics.state_space import brillouin_zone
from qten.symbolics.hilbert_space import U1Basis, HilbertSpace
from qten.geometries.fourier import fourier_transform
from qten.geometries.boundary import PeriodicBoundary


@dataclass(frozen=True)
class Orb:
    name: str


def _mode(r: Offset, orb: str = "s") -> U1Basis:
    return U1Basis(coef=sy.Integer(1), base=(r, Orb(orb)))


def test_fourier_kernel_1d():
    # 1D Lattice a=1
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual

    # Define K points: 0, 0.25, 0.5, 0.75 (fractional in reciprocal basis)
    k_reps = [0, 0.25, 0.5, 0.75]
    K = tuple(Momentum(rep=ImmutableDenseMatrix([k]), space=recip) for k in k_reps)

    # Define R offsets: 0, 1, 2, 3 (fractional/integer in lattice basis)
    r_reps = [0, 1, 2, 3]
    R = tuple(Offset(rep=ImmutableDenseMatrix([r]), space=lat) for r in r_reps)

    # Compute Fourier
    ft = fourier_transform(K, R)

    # Expected: exp(-2pi * i * k * r)
    expected = torch.zeros((4, 4), dtype=torch.complex128)
    for i, k in enumerate(k_reps):
        for j, r in enumerate(r_reps):
            phase = -2j * np.pi * k * r
            expected[i, j] = np.exp(phase)

    assert torch.allclose(ft, expected)

    # Check unitarity
    # The FT matrix F should satisfy F.H @ F = N * I
    N = len(R)
    identity = torch.eye(N, dtype=torch.complex128)
    assert torch.allclose(ft.conj().T @ ft, N * identity)
    assert torch.allclose(ft @ ft.conj().T, N * identity)


def test_fourier_tensor_construction():
    # 1D Lattice
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )  # 2 unit cells
    recip = lat.dual

    # k-space: 2 points (0, 0.5)
    k_space = brillouin_zone(recip)
    assert k_space.dim == 2

    # Region space: 2 sites, one at 0, one at 1.
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lat)

    m0 = _mode(r0, "s")
    m1 = _mode(r1, "s")
    region_space = HilbertSpace.new([m0, m1])

    # Bloch space: 1 site at 0 (unit cell)
    b0 = _mode(r0, "s")
    bloch_space = HilbertSpace.new([b0])

    # Compute FT Tensor
    ft_tensor = fourier_transform(k_space, bloch_space, region_space)

    assert ft_tensor.dims == (k_space, bloch_space, region_space)
    data = ft_tensor.data
    assert data.shape == (2, 1, 2)

    # Check values
    # k=0 (rep 0): exp(0)=1 for both r=0 and r=1
    assert torch.allclose(
        data[0, 0, :], torch.tensor([1.0, 1.0], dtype=torch.complex128)
    )

    # k=0.5 (rep 0.5): r=0 -> 1, r=1 -> -1
    assert torch.allclose(
        data[1, 0, :], torch.tensor([1.0, -1.0], dtype=torch.complex128)
    )


def test_fourier_tensor_unitarity():
    # 1D Lattice
    basis = ImmutableDenseMatrix([[1]])
    n_cells = 4
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(n_cells)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    recip = lat.dual

    # k-space: 4 points
    k_space = brillouin_zone(recip)
    assert k_space.dim == n_cells

    # Region space: 4 sites, one at 0, 1, 2, 3.
    # This is a system with 4 unit cells, and one site per cell.
    # The region is the whole crystal.
    region_modes = []
    for i in range(n_cells):
        r = Offset(rep=ImmutableDenseMatrix([i]), space=lat.affine)
        m = _mode(r, "s")
        region_modes.append(m)
    region_space = HilbertSpace.new(region_modes)
    assert region_space.dim == n_cells

    # Bloch space: 1 site at 0 (defines the unit cell for Bloch states)
    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lat.affine)
    b0 = _mode(r0, "s")
    bloch_space = HilbertSpace.new([b0])
    assert bloch_space.dim == 1

    # Compute FT Tensor
    ft_tensor = fourier_transform(k_space, bloch_space, region_space)

    # The FT tensor reshaped into a matrix should be unitary up to a factor.
    K, B, R = ft_tensor.data.shape
    U_matrix = ft_tensor.data.reshape(K * B, R)

    # The transform from real space to k-space is unnormalized, so U.H @ U = N_cells * I.
    identity = torch.eye(R, dtype=torch.complex128)
    assert torch.allclose(U_matrix.conj().T @ U_matrix, n_cells * identity)
    assert torch.allclose(U_matrix @ U_matrix.conj().T, n_cells * identity)
