import numpy as np
import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import brillouin_zone
from qten.topology import (
    berry_curvature,
    chern_number,
    fubini_study_metric,
    quantum_geometric_tensor,
)


def _chern_insulator(boundary_basis: ImmutableDenseMatrix) -> Tensor:
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(boundary_basis),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(("band", i),)) for i in range(2)
    )
    blocks = []
    for k in k_space.elements():
        kx = 2.0 * torch.pi * float(k.rep[0])
        ky = 2.0 * torch.pi * float(k.rep[1])
        dx = torch.sin(torch.tensor(kx, dtype=torch.float64))
        dy = torch.sin(torch.tensor(ky, dtype=torch.float64))
        dz = -1.0 + torch.cos(torch.tensor(kx)) + torch.cos(torch.tensor(ky))
        blocks.append(
            torch.stack(
                (
                    torch.stack((dz, dx - 1j * dy)),
                    torch.stack((dx + 1j * dy, -dz)),
                )
            ).to(torch.complex128)
        )
    return Tensor(data=torch.stack(blocks), dims=(k_space, band_space, band_space))


def test_quantum_geometry_components_are_consistent():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(16, 16))

    qgt = quantum_geometric_tensor(hamiltonian, n_occupied=1)
    metric = fubini_study_metric(hamiltonian, n_occupied=1)
    curvature = berry_curvature(hamiltonian, n_occupied=1)

    assert qgt.shape == (16, 16, 2, 2)
    assert np.allclose(metric, qgt.real)
    assert np.allclose(curvature, 2.0 * qgt[..., 0, 1].imag)
    assert np.allclose(metric[..., 0, 1], metric[..., 1, 0])
    assert np.all(np.linalg.eigvalsh(metric) > -1e-12)


def test_chern_number_exposes_robust_and_geometric_methods():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(24, 24))

    fhs = chern_number(hamiltonian, n_occupied=1)
    geometric = chern_number(hamiltonian, n_occupied=1, method="qgt")

    assert abs(fhs["nearest_integer"]) == 1
    assert geometric["chern"] == pytest.approx(fhs["chern"], abs=0.1)
    assert geometric["fubini_study_metric"].shape == (24, 24, 2, 2)
    assert geometric["berry_curvature"].shape == (24, 24)


def test_sheared_cell_quantum_geometry_stays_flat():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix([[8, 2], [0, 8]]))

    assert quantum_geometric_tensor(hamiltonian, 1).shape == (64, 2, 2)
    assert berry_curvature(hamiltonian, 1).shape == (64,)
    assert chern_number(hamiltonian, 1)["berry_flux"].shape == (64,)
