from collections import OrderedDict

import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import MomentumSpace, brillouin_zone
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


def _dimensional_hamiltonian(sizes: tuple[int, ...]) -> Tensor:
    dimension = len(sizes)
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(dimension),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(*sizes)),
        unit_cell={"r": ImmutableDenseMatrix.zeros(dimension, 1)},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(("band", i),)) for i in range(2)
    )
    blocks = []
    for momentum in k_space.elements():
        angles = [2.0 * torch.pi * float(value) for value in momentum.rep]
        dx = sum(torch.sin(torch.tensor(angle)) for angle in angles)
        dz = 2.0 + sum(torch.cos(torch.tensor(angle)) for angle in angles)
        blocks.append(torch.stack((torch.stack((dz, dx)), torch.stack((dx, -dz)))))
    return Tensor(
        data=torch.stack(blocks).to(torch.complex128),
        dims=(k_space, band_space, band_space),
    )


def test_quantum_geometry_components_are_consistent():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(16, 16))

    qgt = quantum_geometric_tensor(hamiltonian, n_occupied=1)
    metric = fubini_study_metric(hamiltonian, n_occupied=1)
    curvature = berry_curvature(hamiltonian, n_occupied=1)

    assert isinstance(qgt, Tensor)
    assert qgt.dims[0] is hamiltonian.dims[0]
    assert qgt.data.shape == (16 * 16, 2, 2)
    assert torch.allclose(metric.data, qgt.data.real)
    assert torch.allclose(curvature.data, 2.0 * qgt.data.imag)
    assert torch.allclose(metric.data[..., 0, 1], metric.data[..., 1, 0])
    assert torch.all(torch.linalg.eigvalsh(metric.data) > -1e-12)


def test_chern_number_exposes_robust_and_geometric_methods():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(24, 24))

    fhs = chern_number(hamiltonian, n_occupied=1)
    geometric = chern_number(hamiltonian, n_occupied=1, method="qgt")

    assert fhs["chern"] == pytest.approx(-1.0, abs=1e-12)
    assert fhs["nearest_integer"] == -1
    assert geometric["chern"] == pytest.approx(fhs["chern"], abs=0.1)
    assert geometric["fubini_study_metric"].data.shape == (24 * 24, 2, 2)
    assert geometric["berry_curvature"].data.shape == (24 * 24, 2, 2)


def test_sheared_cell_quantum_geometry_stays_flat():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix([[8, 2], [0, 8]]))

    qgt = quantum_geometric_tensor(hamiltonian, 1)
    assert qgt.data.shape == (64, 2, 2)
    assert qgt.dims[0] is hamiltonian.dims[0]
    assert berry_curvature(hamiltonian, 1).data.shape == (64, 2, 2)
    assert chern_number(hamiltonian, 1)["berry_flux"].shape == (64,)


def test_quantum_geometry_follows_momentum_space_order():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(6, 4))
    reference = quantum_geometric_tensor(hamiltonian, 1)

    reversed_momenta = tuple(reversed(hamiltonian.dims[0].elements()))
    reordered_space = MomentumSpace(
        OrderedDict((momentum, i) for i, momentum in enumerate(reversed_momenta))
    )
    source_index = hamiltonian.dims[0].structure
    reordered_hamiltonian = Tensor(
        data=torch.stack(
            [hamiltonian.data[source_index[momentum]] for momentum in reversed_momenta]
        ),
        dims=(reordered_space, hamiltonian.dims[1], hamiltonian.dims[2]),
    )

    reordered = quantum_geometric_tensor(reordered_hamiltonian, 1)

    assert reordered.dims[0] is reordered_space
    for momentum in reversed_momenta:
        assert torch.allclose(
            reordered.data[reordered_space.structure[momentum]],
            reference.data[hamiltonian.dims[0].structure[momentum]],
        )


@pytest.mark.parametrize("sizes", [(8,), (4, 4, 4)])
def test_quantum_geometry_supports_one_and_three_dimensions(sizes):
    hamiltonian = _dimensional_hamiltonian(sizes)
    dimension = len(sizes)

    qgt = quantum_geometric_tensor(hamiltonian, 1)
    metric = fubini_study_metric(hamiltonian, 1)
    curvature = berry_curvature(hamiltonian, 1)

    expected_shape = (hamiltonian.dims[0].dim, dimension, dimension)
    assert qgt.data.shape == expected_shape
    assert metric.data.shape == expected_shape
    assert curvature.data.shape == expected_shape
    assert qgt.dims[0] is hamiltonian.dims[0]
    assert torch.allclose(curvature.data, -curvature.data.transpose(-2, -1))

    with pytest.raises(ValueError, match="two-dimensional"):
        chern_number(hamiltonian, 1)
