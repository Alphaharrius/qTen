from collections import OrderedDict
import math

import pytest
import sympy as sy
import torch
from sympy import ImmutableDenseMatrix

from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.spatials import Lattice, Offset
from qten.linalg.tensors import Tensor
from qten.phys import Spin
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import (
    IndexSpace,
    MomentumBlockSpace,
    MomentumSpace,
    brillouin_zone,
)
from qten.topology import (
    berry_curvature,
    chern_number,
    fubini_study_metric,
    quantum_geometric_tensor,
    z2_indices,
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


def _trivial_insulator(boundary_basis: ImmutableDenseMatrix) -> Tensor:
    hamiltonian = _chern_insulator(boundary_basis)
    block = torch.diag(torch.tensor([-1.0, 1.0], dtype=torch.complex128))
    return Tensor(
        data=block.expand(hamiltonian.data.shape[0], -1, -1).clone(),
        dims=hamiltonian.dims,
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


def test_trivial_insulator_has_zero_chern_number():
    hamiltonian = _trivial_insulator(ImmutableDenseMatrix.diag(8, 8))

    assert chern_number(hamiltonian, 1)["chern"] == pytest.approx(0.0, abs=1e-12)
    assert chern_number(hamiltonian, 1, method="qgt")["chern"] == pytest.approx(
        0.0, abs=1e-12
    )


@pytest.mark.parametrize("n_occupied", [0, 2])
def test_chern_number_rejects_invalid_occupied_count(n_occupied):
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(4, 4))

    with pytest.raises(ValueError, match="strictly between"):
        chern_number(hamiltonian, n_occupied)


def test_chern_number_rejects_invalid_method():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(4, 4))

    with pytest.raises(ValueError, match="method must be"):
        chern_number(hamiltonian, 1, method="invalid")  # type: ignore[call-overload]


def test_qgt_chern_diagonalizes_once(monkeypatch):
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(8, 8))
    original_eigh = torch.linalg.eigh
    calls = 0

    def counting_eigh(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_eigh(*args, **kwargs)

    monkeypatch.setattr(torch.linalg, "eigh", counting_eigh)

    chern_number(hamiltonian, n_occupied=1, method="qgt")

    assert calls == 1


@pytest.mark.parametrize("method", ["fhs", "qgt"])
def test_chern_gap_warning_is_emitted_once_at_public_call(method):
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(4, 4))

    with pytest.warns(RuntimeWarning) as recorded:
        chern_number(
            hamiltonian,
            n_occupied=1,
            gap_tolerance=float("inf"),
            method=method,
        )

    assert len(recorded) == 1
    assert recorded[0].filename == __file__


def test_sheared_cell_quantum_geometry_stays_flat():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix([[8, 2], [0, 8]]))

    qgt = quantum_geometric_tensor(hamiltonian, 1)
    assert qgt.data.shape == (64, 2, 2)
    assert qgt.dims[0] is hamiltonian.dims[0]
    assert berry_curvature(hamiltonian, 1).data.shape == (64, 2, 2)
    fhs = chern_number(hamiltonian, 1)
    assert fhs["berry_flux"].data.shape == (64,)
    assert fhs["berry_flux"].dims == (hamiltonian.dims[0],)
    assert fhs["chern"] == pytest.approx(-1.0, abs=1e-12)


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


def test_fhs_flux_follows_momentum_space_order():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(6, 4))
    reference = chern_number(hamiltonian, 1)["berry_flux"]

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

    reordered = chern_number(reordered_hamiltonian, 1)["berry_flux"]

    for momentum in reversed_momenta:
        assert reordered.data[
            reordered_space.structure[momentum]
        ].item() == pytest.approx(
            reference.data[hamiltonian.dims[0].structure[momentum]].item()
        )


def test_quantum_geometry_aligns_reordered_band_axes():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(6, 4))
    reference = quantum_geometric_tensor(hamiltonian, 1)
    reordered_columns = hamiltonian.dims[2][[1, 0]]
    reordered_hamiltonian = Tensor(
        data=hamiltonian.data[..., [1, 0]],
        dims=(hamiltonian.dims[0], hamiltonian.dims[1], reordered_columns),
    )

    reordered = quantum_geometric_tensor(reordered_hamiltonian, 1)

    assert torch.allclose(reordered.data, reference.data)


@pytest.mark.parametrize("axis", [1, 2])
def test_quantum_geometry_rejects_non_hilbert_band_axes(axis):
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(4, 4))
    dims = list(hamiltonian.dims)
    dims[axis] = IndexSpace.linear(2)
    invalid = Tensor(data=hamiltonian.data, dims=tuple(dims))

    with pytest.raises(TypeError, match=f"The {'second' if axis == 1 else 'third'}"):
        quantum_geometric_tensor(invalid, 1)


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


def _pauli():
    identity = torch.eye(2, dtype=torch.complex128)
    sigma_x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    sigma_y = torch.tensor([[0.0, -1.0j], [1.0j, 0.0]], dtype=torch.complex128)
    sigma_z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex128)
    return identity, sigma_x, sigma_y, sigma_z


def _wilson_dirac_hamiltonian(mass: float, shape: tuple[int, int, int] = (4, 4, 4)):
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(*shape)),
        unit_cell={"r": ImmutableDenseMatrix.zeros(3, 1)},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(("band", orbital, spin),))
        for orbital in range(2)
        for spin in range(2)
    )
    identity, sigma_x, sigma_y, sigma_z = _pauli()
    beta = torch.kron(sigma_z, identity)
    alpha_x = torch.kron(sigma_x, sigma_x)
    alpha_y = torch.kron(sigma_x, sigma_y)
    alpha_z = torch.kron(sigma_x, sigma_z)
    blocks = []
    for momentum in k_space.elements():
        kx = 2.0 * math.pi * float(momentum.rep[0])
        ky = 2.0 * math.pi * float(momentum.rep[1])
        kz = 2.0 * math.pi * float(momentum.rep[2])
        mass_term = mass + math.cos(kx) + math.cos(ky) + math.cos(kz)
        blocks.append(
            mass_term * beta
            + math.sin(kx) * alpha_x
            + math.sin(ky) * alpha_y
            + math.sin(kz) * alpha_z
        )
    hamiltonian = Tensor(
        data=torch.stack(blocks),
        dims=(k_space, band_space, band_space),
    )
    inversion = Tensor(
        data=beta.expand(hamiltonian.data.shape[0], -1, -1).clone(),
        dims=hamiltonian.dims,
    )
    return hamiltonian, inversion


@pytest.mark.parametrize(
    ("mass", "indices"),
    [
        (-2.0, (1, 0, 0, 0)),
        (-4.0, (0, 0, 0, 0)),
        (0.0, (0, 1, 1, 1)),
    ],
)
def test_z2_parity_indices_of_wilson_dirac_phases(mass, indices):
    hamiltonian, inversion = _wilson_dirac_hamiltonian(mass)

    result = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="parity",
    )

    assert result["method"] == "parity"
    assert result["indices"] == indices
    assert result["direct_gap"] > 0.1


def test_z2_auto_uses_parity_when_inversion_is_supplied():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0)

    result = z2_indices(hamiltonian, n_occupied=2, inversion=inversion)

    assert result["method"] == "parity"
    assert result["indices"] == (1, 0, 0, 0)


def test_z2_wilson_indices_agree_with_parity_for_strong_ti():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0)

    result = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="wilson",
        n_loop=16,
        n_perp=9,
    )

    assert result["method"] == "wilson"
    assert result["indices"] == (1, 0, 0, 0)
    assert result["min_gap"] > 0.1


def test_z2_rejects_odd_occupied_count():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0)

    with pytest.raises(ValueError, match="even occupied count"):
        z2_indices(hamiltonian, n_occupied=1, inversion=inversion, method="parity")


def test_z2_rejects_full_occupation():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))

    with pytest.raises(ValueError, match="strictly between"):
        z2_indices(hamiltonian, n_occupied=4, inversion=inversion, method="parity")


def test_z2_rejects_invalid_method():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))

    with pytest.raises(ValueError, match="method must be"):
        z2_indices(hamiltonian, 2, inversion=inversion, method="invalid")  # type: ignore[call-overload]


def test_z2_rejects_two_dimensional_hamiltonian():
    hamiltonian = _chern_insulator(ImmutableDenseMatrix.diag(4, 4))

    with pytest.raises(ValueError, match="three-dimensional"):
        z2_indices(hamiltonian, n_occupied=2, method="wilson")


def test_z2_parity_requires_inversion_or_offset_labels():
    hamiltonian, _inversion = _wilson_dirac_hamiltonian(-2.0)

    with pytest.raises(RuntimeError, match="Cannot build inversion"):
        z2_indices(hamiltonian, n_occupied=2, method="parity")


def test_z2_both_prefers_parity_indices():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0)

    result = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="both",
        n_loop=16,
        n_perp=9,
    )

    assert result["method"] == "both"
    assert result["indices"] == (1, 0, 0, 0)
    assert result["parity"]["indices"] == (1, 0, 0, 0)
    assert result["wilson"]["indices"] == (1, 0, 0, 0)


def test_z2_auto_falls_back_to_wilson_without_inversion():
    hamiltonian, _inversion = _wilson_dirac_hamiltonian(-2.0)

    with pytest.warns(RuntimeWarning, match="Parity method unavailable"):
        result = z2_indices(
            hamiltonian,
            n_occupied=2,
            method="auto",
            n_loop=16,
            n_perp=9,
        )

    assert result["method"] == "wilson"
    assert result["indices"] == (1, 0, 0, 0)


def _two_site_bonding_insulator(shape: tuple[int, int, int] = (4, 4, 4)) -> Tensor:
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(*shape)),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0, 0]),
            "B": ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    site_a = Offset(rep=ImmutableDenseMatrix([0, 0, 0]), space=lattice)
    site_b = Offset(
        rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0]), space=lattice
    )
    band_space = HilbertSpace.new(
        [
            U1Basis.new(site_a, Spin.up),
            U1Basis.new(site_a, Spin.down),
            U1Basis.new(site_b, Spin.up),
            U1Basis.new(site_b, Spin.down),
        ]
    )
    hopping = torch.zeros((4, 4), dtype=torch.complex128)
    hopping[0, 2] = hopping[2, 0] = 1.0
    hopping[1, 3] = hopping[3, 1] = 1.0
    return Tensor(
        data=hopping.expand(k_space.dim, -1, -1).clone(),
        dims=(k_space, band_space, band_space),
    )


def test_z2_parity_assembles_inversion_from_offset_labels():
    hamiltonian = _two_site_bonding_insulator()

    result = z2_indices(hamiltonian, n_occupied=2, method="parity")

    assert result["method"] == "parity"
    assert result["indices"] == (0, 0, 0, 0)
    assert result["direct_gap"] == pytest.approx(2.0)


def test_z2_auto_uses_offset_assembled_parity():
    hamiltonian = _two_site_bonding_insulator()

    result = z2_indices(hamiltonian, n_occupied=2)

    assert result["method"] == "parity"
    assert result["indices"] == (0, 0, 0, 0)


def test_z2_parity_on_odd_mesh_interpolates_trim():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0, shape=(3, 3, 3))

    result = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="parity",
    )

    assert result["indices"] == (1, 0, 0, 0)
    assert result["direct_gap"] > 0.1


def test_z2_wilson_indices_of_weak_ti():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(0.0)

    result = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="wilson",
        n_loop=16,
        n_perp=9,
    )

    assert result["method"] == "wilson"
    assert result["indices"] == (0, 1, 1, 1)


def test_z2_rejects_sheared_periodic_cell():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        boundaries=PeriodicBoundary(
            ImmutableDenseMatrix([[2, 1, 0], [0, 2, 0], [0, 0, 2]])
        ),
        unit_cell={"r": ImmutableDenseMatrix.zeros(3, 1)},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=(("band", i),)) for i in range(4)
    )
    identity = torch.eye(4, dtype=torch.complex128)
    hamiltonian = Tensor(
        data=identity.expand(k_space.dim, -1, -1).clone(),
        dims=(k_space, band_space, band_space),
    )

    with pytest.raises(ValueError, match="diagonal periodic cell"):
        z2_indices(hamiltonian, n_occupied=2, method="wilson")


def test_z2_both_requires_parity():
    hamiltonian, _inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))

    with pytest.raises(RuntimeError, match="Cannot build inversion"):
        z2_indices(
            hamiltonian,
            n_occupied=2,
            method="both",
            n_loop=8,
            n_perp=5,
        )


def test_z2_parity_aligns_reordered_band_axes():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))
    reference = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="parity",
    )["indices"]
    order = [1, 0, 3, 2]
    reordered_columns = hamiltonian.dims[2][order]
    reordered = Tensor(
        data=hamiltonian.data[..., order],
        dims=(hamiltonian.dims[0], hamiltonian.dims[1], reordered_columns),
    )

    result = z2_indices(
        reordered,
        n_occupied=2,
        inversion=inversion,
        method="parity",
    )

    assert result["indices"] == reference


def test_z2_parity_returns_labeled_tensors():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0)

    parity = z2_indices(
        hamiltonian, n_occupied=2, inversion=inversion, method="parity"
    )
    eigenvalues = next(iter(parity["diagnostics"].values()))["parity_eigenvalues"]
    assert isinstance(eigenvalues, Tensor)
    assert eigenvalues.data.shape == (2,)

    wilson = z2_indices(
        hamiltonian,
        n_occupied=2,
        inversion=inversion,
        method="wilson",
        n_loop=16,
        n_perp=9,
    )
    plane = next(iter(wilson["planes"].values()))
    assert isinstance(plane["wcc"], Tensor)
    assert isinstance(plane["gap_pos"], Tensor)
    assert isinstance(plane["sweep"], Tensor)
    assert plane["wcc"].data.shape[0] == 9
    assert plane["wcc"].data.shape[1] == 2


def test_z2_warns_when_time_reversal_is_broken():
    hamiltonian = _two_site_bonding_insulator()
    zeeman = torch.diag(torch.tensor([0.3, -0.3, 0.3, -0.3], dtype=torch.complex128))
    broken = Tensor(data=hamiltonian.data + zeeman, dims=hamiltonian.dims)

    with pytest.warns(RuntimeWarning, match="Kramers-degenerate|Time-reversal"):
        z2_indices(broken, n_occupied=2, method="parity")


def test_z2_accepts_diagonal_momentum_block_space():
    hamiltonian, inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))
    momenta = tuple(hamiltonian.dims[0].elements())
    pair_space = MomentumBlockSpace(
        structure=OrderedDict(((momentum, momentum), i) for i, momentum in enumerate(momenta))
    )
    blocked = Tensor(
        data=hamiltonian.data,
        dims=(pair_space, hamiltonian.dims[1], hamiltonian.dims[2]),
    )
    inversion_blocked = Tensor(data=inversion.data, dims=blocked.dims)

    result = z2_indices(
        blocked, n_occupied=2, inversion=inversion_blocked, method="parity"
    )

    assert result["indices"] == (1, 0, 0, 0)


def test_z2_rejects_off_diagonal_momentum_blocks():
    hamiltonian, _inversion = _wilson_dirac_hamiltonian(-2.0, shape=(2, 2, 2))
    momenta = tuple(hamiltonian.dims[0].elements())
    pairs = tuple(
        (momenta[i], momenta[(i + 1) % len(momenta)]) for i in range(len(momenta))
    )
    pair_space = MomentumBlockSpace(
        structure=OrderedDict((pair, i) for i, pair in enumerate(pairs))
    )
    blocked = Tensor(
        data=hamiltonian.data,
        dims=(pair_space, hamiltonian.dims[1], hamiltonian.dims[2]),
    )

    with pytest.raises(ValueError, match="diagonal momentum-block"):
        z2_indices(blocked, n_occupied=2, method="wilson")
