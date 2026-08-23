import warnings

import numpy as np
import pytest
import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

from qten.bands import (
    assert_pure,
    bandcounts,
    bandfillings,
    nearest_bands,
    bandselect,
    interpolate_path,
    _infer_wannier_bridge,
    fhs_chern_number,
    proj_wannierization,
    svd_projection,
    von_neumann,
)
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.fourier import fourier_transform
from qten.geometries.spatials import AffineSpace, KPointSet, Lattice, Offset
from qten.linalg.tensors import Tensor
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import (
    BzPath,
    IndexSpace,
    MomentumBlockSpace,
    MomentumSpace,
    brillouin_zone,
)


def _space(name: str, n: int) -> HilbertSpace:
    return HilbertSpace.new(
        U1Basis(coef=sy.Integer(1), base=((name, i),)) for i in range(n)
    )


def _mode(r: Offset, orb: str = "s") -> U1Basis:
    return U1Basis(coef=sy.Integer(1), base=(r, orb))


def _band_tensor() -> tuple[Tensor, HilbertSpace]:
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 4)

    energies = torch.tensor(
        [
            [-3.0, -1.0, 2.0, 4.0],
            [-2.0, 1.0, 3.0, 5.0],
        ],
        dtype=torch.float64,
    )
    data = torch.diag_embed(energies).to(torch.complex128)
    tensor = Tensor(data=data, dims=(k_space, band_space, band_space))
    return tensor, band_space


def test_fhs_chern_number_for_two_band_chern_insulator():
    size = 8
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(size, size)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("chern", 2)
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
    hamiltonian = Tensor(
        data=torch.stack(blocks), dims=(k_space, band_space, band_space)
    )

    result = fhs_chern_number(hamiltonian, n_occupied=1)

    assert abs(result["nearest_integer"]) == 1
    assert result["chern"] == pytest.approx(result["nearest_integer"], abs=1e-12)
    assert result["direct_gap"] > 0
    assert result["berry_flux"].shape == (size, size)


def test_bandselect_supports_slice_criterion():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, lowest_two=slice(0, 2))["lowest_two"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.eye(4, dtype=torch.complex128)[:, :2].expand(2, -1, -1)
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_index_tuple_criterion():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, picked=(0, 2))["picked"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.eye(4, dtype=torch.complex128)[:, (0, 2)].expand(2, -1, -1)
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_energy_window_criterion_with_padding():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, window=(-1.5, 2.5))["window"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.zeros((2, 4, 2), dtype=torch.complex128)
    expected[0, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 1]
    expected[0, :, 1] = torch.eye(4, dtype=torch.complex128)[:, 2]
    expected[1, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 1]
    assert torch.allclose(selected.data, expected)


def test_bandselect_supports_callable_criterion_with_padding():
    tensor, band_space = _band_tensor()

    selected = bandselect(tensor, negative=lambda e: e < 0)["negative"]

    assert selected.dims[:2] == (tensor.dims[0], band_space)
    assert isinstance(selected.dims[2], IndexSpace)
    assert selected.dims[2].dim == 2
    expected = torch.zeros((2, 4, 2), dtype=torch.complex128)
    expected[0, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 0]
    expected[0, :, 1] = torch.eye(4, dtype=torch.complex128)[:, 1]
    expected[1, :, 0] = torch.eye(4, dtype=torch.complex128)[:, 0]
    assert torch.allclose(selected.data, expected)


def test_bandcounts_counts_nonzero_columns_per_momentum_sector():
    tensor, band_space = _band_tensor()
    selected = bandselect(tensor, window=(-1.5, 2.5))["window"]

    counts = bandcounts(selected)

    assert counts.dims == (tensor.dims[0],)
    assert counts.data.dtype == torch.int64
    assert torch.equal(counts.data, torch.tensor([2, 1], dtype=torch.int64))


def test_bandcounts_accepts_hilbertspace_as_trailing_statespace():
    tensor, _ = _band_tensor()

    counts = bandcounts(tensor)

    assert counts.dims == (tensor.dims[0],)
    assert torch.equal(counts.data, torch.tensor([4, 4], dtype=torch.int64))


def test_von_neumann_returns_zero_for_projector():
    tensor, _ = _band_tensor()

    entropy = von_neumann(tensor)
    per_k = von_neumann(tensor, mode="per-k")

    assert entropy == pytest.approx(0.0)
    assert per_k.dims == (tensor.dims[0],)
    assert torch.allclose(per_k.data, torch.zeros(2, dtype=torch.float64))


def test_von_neumann_returns_momentum_resolved_entropy():
    tensor, band_space = _band_tensor()
    mixed = Tensor(
        data=torch.diag_embed(
            torch.tensor(
                [
                    [0.0, 1.0, 1.0, 0.0],
                    [0.0, 0.5, 1.0, 0.0],
                ],
                dtype=torch.float64,
            )
        ).to(torch.complex128),
        dims=(tensor.dims[0], band_space, band_space),
    )

    per_k = von_neumann(mixed, mode="per-k")
    mean = von_neumann(mixed, mode="mean")
    total = von_neumann(mixed, mode="sum")

    expected = torch.tensor([0.0, np.log(2.0)], dtype=torch.float64)
    assert torch.allclose(per_k.data, expected)
    assert mean == pytest.approx(float(expected.mean().item()))
    assert total == pytest.approx(float(expected.sum().item()))


def test_von_neumann_rejects_unknown_mode():
    tensor, _ = _band_tensor()

    with pytest.raises(ValueError, match="mode must be one of"):
        von_neumann(tensor, mode="bad")  # type: ignore[arg-type]


def test_assert_pure_accepts_projector_tensor():
    tensor, _ = _band_tensor()

    assert_pure(tensor)


def test_assert_pure_reports_bad_momenta():
    tensor, band_space = _band_tensor()
    mixed = Tensor(
        data=torch.diag_embed(
            torch.tensor(
                [
                    [0.0, 0.5, 1.0, 0.0],
                    [0.25, 1.0, 1.0, 0.0],
                ],
                dtype=torch.float64,
            )
        ).to(torch.complex128),
        dims=(tensor.dims[0], band_space, band_space),
    )

    with pytest.raises(AssertionError, match="Band tensor is not pure") as excinfo:
        assert_pure(mixed)

    message = str(excinfo.value)
    assert "2 / 2 momentum sectors" in message
    assert "k[0]" in message
    assert "k[1]" in message
    assert "S =" in message


def test_assert_pure_uses_user_threshold():
    tensor, band_space = _band_tensor()
    almost_pure = Tensor(
        data=torch.diag_embed(
            torch.tensor(
                [
                    [0.0, 1e-12, 1.0 - 1e-12, 0.0],
                    [0.0, 1.0, 1.0, 0.0],
                ],
                dtype=torch.float64,
            )
        ).to(torch.complex128),
        dims=(tensor.dims[0], band_space, band_space),
    )

    assert_pure(almost_pure)

    with pytest.raises(AssertionError):
        assert_pure(almost_pure, threshold=1e-12)


def test_assert_pure_rejects_negative_threshold():
    tensor, _ = _band_tensor()

    with pytest.raises(ValueError, match="non-negative threshold"):
        assert_pure(tensor, threshold=-1.0)


def test_assert_pure_uses_user_report_limit():
    tensor, band_space = _band_tensor()
    mixed = Tensor(
        data=torch.full(
            (tensor.dims[0].dim, band_space.dim),
            0.5,
            dtype=torch.complex128,
        ).diag_embed(),
        dims=(tensor.dims[0], band_space, band_space),
    )

    with pytest.raises(AssertionError) as excinfo:
        assert_pure(mixed, report_limit=1)

    message = str(excinfo.value)
    assert "Largest 1 violating momentum sectors:" in message
    assert message.count("S =") == 1


@pytest.mark.parametrize("report_limit", [0, -1])
def test_assert_pure_rejects_non_positive_report_limit(report_limit: int):
    tensor, _ = _band_tensor()

    with pytest.raises(ValueError, match="positive report_limit"):
        assert_pure(tensor, report_limit=report_limit)


@pytest.mark.parametrize("report_limit", [1.5, True])
def test_assert_pure_rejects_non_integer_report_limit(report_limit: object):
    tensor, _ = _band_tensor()

    with pytest.raises(TypeError, match="report_limit to be an integer"):
        assert_pure(tensor, report_limit=report_limit)  # type: ignore[arg-type]


def test_svd_projection_ignores_zero_padded_filled_bands():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    seed_space = IndexSpace.linear(2)

    energies = torch.tensor(
        [[-2.0, 0.0], [-1.0, -1.0]],
        dtype=torch.float64,
    )
    hamiltonian = Tensor(
        data=torch.diag_embed(energies).to(torch.complex128),
        dims=(k_space, band_space, band_space),
    )
    filled = bandfillings(hamiltonian, 0.5)

    seeds = Tensor(
        data=torch.eye(2, dtype=torch.complex128).expand(k_space.dim, -1, -1),
        dims=(k_space, band_space, seed_space),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = svd_projection(filled, seeds)

    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert torch.allclose(projected.data, filled.data)
    assert torch.allclose(
        projected.data[0, :, 1], torch.zeros(2, dtype=torch.complex128)
    )


def test_svd_projection_ignores_zero_padded_bands_and_sources():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    packed_space = IndexSpace.linear(2)

    eigenvectors = Tensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    seeds = Tensor(
        data=torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = svd_projection(eigenvectors, seeds)

    assert not [w for w in caught if issubclass(w.category, UserWarning)]
    assert torch.allclose(projected.data, eigenvectors.data)
    assert torch.allclose(
        projected.data[0, :, 1], torch.zeros(2, dtype=torch.complex128)
    )


def test_svd_projection_warns_but_keeps_full_active_source_block():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    packed_space = IndexSpace.linear(2)

    target = Tensor(
        data=torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    source = Tensor(
        data=torch.tensor(
            [[[1.0, 1.0], [0.0, 0.0]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        projected = svd_projection(target, source)

    assert [w for w in caught if issubclass(w.category, UserWarning)]
    gram = projected.h(-2, -1) @ projected
    assert torch.allclose(
        gram.data[0], torch.eye(2, dtype=torch.complex128), atol=1e-12
    )


def test_svd_projection_ignores_numerically_noisy_padded_columns():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    packed_space = IndexSpace.linear(2)

    target = Tensor(
        data=torch.tensor(
            [[[1.0, 1e-10], [0.0, 1e-10]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    source = Tensor(
        data=torch.tensor(
            [[[1.0, -1e-10], [0.0, 1e-10]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    projected = svd_projection(target, source)
    singular_values = torch.linalg.svdvals(projected.data[0])

    assert singular_values[1].abs() < 1e-12


def test_svd_projection_output_rank_is_bounded_by_target_rank():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 3)
    packed_space = IndexSpace.linear(3)

    target = Tensor(
        data=torch.tensor(
            [[[1.0, 0.0, 1e-9], [0.0, 1.0, -1e-9], [0.0, 0.0, 1e-9]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    source = Tensor(
        data=torch.tensor(
            [[[1.0, 0.5, -0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    projected = svd_projection(target, source)
    singular_values = torch.linalg.svdvals(projected.data[0])

    assert singular_values[2].abs() < 1e-12


def test_svd_projection_ignores_zero_padded_columns_in_middle_across_k():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 3)
    packed_space = IndexSpace.linear(3)

    target = Tensor(
        data=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    source = Tensor(
        data=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    projected = svd_projection(target, source)

    assert torch.allclose(projected.data, source.data)
    assert torch.allclose(
        projected.data[0, :, 1], torch.zeros(3, dtype=torch.complex128)
    )


def test_svd_projection_scatter_restores_original_source_column_positions():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 3)
    packed_space = IndexSpace.linear(3)

    target = Tensor(
        data=torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )
    source = Tensor(
        data=torch.tensor(
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            dtype=torch.complex128,
        ),
        dims=(k_space, band_space, packed_space),
    )

    projected = svd_projection(target, source)
    expected = torch.tensor(
        [[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.complex128,
    )

    assert torch.allclose(projected.data, expected)
    assert torch.allclose(
        projected.data[0, :, 1], torch.zeros(3, dtype=torch.complex128)
    )
    assert torch.allclose(
        projected.data[0, :, 0], torch.tensor([0.0, 1.0, 0.0], dtype=torch.complex128)
    )
    assert torch.allclose(
        projected.data[0, :, 2], torch.tensor([1.0, 0.0, 0.0], dtype=torch.complex128)
    )


def test_proj_wannierization_matches_explicit_fourier_then_svd():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)

    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lattice.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lattice.affine)
    bloch_space = HilbertSpace.new([_mode(r0, "s")])
    local_seed_space = HilbertSpace.new([_mode(r0, "s"), _mode(r1, "s")])
    seed_columns = IndexSpace.linear(1)

    seeds = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(local_seed_space, seed_columns),
    )
    crystal_seeds = fourier_transform(k_space, bloch_space, local_seed_space) @ seeds
    eigenvectors = Tensor(
        data=crystal_seeds.data.clone(),
        dims=crystal_seeds.dims,
    )

    projected = proj_wannierization(
        eigenvectors, seeds, svd_threshold=1e-6, wannierize_lattice=False
    )
    expected = svd_projection(eigenvectors, crystal_seeds, svd_threshold=1e-6)

    assert projected.dims == expected.dims
    assert torch.allclose(projected.data, expected.data)


def test_proj_wannierization_infers_lattice_when_seed_columns_are_hilbert_labeled():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)

    r0 = Offset(rep=ImmutableDenseMatrix([0]), space=lattice.affine)
    r1 = Offset(rep=ImmutableDenseMatrix([1]), space=lattice.affine)
    bloch_space = HilbertSpace.new([_mode(r0, "s")])
    local_seed_space = HilbertSpace.new([_mode(r0, "s"), _mode(r1, "s")])
    seed_col_space = HilbertSpace.new([_mode(r0, "s")])

    seeds = Tensor(
        data=torch.tensor([[1.0], [0.0]], dtype=torch.complex128),
        dims=(local_seed_space, seed_col_space),
    )
    eigenvectors = fourier_transform(k_space, bloch_space, local_seed_space) @ seeds

    projected = proj_wannierization(
        eigenvectors, seeds, svd_threshold=1e-6, wannierize_lattice=True
    )

    assert isinstance(projected.dims[0], MomentumBlockSpace)
    inferred_offset = projected.dims[2].elements()[0].irrep_of(Offset)
    assert isinstance(inferred_offset.space, Lattice)
    assert inferred_offset.fractional().rep == ImmutableDenseMatrix([0])


def test_infer_wannier_bridge_rebases_seed_offsets_before_unit_cell_extraction():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 1)
    seed_space = HilbertSpace.new(
        [
            U1Basis(
                coef=sy.Integer(1),
                base=(
                    Offset(
                        rep=ImmutableDenseMatrix([sy.Rational(1, 2)]),
                        space=AffineSpace(ImmutableDenseMatrix([[2]])),
                    ),
                    "seed",
                ),
            )
        ]
    )

    eigenvectors = Tensor(
        data=torch.ones((k_space.dim, band_space.dim, 1), dtype=torch.complex128),
        dims=(k_space, band_space, IndexSpace.linear(1)),
    )

    bridge = _infer_wannier_bridge(eigenvectors, seed_space)

    assert bridge is not None
    inferred_offset = bridge.dims[2].elements()[0].irrep_of(Offset)
    assert inferred_offset.space.unit_cell["r0"].rep == ImmutableDenseMatrix([0])
    assert inferred_offset.fractional().rep == ImmutableDenseMatrix([0])


def test_svd_projection_lattice_output_keeps_lattice_backed_offsets_on_dim_2():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(1, 1)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 1)
    seed_space = HilbertSpace.new(
        [
            U1Basis(
                coef=sy.Integer(1),
                base=(
                    Offset(
                        rep=ImmutableDenseMatrix(
                            [sy.Rational(1, 8), sy.Rational(1, 8)]
                        ),
                        space=lattice.affine,
                    ),
                    "seed",
                ),
            )
        ]
    )

    eigenvectors = Tensor(
        data=torch.ones((k_space.dim, band_space.dim, 1), dtype=torch.complex128),
        dims=(k_space, band_space, IndexSpace.linear(1)),
    )
    seeds = Tensor(
        data=torch.ones(
            (k_space.dim, band_space.dim, seed_space.dim), dtype=torch.complex128
        ),
        dims=(k_space, band_space, seed_space),
    )

    projected = svd_projection(eigenvectors, seeds, infer_lattice=True)

    inferred_offset = projected.dims[2].elements()[0].irrep_of(Offset)
    assert isinstance(inferred_offset.space, Lattice)
    assert str(inferred_offset) == "r[r0; 1/8, 1/8]"


# --- interpolate_path tests ---


def _recip_2d():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    return lattice.dual


def _recip_3d():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0, 0])},
    )
    return lattice.dual


def test_interpolate_path_returns_bzpath_with_correct_n_points():
    recip = _recip_2d()
    waypoints = ["G", "X", "M"]
    kpoints = KPointSet.from_points(
        recip, {"G": (0, 0), "X": (0.5, 0), "M": (0.5, 0.5)}
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=50)

    assert isinstance(path, BzPath)
    assert isinstance(path.k_space, MomentumSpace)
    assert len(path.path_order) == 50
    assert path.k_space.dim == 50


def test_interpolate_path_waypoint_indices_match_waypoints():
    recip = _recip_2d()
    waypoints = ["G", "X", "M", "G"]
    kpoints = KPointSet.from_points(
        recip, {"G": (0, 0), "X": (0.5, 0), "M": (0.5, 0.5)}
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=100)

    assert len(path.waypoint_indices) == len(waypoints)
    assert len(path.path_order) == 100
    assert path.waypoint_indices[0] == 0
    assert path.waypoint_indices[-1] == 99
    assert path.labels == tuple(waypoints)

    elements = path.k_space.elements()
    for wp_pos, wp in zip(path.waypoint_indices, waypoints):
        k_idx = path.path_order[wp_pos]
        k = elements[k_idx]
        frac = np.array([float(k.rep[j, 0]) for j in range(recip.dim)])
        expected_k = kpoints.points[wp]
        expected = np.array([float(expected_k.rep[j, 0]) for j in range(recip.dim)])
        assert np.allclose(frac, expected, atol=1e-9)


def test_interpolate_path_requires_kpointset():
    recip = _recip_2d()
    with pytest.raises(TypeError, match="KPointSet"):
        interpolate_path(recip, ["G", "X"], None, n_points=10)


def test_interpolate_path_too_few_waypoints_raises():
    recip = _recip_2d()
    kpoints = KPointSet.from_points(recip, {"G": (0, 0)})
    with pytest.raises(ValueError, match="two waypoints"):
        interpolate_path(recip, ["G"], kpoints, n_points=10)


def test_interpolate_path_3d_lattice():
    recip = _recip_3d()
    waypoints = ["G", "X", "M", "G"]
    kpoints = KPointSet.from_points(
        recip, {"G": (0, 0, 0), "X": (0.5, 0, 0), "M": (0.5, 0.5, 0)}
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=80)

    assert len(path.path_order) == 80
    # Closed loop: first and last waypoint share one k-point, so k_space has
    # one fewer unique element than path length.
    assert path.k_space.dim == 79
    assert len(path.waypoint_indices) == 4
    assert len(path.labels) == 4


def test_interpolate_path_distributes_proportionally():
    recip = _recip_2d()
    waypoints = ["G", "X", "M"]
    kpoints = KPointSet.from_points(
        recip,
        {
            "G": (0, 0),
            "X": (sy.Rational(2, 3), 0),
            "M": (sy.Rational(2, 3), sy.Rational(1, 3)),
        },
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=100)

    idx0, idx1, idx2 = path.waypoint_indices
    seg1_count = idx1 - idx0
    seg2_count = idx2 - idx1

    # Segment 1 has length 2/3, segment 2 has length 1/3 (in Cartesian space
    # with unit-basis reciprocal lattice), so ~2:1 ratio.
    assert seg1_count > seg2_count
    assert abs(seg1_count / max(seg2_count, 1) - 2.0) < 0.5


def test_interpolate_path_path_positions_are_monotonic():
    recip = _recip_2d()
    waypoints = ["G", "X", "M", "G"]
    kpoints = KPointSet.from_points(
        recip, {"G": (0, 0), "X": (0.5, 0), "M": (0.5, 0.5)}
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=60)

    positions = np.array(path.path_positions)
    assert positions[0] == 0.0
    assert np.all(np.diff(positions) >= -1e-15)
    assert len(positions) == len(path.path_order)


def test_interpolate_path_closed_loop_deduplicates():
    recip = _recip_2d()
    waypoints = ["G", "X", "G"]
    kpoints = KPointSet.from_points(recip, {"G": (0, 0), "X": (0.5, 0)})
    path = interpolate_path(recip, waypoints, kpoints, n_points=20)

    assert len(path.path_order) == 20
    # First and last path positions map to the same k_space index.
    assert path.path_order[0] == path.path_order[-1]
    assert path.k_space.dim == 19


def test_interpolate_path_named_route_uses_kpointset():
    recip = _recip_2d()
    waypoints = ["Gamma", "X", "M", "Gamma"]
    kpoints = KPointSet.from_points(
        recip, {"Gamma": (0, 0), "X": (0.5, 0), "M": (0.5, 0.5)}
    )
    path = interpolate_path(recip, waypoints, kpoints, n_points=40)

    assert len(path.path_order) == 40
    assert path.labels == tuple(waypoints)

    elements = path.k_space.elements()
    first_k = elements[path.path_order[0]]
    frac = np.array([float(first_k.rep[j, 0]) for j in range(recip.dim)])
    assert np.allclose(frac, [0, 0], atol=1e-9)


def test_interpolate_path_named_route_missing_point_raises():
    recip = _recip_2d()
    waypoints = ["Gamma", "Z"]
    kpoints = KPointSet.from_points(recip, {"Gamma": (0, 0)})
    with pytest.raises(ValueError, match="not found"):
        interpolate_path(recip, waypoints, kpoints, n_points=10)


def test_interpolate_path_rebases_from_kpointset_source_recip():
    recip_target = _recip_2d()
    lattice_other = Lattice(
        basis=ImmutableDenseMatrix([[2, 0], [0, 1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    recip_other = lattice_other.dual

    kpoints = KPointSet.from_points(
        recip_other,
        {
            "G": (0, 0),
            "X": (sy.Rational(1, 2), sy.Rational(1, 4)),
        },
    )
    waypoints = ["G", "X"]
    path = interpolate_path(recip_target, waypoints, kpoints, n_points=8)
    elements = path.k_space.elements()
    first_k = elements[path.path_order[path.waypoint_indices[0]]]
    last_k = elements[path.path_order[path.waypoint_indices[1]]]

    expected_start = kpoints.points["G"].rebase(recip_target)
    expected_end = kpoints.points["X"].rebase(recip_target)
    assert first_k == expected_start
    assert last_k == expected_end


def test_interpolate_path_wraps_rebased_waypoints_to_fractional_cell():
    recip = _recip_2d()
    kpoints = KPointSet.from_points(
        recip,
        {
            "G": (0, 0),
            "X": (sy.Rational(3, 2), sy.Rational(-1, 4)),
        },
    )

    waypoints = ["G", "X"]
    path = interpolate_path(recip, waypoints, kpoints, n_points=8)
    elements = path.k_space.elements()
    end_k = elements[path.path_order[path.waypoint_indices[1]]]

    assert end_k == kpoints.points["X"].rebase(recip).fractional()
    assert end_k.rep == ImmutableDenseMatrix([sy.Rational(1, 2), sy.Rational(3, 4)])


def test_interpolate_path_can_preserve_unwrapped_rebased_waypoints():
    recip = _recip_2d()
    kpoints = KPointSet.from_points(
        recip,
        {
            "G": (0, 0),
            "X": (sy.Rational(3, 2), sy.Rational(-1, 4)),
        },
    )

    waypoints = ["G", "X"]
    path = interpolate_path(
        recip,
        waypoints,
        kpoints,
        n_points=8,
        wrap_fractional=False,
    )
    elements = path.k_space.elements()
    end_k = elements[path.path_order[path.waypoint_indices[1]]]

    assert end_k == kpoints.points["X"].rebase(recip)
    assert end_k.rep == ImmutableDenseMatrix([sy.Rational(3, 2), sy.Rational(-1, 4)])


def test_interpolate_path_accessible_via_ops():
    from qten.bands import interpolate_path as ip

    recip = _recip_2d()
    waypoints = ["G", "X"]
    kpoints = KPointSet.from_points(recip, {"G": (0, 0), "X": (0.5, 0)})
    path = ip(recip, waypoints, kpoints, n_points=10)
    assert isinstance(path, BzPath)


# --- bands_near_value_as_tensor_KHH tests ---


def _nondiag_band_tensor() -> Tensor:
    """Build a (2, 2, 2) Hamiltonian with non-diagonal anchor eigenbasis.

    H(k=Gamma) = [[0, 1], [1, 0]] has eigvecs (1,-1)/sqrt(2) (eigval -1)
    and (1, 1)/sqrt(2) (eigval +1).
    H(k=X)     = [[1, 2], [2, 3]] is used to exercise the projection math.
    """
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)

    h_gamma = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex128)
    h_x = torch.tensor([[1.0, 2.0], [2.0, 3.0]], dtype=torch.complex128)
    data = torch.stack([h_gamma, h_x], dim=0)
    return Tensor(data=data, dims=(k_space, band_space, band_space))


def test_bands_near_value_selects_single_band_at_gamma():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=-1.0, tol=1e-6)

    assert result.dims[0] == tensor.dims[0]
    assert isinstance(result.dims[1], IndexSpace)
    assert isinstance(result.dims[2], IndexSpace)
    assert result.dims[1].dim == 1
    assert result.dims[2].dim == 1
    # Diagonal H: eigenvector for eigenvalue -1 at Gamma is e_1, so the
    # projection just picks the (1, 1) diagonal entry at every k.
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_selects_multiple_bands_in_tolerance_window():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=0.5, tol=1.6)

    assert result.dims[1].dim == 2
    assert result.dims[2].dim == 2
    expected = torch.zeros((2, 2, 2), dtype=torch.complex128)
    expected[0] = torch.diag(torch.tensor([-1.0, 2.0], dtype=torch.complex128))
    expected[1] = torch.diag(torch.tensor([1.0, 3.0], dtype=torch.complex128))
    assert torch.allclose(result.data, expected)


def test_bands_near_value_with_points_dict_non_gamma():
    tensor, _ = _band_tensor()

    result = nearest_bands(
        tensor,
        point="X",
        close_to=1.0,
        tol=1e-6,
        points={"X": (0.5,)},
    )

    assert result.dims[1].dim == 1
    # At X the eigenvalue 1.0 belongs to band 1; the projection picks (1, 1).
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_with_explicit_fractional_tuple():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point=(0.5,), close_to=3.0, tol=1e-6)

    assert result.dims[1].dim == 1
    # Band 2 (eigvalue 3 at X) projects to (2, 2) diagonal entries.
    expected = torch.tensor([[[2.0]], [[3.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_empty_subspace_when_no_match():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, close_to=1000.0, tol=1e-6)

    assert result.dims[1].dim == 0
    assert result.dims[2].dim == 0
    assert result.data.shape == (2, 0, 0)


def test_bands_near_value_non_diagonal_projection_math():
    tensor = _nondiag_band_tensor()

    result = nearest_bands(tensor, point="Gamma", close_to=-1.0, tol=1e-6)

    # Selected eigenvector at Gamma is v = (1, -1)/sqrt(2).
    # v^H H(Gamma) v = -1, v^H H(X) v = 0.5*(1 - 2 - 2 + 3) = 0.
    assert result.dims[1].dim == 1
    expected = torch.tensor([[[-1.0]], [[0.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected, atol=1e-10)


def test_bands_near_value_wraps_fractional_coordinates():
    tensor, _ = _band_tensor()

    result = nearest_bands(tensor, point=(1.0,), close_to=-1.0, tol=1e-6)

    # (1.0,) wraps to Gamma = (0,), so this matches the Gamma selection.
    expected = torch.tensor([[[-1.0]], [[1.0]]], dtype=torch.complex128)
    assert torch.allclose(result.data, expected)


def test_bands_near_value_unknown_label_without_points_raises():
    tensor, _ = _band_tensor()

    with pytest.raises(KeyError, match="not found"):
        nearest_bands(tensor, point="Z")


def test_bands_near_value_dimension_mismatch_raises():
    tensor, _ = _band_tensor()

    with pytest.raises(ValueError, match="coordinates"):
        nearest_bands(tensor, point=(0.0, 0.0))


def test_bands_near_value_rejects_wrong_rank():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    band_space = _space("band", 2)
    data = torch.zeros((2, 2), dtype=torch.complex128)
    rank2 = Tensor(data=data, dims=(k_space, band_space))

    with pytest.raises(ValueError, match="rank 3"):
        nearest_bands(rank2)
