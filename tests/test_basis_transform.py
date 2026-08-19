import torch
import sympy as sy
from dataclasses import dataclass
from sympy import ImmutableDenseMatrix
import pytest

from qten.geometries.spatials import (
    Lattice,
    Offset,
    AffineSpace,
    ReciprocalLattice,
    Momentum,
)
from qten.symbolics.state_space import MomentumSpace, brillouin_zone
from qten.symbolics.hilbert_space import U1Basis, HilbertSpace
from qten.symbolics import same_rays
from qten.linalg.tensors import Tensor
from qten.geometries.basis_transform import BasisTransform, InverseBasisTransform
from qten.bands import (
    bandfold,
    bandtransform,
    bandunfold,
    cartesian_scale,
    get_band_fold,
    get_band_transform,
)
from qten.linalg._mb_tensor import MomentumBlockTensor
from qten.geometries.boundary import PeriodicBoundary
from qten.pointgroups import PointGroupOpr, pointgroup


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
    h_space = HilbertSpace.new([_mode(r_offset)])
    assert h_space.dim == 1

    # 1c. Create an input tensor (4, 1, 1)
    # Data is just a sequence of numbers for easy tracking
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double the unit cell)
    M = ImmutableDenseMatrix([[2]])
    transform = BasisTransform(M)

    # 2. Execute
    tensor_out = bandfold(transform, tensor_in)

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
    h_space = HilbertSpace.new([_mode(r_offset, "s")])
    assert h_space.dim == 1

    # 1c. Create input tensor (4, 1, 1)
    # Data: 0, 1, 2, 3
    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    # 1d. Define scaling matrix (double in both directions)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])
    transform = BasisTransform(M)

    # 2. Execute
    tensor_out = bandfold(transform, tensor_in)

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


def test_get_band_fold_factorizes_bandfold_with_alignable_matrix_dims():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    a_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b_offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice)
    h_space = HilbertSpace.new([_mode(a_offset, "a"), _mode(b_offset, "b")])
    h_space_perm = HilbertSpace.new([_mode(b_offset, "b"), _mode(a_offset, "a")])
    assert same_rays(h_space, h_space_perm)

    data = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2).to(torch.complex128)
    tensor_in = Tensor(data=data, dims=(k_space, h_space_perm, h_space))
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    T_g = get_band_fold(transform, tensor_in, side="right")
    folded_ref = bandfold(transform, tensor_in, opt="right")
    folded_factored = tensor_in @ T_g.h(-2, -1)

    assert isinstance(T_g, MomentumBlockTensor)
    assert isinstance(folded_ref, MomentumBlockTensor)
    assert folded_ref.dims[0] == T_g.h(-2, -1).dims[0]
    assert torch.allclose(
        folded_factored.align_all(folded_ref.dims).data, folded_ref.data
    )


def test_get_band_fold_supports_left_sample_side():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    a_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b_offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice)
    h_space = HilbertSpace.new([_mode(a_offset, "a"), _mode(b_offset, "b")])
    h_space_perm = HilbertSpace.new([_mode(b_offset, "b"), _mode(a_offset, "a")])

    data = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2).to(torch.complex128)
    tensor_in = Tensor(data=data, dims=(k_space, h_space_perm, h_space))
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    T_g = get_band_fold(transform, tensor_in, side="left")
    folded_ref = bandfold(transform, tensor_in, opt="left")
    folded_factored = T_g @ tensor_in

    assert isinstance(T_g, MomentumBlockTensor)
    assert isinstance(folded_ref, MomentumBlockTensor)
    assert folded_ref.dims[0] == T_g.dims[0]
    assert T_g.dims[2] == h_space_perm
    assert torch.allclose(
        folded_factored.align_all(folded_ref.dims).data, folded_ref.data
    )


def test_get_band_fold_rejects_invalid_side():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    a_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b_offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice)
    h_space = HilbertSpace.new([_mode(a_offset, "a"), _mode(b_offset, "b")])
    tensor_in = Tensor(
        data=torch.ones((4, 2, 2), dtype=torch.complex128),
        dims=(k_space, h_space, h_space),
    )
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    with pytest.raises(ValueError, match="side must be 'left' or 'right'"):
        get_band_fold(transform, tensor_in, side="rigth")  # type: ignore[arg-type]


def test_bandfold_supports_both_sides_for_distinct_hilbert_spaces():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    a_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b_offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice)
    left_space = HilbertSpace.new([_mode(a_offset, "la"), _mode(b_offset, "lb")])
    right_space = HilbertSpace.new([_mode(a_offset, "ra"), _mode(b_offset, "rb")])
    tensor_in = Tensor(
        data=torch.arange(16, dtype=torch.float64)
        .reshape(4, 2, 2)
        .to(torch.complex128),
        dims=(k_space, left_space, right_space),
    )
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    left_fold = get_band_fold(transform, tensor_in, side="left")
    right_fold = get_band_fold(transform, tensor_in, side="right")
    folded_ref = bandfold(transform, tensor_in, opt="both")
    folded_factored = left_fold @ tensor_in @ right_fold.h(-2, -1)

    assert isinstance(folded_ref.dims[0], MomentumSpace)
    assert torch.allclose(
        folded_factored.align_all(folded_ref.dims).data, folded_ref.data
    )


def test_bandtransform_one_sided_modes_return_momentum_block_tensors():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    h_space = HilbertSpace.new([_mode(r_offset)])
    tensor_in = Tensor(
        data=torch.arange(4, dtype=torch.float64).reshape(4, 1, 1).to(torch.complex128),
        dims=(k_space, h_space, h_space),
    )
    t = PointGroupOpr(pointgroup("m-x:x"))

    left_transform = get_band_transform(t, tensor_in, side="left")
    right_transform = get_band_transform(t, tensor_in, side="right")
    transformed_left = bandtransform(t, tensor_in, opt="left")
    transformed_right = bandtransform(t, tensor_in, opt="right")

    assert isinstance(left_transform, MomentumBlockTensor)
    assert isinstance(right_transform, MomentumBlockTensor)
    assert isinstance(transformed_left, MomentumBlockTensor)
    assert isinstance(transformed_right, MomentumBlockTensor)
    assert transformed_left.dims[0] == left_transform.dims[0]
    assert transformed_right.dims[0] == right_transform.h(-2, -1).dims[0]
    assert torch.allclose(
        (left_transform @ tensor_in).align_all(transformed_left.dims).data,
        transformed_left.data,
    )
    assert torch.allclose(
        (tensor_in @ right_transform.h(-2, -1)).align_all(transformed_right.dims).data,
        transformed_right.data,
    )


def test_bandtransform_both_returns_plain_momentum_space_tensor():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    h_space = HilbertSpace.new([_mode(r_offset)])
    tensor_in = Tensor(
        data=torch.arange(4, dtype=torch.float64).reshape(4, 1, 1).to(torch.complex128),
        dims=(k_space, h_space, h_space),
    )
    t = PointGroupOpr(pointgroup("m-x:x"))

    transformed = bandtransform(t, tensor_in, opt="both")
    left_transform = get_band_transform(t, tensor_in, side="left")
    right_transform = get_band_transform(t, tensor_in, side="right")

    assert isinstance(transformed.dims[0], MomentumSpace)
    assert torch.allclose(
        (left_transform @ tensor_in @ right_transform.h(-2, -1))
        .align_all(transformed.dims)
        .data,
        transformed.data,
    )


def test_bandfold_rejects_invalid_opt():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    a_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    b_offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice)
    h_space = HilbertSpace.new([_mode(a_offset, "a"), _mode(b_offset, "b")])
    tensor_in = Tensor(
        data=torch.ones((4, 2, 2), dtype=torch.complex128),
        dims=(k_space, h_space, h_space),
    )
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    with pytest.raises(ValueError, match="opt must be 'left', 'right', or 'both'"):
        bandfold(transform, tensor_in, opt="bothe")  # type: ignore[arg-type]


def test_bandunfold_handles_fractional_sector_collisions():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    h_space = HilbertSpace.new([_mode(r_offset)])

    data = torch.arange(4, dtype=torch.float64).reshape(4, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    folded = bandfold(transform, tensor_in)
    unfolded = bandunfold(InverseBasisTransform(transform.M), folded)

    assert unfolded.dims[0].dim == tensor_in.dims[0].dim
    assert unfolded.dims[1].dim == tensor_in.dims[1].dim
    assert unfolded.dims[2].dim == tensor_in.dims[2].dim
    assert torch.allclose(
        unfolded.data, tensor_in.data.to(unfolded.data.dtype), atol=1e-10
    )


def test_bandfold_preserves_transformed_unit_cell_offsets_for_multiorbital_site():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    r_offset = Offset(rep=ImmutableDenseMatrix([0]), space=lattice)
    h_space = HilbertSpace.new([_mode(r_offset, "s"), _mode(r_offset, "p")])

    tensor_in = Tensor(
        data=torch.arange(16, dtype=torch.float64)
        .reshape(4, 2, 2)
        .to(torch.complex128),
        dims=(k_space, h_space, h_space),
    )
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    folded = bandfold(transform, tensor_in)
    folded_offsets = [psi.irrep_of(Offset) for psi in folded.dims[1].elements()]
    folded_reps = [tuple(offset.rep) for offset in folded_offsets]
    expected_reps = [
        tuple(offset.rep)
        for offset in sorted(
            transform(lattice).unit_cell.values(), key=lambda offset: tuple(offset.rep)
        )
    ]

    assert folded.dims[1].dim == 4
    for rep in expected_reps:
        assert folded_reps.count(rep) == 2


def test_bandunfold_preserves_primitive_unit_cell_metadata():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    h_space = HilbertSpace.new(
        [
            _mode(Offset(rep=ImmutableDenseMatrix([0]), space=lattice), "a"),
            _mode(
                Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice),
                "b",
            ),
        ]
    )

    data = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2).to(torch.complex128)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    unfolded = bandunfold(
        InverseBasisTransform(transform.M), bandfold(transform, tensor_in)
    )
    restored_lattice = unfolded.dims[1].elements()[0].irrep_of(Offset).space

    assert len(restored_lattice.unit_cell) == 2
    assert set(restored_lattice.unit_cell.keys()) == {"a", "b"}
    restored_offsets = {site.rep for site in restored_lattice.unit_cell.values()}
    assert ImmutableDenseMatrix([0]) in restored_offsets
    assert ImmutableDenseMatrix([sy.Rational(1, 2)]) in restored_offsets


def test_bandunfold_accepts_inverse_transform():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    k_space = brillouin_zone(lattice.dual)
    h_space = HilbertSpace.new(
        [
            _mode(Offset(rep=ImmutableDenseMatrix([0]), space=lattice), "a"),
            _mode(
                Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2)]), space=lattice),
                "b",
            ),
        ]
    )
    data = torch.arange(16, dtype=torch.float64).reshape(4, 2, 2).to(torch.complex128)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))

    folded = bandfold(transform, tensor_in)
    unfolded = bandunfold(InverseBasisTransform(transform.M), folded)
    assert unfolded.dims[0].dim == tensor_in.dims[0].dim
    assert unfolded.dims[1].dim == tensor_in.dims[1].dim
    assert unfolded.dims[2].dim == tensor_in.dims[2].dim


def test_bandunfold_rejects_forward_basis_transform():
    basis = ImmutableDenseMatrix([[1]])
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={"r": ImmutableDenseMatrix([0])},
    )
    k_space = brillouin_zone(lattice.dual)
    h_space = HilbertSpace.new(
        [_mode(Offset(rep=ImmutableDenseMatrix([0]), space=lattice))]
    )
    tensor_in = Tensor(
        data=torch.arange(4, dtype=torch.float64).reshape(4, 1, 1),
        dims=(k_space, h_space, h_space),
    )
    transform = BasisTransform(ImmutableDenseMatrix([[2]]))
    folded = bandfold(transform, tensor_in)

    with pytest.raises(TypeError, match="InverseBasisTransform"):
        bandunfold(transform, folded)  # type: ignore[arg-type]


def test_affine_space_transform():
    basis = ImmutableDenseMatrix([[1, 0], [0, 1]])
    space = AffineSpace(basis=basis)
    M = ImmutableDenseMatrix([[2, 0], [0, 2]])
    t = BasisTransform(M)

    new_space = t(space)
    assert isinstance(new_space, AffineSpace)
    assert new_space.basis == basis @ M


def test_basis_transform_rejects_non_invertible_matrix():
    with pytest.raises(ValueError, match="positive determinant"):
        BasisTransform(ImmutableDenseMatrix([[1, 0], [0, 0]]))


def test_basis_transform_inv_roundtrip_type():
    forward = BasisTransform(ImmutableDenseMatrix([[2]]))
    inverse = forward.inv()
    assert isinstance(inverse, InverseBasisTransform)
    assert inverse.M == forward.M
    assert isinstance(inverse.inv(), BasisTransform)
    assert inverse.inv().M == forward.M


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


def test_inverse_lattice_transform_recovers_primitive_metadata():
    basis = ImmutableDenseMatrix([[1]])
    lat = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "a": ImmutableDenseMatrix([0]),
            "b": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    M = ImmutableDenseMatrix([[2]])
    forward = BasisTransform(M)
    inverse = InverseBasisTransform(M)

    restored = inverse(forward(lat))
    assert isinstance(restored, Lattice)
    assert restored.basis == lat.basis
    assert restored.boundaries.basis == lat.boundaries.basis
    assert set(restored.unit_cell.keys()) == {"a", "b"}
    restored_offsets = {site.rep for site in restored.unit_cell.values()}
    assert ImmutableDenseMatrix([0]) in restored_offsets
    assert ImmutableDenseMatrix([sy.Rational(1, 2)]) in restored_offsets


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


def test_cartesian_scale_preserves_tensor_information_and_rescales_spaces():
    basis = ImmutableDenseMatrix.diag(2, 3, 4)
    lattice = Lattice(
        basis=basis,
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0])},
    )
    k_space = brillouin_zone(lattice.dual)
    offset = Offset(rep=ImmutableDenseMatrix([sy.Rational(1, 2), 0, 0]), space=lattice)
    h_space = HilbertSpace.new([_mode(offset)])
    data = torch.arange(k_space.dim, dtype=torch.float64).reshape(k_space.dim, 1, 1)
    tensor_in = Tensor(data=data, dims=(k_space, h_space, h_space))

    scaled = cartesian_scale(tensor_in, (sy.Rational(1, 2),) * 3)

    assert scaled.data is tensor_in.data
    assert scaled.data.shape == tensor_in.data.shape
    scaled_k = scaled.dims[0]
    assert isinstance(scaled_k, MomentumSpace)
    scaled_reciprocal = scaled_k.elements()[0].space
    assert scaled_reciprocal.dual.basis == ImmutableDenseMatrix.diag(
        1, sy.Rational(3, 2), 2
    )
    assert scaled_reciprocal.basis == lattice.dual.basis * 2
    assert [k.rep for k in scaled_k.elements()] == [k.rep for k in k_space.elements()]

    for dim in scaled.dims[1:]:
        assert isinstance(dim, HilbertSpace)
        scaled_offset = dim.elements()[0].irrep_of(Offset)
        assert scaled_offset.rep == offset.rep
        assert scaled_offset.space == scaled_reciprocal.dual


def test_cartesian_scale_rejects_invalid_scale():
    lattice = Lattice(basis=ImmutableDenseMatrix([[2]]), shape=(2,))
    tensor = Tensor(data=torch.arange(2), dims=(brillouin_zone(lattice.dual),))

    with pytest.raises(ValueError, match="expected 1, got 2"):
        cartesian_scale(tensor, (1, 1))
    with pytest.raises(ValueError, match="nonzero"):
        cartesian_scale(tensor, (0,))
    with pytest.raises(TypeError, match="int or sympy.Expr"):
        cartesian_scale(tensor, (0.5,))  # type: ignore[arg-type]
