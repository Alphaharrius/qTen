import math

import sympy as sy
import torch
import pytest
from sympy import ImmutableDenseMatrix

from qten.phys import Spin, expand_spin, su2_from_so3, su2_of_point_group
from qten.phys.spin import proper_rotation_matrix
from qten.pointgroups import (
    FinitePointGroup,
    JointSpinfulPhaseSector,
    PointGroupElement,
    PointGroupOpr,
    SpinorIrrepSector,
    SpinfulPhaseSector,
    pointgroup,
)
from qten.pointgroups.ops import (
    _hilbert_opr_repr,
    joint_point_group_column_symmetrize,
    spinful_hilbert_opr_repr,
    spinful_transform_basis,
)
from qten.pointgroups._registry import known_point_group_symbols
import qten
import qten.ops as Q
from qten.precision import get_precision_config
from qten.symbolics import HilbertSpace, IndexSpace, U1Basis
from qten.geometries.spatials import AffineSpace, Lattice, Offset


def _affine_space():
    return AffineSpace(basis=ImmutableDenseMatrix.eye(3))


def _site(x=0, y=0, z=0):
    space = _affine_space()
    return Offset(rep=ImmutableDenseMatrix([x, y, z]), space=space)


def _c4z():
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z)))


def _c2z():
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z)))


def test_spin_labels_and_ordering():
    assert Spin.up.is_up and Spin.down.is_down
    assert Spin.up.ms == sy.Rational(1, 2)
    assert Spin.up < Spin.down
    with pytest.raises(ValueError):
        Spin(0)


def test_su2_identity_and_unitarity():
    identity = ImmutableDenseMatrix.eye(3)
    u = su2_from_so3(identity)
    assert u == ImmutableDenseMatrix.eye(2)

    R = ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  # C4z
    u = su2_from_so3(R)
    uh_u = sy.simplify(u.H @ u)
    assert uh_u == ImmutableDenseMatrix.eye(2)
    assert sy.simplify(u.det()) == 1


def test_su2_rejects_non_so3_inputs():
    with pytest.raises(ValueError, match="3x3"):
        su2_from_so3(ImmutableDenseMatrix.eye(2))
    with pytest.raises(ValueError, match="orthogonal"):
        su2_from_so3(ImmutableDenseMatrix([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))


def test_su2_of_point_group_pads_2d_xy_to_c4z():
    planar = pointgroup("c4-xy:xy")
    padded = su2_from_so3(ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    assert sy.simplify(su2_of_point_group(planar) - padded) == (
        ImmutableDenseMatrix.zeros(2)
    )


def test_su2_c2z_is_minus_i_sigma_z():
    R = ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    u = su2_from_so3(R)
    expected = ImmutableDenseMatrix([[-sy.I, 0], [0, sy.I]])
    assert sy.simplify(u - expected) == ImmutableDenseMatrix.zeros(2)


@pytest.mark.parametrize("angle_sign", [1, -1])
def test_su2_near_pi_uses_stable_oriented_axis(angle_sign):
    delta = sy.Float("1e-11", 30)
    angle = angle_sign * (sy.pi - delta)
    rotation = ImmutableDenseMatrix(
        [
            [sy.cos(angle), -sy.sin(angle), 0],
            [sy.sin(angle), sy.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    u = su2_from_so3(rotation)
    sigma_z = ImmutableDenseMatrix([[1, 0], [0, -1]])
    expected = sy.cos(angle / 2) * sy.eye(2) - sy.I * sy.sin(angle / 2) * sigma_z
    max_error = max(abs(complex(sy.N(entry))) for entry in u - expected)
    assert max_error < 1e-9


def test_su2_inexact_arbitrary_axis_is_stable_near_pi():
    axis = [component / math.sqrt(14) for component in (1, 2, 3)]
    angle = math.pi - 1.1e-10
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = ImmutableDenseMatrix(
        [
            [
                cosine + x * x * (1 - cosine),
                x * y * (1 - cosine) - z * sine,
                x * z * (1 - cosine) + y * sine,
            ],
            [
                y * x * (1 - cosine) + z * sine,
                cosine + y * y * (1 - cosine),
                y * z * (1 - cosine) - x * sine,
            ],
            [
                z * x * (1 - cosine) - y * sine,
                z * y * (1 - cosine) + x * sine,
                cosine + z * z * (1 - cosine),
            ],
        ]
    )
    u = su2_from_so3(rotation)
    sigma_x = ImmutableDenseMatrix([[0, 1], [1, 0]])
    sigma_y = ImmutableDenseMatrix([[0, -sy.I], [sy.I, 0]])
    sigma_z = ImmutableDenseMatrix([[1, 0], [0, -1]])
    expected = math.cos(angle / 2) * sy.eye(2) - sy.I * math.sin(angle / 2) * (
        x * sigma_x + y * sigma_y + z * sigma_z
    )
    max_error = max(abs(complex(sy.N(entry))) for entry in u - expected)
    assert max_error < 1e-12


def test_su2_inexact_small_rotation_does_not_collapse_to_identity():
    angle = 1.5e-8
    rotation = ImmutableDenseMatrix(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    u = su2_from_so3(rotation)
    assert u != ImmutableDenseMatrix.eye(2)
    expected = ImmutableDenseMatrix(
        [
            [math.cos(angle / 2) - sy.I * math.sin(angle / 2), 0],
            [0, math.cos(angle / 2) + sy.I * math.sin(angle / 2)],
        ]
    )
    assert max(abs(complex(sy.N(entry))) for entry in u - expected) < 1e-15


def test_su2_rejects_parameterized_symbolic_rotation_until_substituted():
    angle = sy.symbols("angle", real=True)
    rotation = ImmutableDenseMatrix(
        [
            [sy.cos(angle), -sy.sin(angle), 0],
            [sy.sin(angle), sy.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    with pytest.raises(ValueError, match="substitute"):
        su2_from_so3(rotation)


def test_su2_c3_matches_axis_angle_and_has_double_order():
    rotation = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    u = su2_from_so3(rotation)
    expected = ImmutableDenseMatrix(
        [
            [(1 - sy.I) / 2, (-1 - sy.I) / 2],
            [(1 - sy.I) / 2, (1 + sy.I) / 2],
        ]
    )
    assert sy.simplify(u - expected) == ImmutableDenseMatrix.zeros(2)
    assert sy.simplify(u**3) == -ImmutableDenseMatrix.eye(2)
    assert sy.simplify(u**6) == ImmutableDenseMatrix.eye(2)


def test_su2_point_group_lift_is_invariant_under_nonorthogonal_rebase():
    x, y, z = sy.symbols("x y z")
    rotation = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    opr = PointGroupOpr(PointGroupElement(irrep=rotation, axes=(x, y, z)))
    nonorthogonal = AffineSpace(
        basis=ImmutableDenseMatrix([[1, sy.Rational(1, 2), 0], [0, 1, 0], [0, 0, 2]])
    )
    center = Offset(rep=ImmutableDenseMatrix.zeros(3, 1), space=nonorthogonal)
    rebased = opr.fixpoint_at(center, rebase=True)
    assert rebased.g.irrep != rotation
    assert sy.simplify(su2_of_point_group(rebased) - su2_of_point_group(opr)) == (
        ImmutableDenseMatrix.zeros(2)
    )


def test_improper_uses_proper_factor():
    # mirror σ_z = diag(1,1,-1) → proper part -σ_z = C2z
    R = ImmutableDenseMatrix([[1, 0, 0], [0, 1, 0], [0, 0, -1]])
    assert proper_rotation_matrix(R) == ImmutableDenseMatrix(
        [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    )


def test_expand_spin_c2z_phases():
    opr = _c2z()
    up_img = {spin: amp for amp, spin in expand_spin(opr, Spin.up)}
    dn_img = {spin: amp for amp, spin in expand_spin(opr, Spin.down)}
    assert set(up_img) == {Spin.up}
    assert set(dn_img) == {Spin.down}
    assert sy.simplify(up_img[Spin.up] - (-sy.I)) == 0
    assert sy.simplify(dn_img[Spin.down] - sy.I) == 0


def test_spinful_transform_basis_moves_site_and_spin():
    opr = _c4z().fixpoint_at(_site())
    psi = U1Basis.new(_site(1, 0, 0), Spin.up)
    image = spinful_transform_basis(opr, psi)
    # C4z: (1,0,0) -> (0,1,0); spin gets diagonal SU(2) phases
    sites = {term.irrep_of(Offset).rep for term in image.span}
    spins = {term.irrep_of(Spin) for term in image.span}
    assert sites == {ImmutableDenseMatrix([0, 1, 0])}
    assert spins == {Spin.up}
    assert len(image.span) == 1
    assert sy.simplify(image.span[0].coef - (1 - sy.I) / sy.sqrt(2)) == 0


def test_spinful_hilbert_repr_is_unitary_and_mixes_for_c3():
    # C3 about [111]
    x, y, z = sy.symbols("x y z")
    R = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    opr = PointGroupOpr(PointGroupElement(irrep=R, axes=(x, y, z))).fixpoint_at(_site())
    space = HilbertSpace.new(
        [
            U1Basis.new(_site(), Spin.up),
            U1Basis.new(_site(), Spin.down),
        ]
    )
    D = spinful_hilbert_opr_repr(opr, space)
    assert D.data.shape == (2, 2)
    # unitarity
    eye = torch.eye(2, dtype=D.data.dtype)
    assert torch.allclose(D.data.conj().T @ D.data, eye, rtol=0, atol=1e-12)
    expected = torch.tensor(
        [
            [(1 - 1j) / 2, (-1 - 1j) / 2],
            [(1 - 1j) / 2, (1 + 1j) / 2],
        ],
        dtype=D.data.dtype,
    )
    assert torch.allclose(D.data, expected, rtol=0, atol=1e-12)


def test_hilbert_opr_repr_dispatches_to_spinful():
    opr = _c2z().fixpoint_at(_site())
    space = HilbertSpace.new(
        [U1Basis.new(_site(), Spin.up), U1Basis.new(_site(), Spin.down)]
    )
    D = _hilbert_opr_repr(opr, space)
    # diag(-i, i)
    assert torch.allclose(
        D.data,
        torch.tensor([[-1j, 0], [0, 1j]], dtype=D.data.dtype),
        rtol=0,
        atol=1e-12,
    )


def test_spinful_hilbert_repr_respects_basis_gauge_phases():
    site = _site()
    x, y, z = sy.symbols("x y z")
    rotation = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    opr = PointGroupOpr(PointGroupElement(irrep=rotation, axes=(x, y, z))).fixpoint_at(
        site
    )
    canonical_space = HilbertSpace.new(
        [U1Basis.new(site, Spin.up), U1Basis.new(site, Spin.down)]
    )
    space = HilbertSpace.new(
        [
            U1Basis(sy.I, (site, Spin.up)),
            U1Basis(-sy.I, (site, Spin.down)),
        ]
    )
    canonical = spinful_hilbert_opr_repr(opr, canonical_space)
    D = spinful_hilbert_opr_repr(opr, space)
    gauge = torch.diag(torch.tensor([1j, -1j], dtype=D.data.dtype))
    expected = gauge.conj().T @ canonical.data @ gauge
    assert torch.allclose(D.data, expected, rtol=0, atol=1e-12)


def test_spinful_hilbert_repr_rejects_nonunit_basis_coefficients():
    site = _site()
    space = HilbertSpace.new(
        [
            U1Basis(sy.Integer(2), (site, Spin.up)),
            U1Basis(sy.Integer(2), (site, Spin.down)),
        ]
    )
    with pytest.raises(ValueError, match="unit modulus"):
        spinful_hilbert_opr_repr(_c2z().fixpoint_at(site), space)


def test_spinful_hilbert_repr_rejects_duplicate_physical_rays():
    site = _site()
    space = HilbertSpace.new(
        [
            U1Basis(sy.Integer(1), (site, Spin.up)),
            U1Basis(sy.Integer(-1), (site, Spin.up)),
            U1Basis(sy.Integer(1), (site, Spin.down)),
            U1Basis(sy.Integer(-1), (site, Spin.down)),
        ]
    )
    with pytest.raises(ValueError, match="unique physical rays"):
        spinful_hilbert_opr_repr(_c2z().fixpoint_at(site), space)


def test_spinful_hilbert_repr_rejects_symbolic_gauge_phase():
    site = _site()
    phase = sy.symbols("phase", real=True)
    space = HilbertSpace.new(
        [
            U1Basis(sy.exp(sy.I * phase), (site, Spin.up)),
            U1Basis.new(site, Spin.down),
        ]
    )
    with pytest.raises(ValueError, match="numerically evaluable"):
        spinful_hilbert_opr_repr(_c2z().fixpoint_at(site), space)


def test_spinful_hilbert_repr_rejects_mixed_spin_space():
    site = _site()
    space = HilbertSpace.new([U1Basis.new(site, Spin.up), U1Basis.new(_site(1, 0, 0))])
    with pytest.raises(ValueError, match="exactly one Spin label"):
        spinful_hilbert_opr_repr(_c2z().fixpoint_at(site), space)


def test_point_group_column_symmetrize_spinful_c2():
    opr = _c2z().fixpoint_at(_site())
    space = HilbertSpace.new(
        [U1Basis.new(_site(), Spin.up), U1Basis.new(_site(), Spin.down)]
    )
    # seed = up + down
    w = qten.Tensor(
        data=torch.tensor([[1.0], [1.0]], dtype=torch.complex128)
        / torch.sqrt(torch.tensor(2.0)),
        dims=(space, IndexSpace.linear(1)),
    )
    D = spinful_hilbert_opr_repr(opr, space)
    identity = torch.eye(2, dtype=D.data.dtype)
    powers = [identity]
    for _ in range(3):
        powers.append(powers[-1] @ D.data)
    projectors = []
    for m in range(4):
        phase = torch.exp(torch.tensor(2j * torch.pi * m / 4, dtype=D.data.dtype))
        projector = sum((phase ** (-k)) * power for k, power in enumerate(powers)) / 4
        projectors.append(projector)
        assert torch.allclose(projector @ projector, projector, rtol=0, atol=1e-12)
    assert torch.allclose(sum(projectors), identity, rtol=0, atol=1e-12)

    out = Q.point_group_column_symmetrize(opr, w, full_sector=True)
    assert out.data.shape == (2, 2)
    assert isinstance(out.dims[1], HilbertSpace)
    for j, label in enumerate(out.dims[1].elements()):
        sector = label.irrep_of(SpinfulPhaseSector)
        phase = complex(sy.N(sector.phase))
        assert sector.spatial_order == 2
        assert torch.allclose(
            D.data @ out.data[:, j],
            phase * out.data[:, j],
            rtol=0,
            atol=1e-12,
        )


def test_su2_of_td_element_unitarity():
    td = pointgroup("-43m")
    assert isinstance(td, object)
    for g in td.elements():
        u = su2_of_point_group(g)
        assert sy.simplify(u.H @ u) == ImmutableDenseMatrix.eye(2)
        assert sy.simplify(u.det()) == 1


def test_full_td_spinful_symmetrize_uses_spinor_character_table():
    td = pointgroup("-43m")
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    elements = td.elements()
    representations = [
        spinful_hilbert_opr_repr(PointGroupOpr(element).fixpoint_at(center), space).data
        for element in elements
    ]
    projectors: dict[str, torch.Tensor] = {}
    for irrep_name, irrep_data in td.spinor_irreps["irreps"].items():
        characters = td.spinor_irrep_characters_by_element(irrep_name)
        projector = sum(
            (
                character.conjugate() * representation
                for character, representation in zip(characters, representations)
            ),
            torch.zeros_like(representations[0]),
        )
        projector *= int(irrep_data["dim"]) / td.order
        projectors[irrep_name] = projector
        assert torch.allclose(projector, projector.conj().T, rtol=0, atol=1e-12)
        assert torch.allclose(projector @ projector, projector, rtol=0, atol=1e-12)

        signs = [1 if index % 2 == 0 else -1 for index in range(td.order)]
        resectioned = sum(
            (
                (sign * character).conjugate() * (sign * representation)
                for sign, character, representation in zip(
                    signs, characters, representations
                )
            ),
            torch.zeros_like(projector),
        )
        resectioned *= int(irrep_data["dim"]) / td.order
        assert torch.allclose(resectioned, projector, rtol=0, atol=1e-12)

        full_double_group = sum(
            (
                character.conjugate() * representation
                + (-character).conjugate() * (-representation)
                for character, representation in zip(characters, representations)
            ),
            torch.zeros_like(projector),
        )
        full_double_group *= int(irrep_data["dim"]) / (2 * td.order)
        assert torch.allclose(full_double_group, projector, rtol=0, atol=1e-12)

    identity = torch.eye(space.dim, dtype=torch.complex128)
    assert torch.allclose(sum(projectors.values()), identity, rtol=0, atol=1e-12)
    projector_values = list(projectors.values())
    for i, left in enumerate(projector_values):
        for right in projector_values[i + 1 :]:
            assert torch.allclose(
                left @ right, torch.zeros_like(left), rtol=0, atol=1e-12
            )

    w = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    projected = Q.point_group_column_symmetrize(
        td, w, full_sector=True, fixpoint=center
    )
    assert projected.data.shape == (2, 2)
    assert torch.allclose(
        projected.data.conj().T @ projected.data,
        identity,
        rtol=0,
        atol=1e-12,
    )
    sectors = [
        label.irrep_of(SpinorIrrepSector) for label in projected.dims[1].elements()
    ]
    assert len({sector.irrep for sector in sectors}) == 1
    assert all(
        sector.dim == 2 and sector.source == "qten-su2-principal-v1"
        for sector in sectors
    )


def test_custom_spinful_finite_group_without_table_computes_spinor_irreps():
    x, y, z = sy.symbols("x y z")
    custom = FinitePointGroup.from_matrices(
        (ImmutableDenseMatrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),),
        axes=(x, y, z),
        symbol="custom-c2",
    )
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    seed = qten.Tensor(
        data=torch.eye(2, 1, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(1)),
    )
    projected = Q.point_group_column_symmetrize(custom, seed, fixpoint=center)
    assert projected.data.shape[0] == 2
    assert all(
        isinstance(label.irrep_of(SpinorIrrepSector), SpinorIrrepSector)
        for label in projected.dims[1].elements()
    )


def test_all_packaged_spinor_tables_resolve_bare_spin_space():
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    seed = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    identity = torch.eye(2, dtype=torch.complex128)

    for symbol in known_point_group_symbols():
        projected = Q.point_group_column_symmetrize(
            pointgroup(symbol),
            seed,
            full_sector=True,
            fixpoint=center,
        )
        assert projected.data.shape == (2, 2), symbol
        assert torch.allclose(
            projected.data.conj().T @ projected.data,
            identity,
            rtol=0,
            atol=1e-11,
        ), symbol
        assert all(
            isinstance(label.irrep_of(SpinorIrrepSector), SpinorIrrepSector)
            for label in projected.dims[1].elements()
        )


def test_spinful_td_operator_average_needs_no_double_group_table():
    td = pointgroup("-43m")
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    operator = qten.Tensor(
        data=torch.tensor([[3, 1 + 2j], [1 - 2j, 1]], dtype=torch.complex128),
        dims=(space, space),
    )

    averaged = Q.point_group_operator_symmetrize(td, operator, fixpoint=center)
    expected = 2 * torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(averaged.data, expected, rtol=0, atol=1e-12)

    averaged_twice = Q.point_group_operator_symmetrize(td, averaged, fixpoint=center)
    assert torch.allclose(averaged_twice.data, averaged.data, rtol=0, atol=1e-12)

    for element in td.elements():
        representation = spinful_hilbert_opr_repr(
            PointGroupOpr(element).fixpoint_at(center), space
        )
        transformed = representation.data @ averaged.data @ representation.data.conj().T
        assert torch.allclose(transformed, averaged.data, rtol=0, atol=1e-12)


def test_operator_average_rejects_different_leg_gauges_clearly():
    td = pointgroup("-43m")
    center = _site()
    row_space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    col_space = HilbertSpace.new(
        [
            U1Basis(sy.I, (center, Spin.up)),
            U1Basis(-sy.I, (center, Spin.down)),
        ]
    )
    operator = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(row_space, col_space),
    )
    with pytest.raises(ValueError, match="same ordered Hilbert-space basis"):
        Q.point_group_operator_symmetrize(td, operator, fixpoint=center)


def test_joint_spinful_symmetrize_requires_parent_group_and_commuting_lifts():
    x, y, z = sy.symbols("x y z")
    center = _site()
    c2x = PointGroupOpr(
        PointGroupElement(
            irrep=ImmutableDenseMatrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
            axes=(x, y, z),
        )
    ).fixpoint_at(center)
    c2y = PointGroupOpr(
        PointGroupElement(
            irrep=ImmutableDenseMatrix([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
            axes=(x, y, z),
        )
    ).fixpoint_at(center)
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    w = qten.Tensor(
        data=torch.eye(2, 1, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(1)),
    )
    with pytest.raises(ValueError, match="group="):
        joint_point_group_column_symmetrize((c2x, c2y), w)

    d2 = pointgroup("222")
    with pytest.raises(ValueError, match="commuting Hilbert-space"):
        joint_point_group_column_symmetrize((c2x, c2y), w, group=d2)


def test_joint_spinful_symmetrize_accepts_commuting_elements_of_one_group():
    center = _site()
    c4 = pointgroup("4")
    by_order = {element.group_order(): element for element in c4.elements()}
    oprs = (
        PointGroupOpr(by_order[4]).fixpoint_at(center),
        PointGroupOpr(by_order[2]).fixpoint_at(center),
    )
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    w = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    projected = joint_point_group_column_symmetrize(oprs, w, full_sector=True, group=c4)
    assert projected.data.shape[0] == 2
    assert projected.data.shape[1] >= 1
    representations = [_hilbert_opr_repr(opr, space).data for opr in oprs]
    for j, label in enumerate(projected.dims[1].elements()):
        column = projected.data[:, j]
        assert torch.allclose(
            column.conj() @ column, torch.tensor(1.0, dtype=column.dtype), atol=1e-12
        )
        sector = label.irrep_of(JointSpinfulPhaseSector)
        assert len(sector.phases) == 2
        for representation, phase_expr in zip(representations, sector.phases):
            phase = complex(sy.N(phase_expr))
            assert torch.allclose(
                representation @ column, phase * column, rtol=0, atol=1e-12
            )


def test_spinful_hilbert_opr_repr_folds_fractional_lattice_offsets():
    """
    Unit-cell spaces store Offset.fractional() labels; D(g) must fold images.

    C2z sends B=(1/2,1/2,1/2) -> (-1/2,-1/2,1/2), which equals B only after
    fractional fold. Without folding, lookup raises even though the unit cell
    is closed under the operation.
    """
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(3),
        shape=(2, 2, 2),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0, 0]),
            "B": ImmutableDenseMatrix(
                [sy.Rational(1, 2), sy.Rational(1, 2), sy.Rational(1, 2)]
            ),
        },
    )
    A = lattice.unit_cell["A"]
    B = lattice.unit_cell["B"]
    assert A == A.fractional() and B == B.fractional()
    space = HilbertSpace.new(
        [
            U1Basis.new(A, Spin.up),
            U1Basis.new(A, Spin.down),
            U1Basis.new(B, Spin.up),
            U1Basis.new(B, Spin.down),
        ]
    )
    opr = _c2z().fixpoint_at(A, rebase=True)
    # Raw geometric image of B is outside the unit cell.
    moved = opr @ B
    assert moved != B
    assert moved.fractional() == B

    D = spinful_hilbert_opr_repr(opr, space)
    assert D.data.shape == (4, 4)
    eye = torch.eye(4, dtype=D.data.dtype)
    assert torch.allclose(D.data.conj().T @ D.data, eye, rtol=0, atol=1e-12)


def test_spinful_hilbert_repr_does_not_fold_open_affine_offsets():
    site = _site(sy.Rational(1, 2), 0, 0)
    space = HilbertSpace.new([U1Basis.new(site, Spin.up), U1Basis.new(site, Spin.down)])
    with pytest.raises(ValueError, match="not in space"):
        spinful_hilbert_opr_repr(_c2z().fixpoint_at(_site()), space)


def test_spinful_projection_uses_active_precision():
    previous_precision = get_precision_config()
    previous_torch_default = torch.get_default_dtype()
    qten.set_precision(32)
    try:
        site = _site()
        opr = _c2z().fixpoint_at(site)
        space = HilbertSpace.new(
            [U1Basis.new(site, Spin.up), U1Basis.new(site, Spin.down)]
        )
        w = qten.Tensor(
            data=1e-6 * torch.tensor([[1.0], [1.0]], dtype=torch.complex64),
            dims=(space, IndexSpace.linear(1)),
        )
        out = Q.point_group_column_symmetrize(opr, w, full_sector=True)
        assert out.data.dtype == torch.complex64
        assert out.data.shape == (2, 2)
    finally:
        qten.set_precision(previous_precision.torch_float, set_torch_default=False)
        torch.set_default_dtype(previous_torch_default)


def test_point_group_basis_tensor_spin_c4v_is_closed():
    c4v = pointgroup("C4v")
    e_basis = c4v.irrep_basis(1, "E")
    assert {sy.simplify(basis.expr) for basis in e_basis} == set(sy.symbols("x y"))
    space = HilbertSpace.new(
        [U1Basis.new(basis, Spin.up) for basis in e_basis]
        + [U1Basis.new(basis, Spin.down) for basis in e_basis]
    )
    representation = _hilbert_opr_repr(PointGroupOpr(c4v.generators[0]), space)
    identity = torch.eye(space.dim, dtype=representation.data.dtype)
    assert torch.allclose(
        representation.data.conj().T @ representation.data,
        identity,
        rtol=0,
        atol=1e-12,
    )
    seed = qten.Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    projected = Q.point_group_column_symmetrize(c4v, seed, full_sector=True)
    assert projected.data.shape[0] == space.dim
    gram = projected.data.conj().T @ projected.data
    assert torch.allclose(
        gram, torch.eye(gram.shape[0], dtype=gram.dtype), rtol=0, atol=1e-11
    )


def test_point_group_basis_tensor_spin_td_projectors_are_complete():
    td = pointgroup("-43m")
    t2 = td.irrep_basis(1, "T2")
    space = HilbertSpace.new(
        [U1Basis.new(basis, Spin.up) for basis in t2]
        + [U1Basis.new(basis, Spin.down) for basis in t2]
    )
    seed = qten.Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    projected = Q.point_group_column_symmetrize(td, seed, full_sector=True)
    assert projected.data.shape == (6, 6)
    identity = torch.eye(6, dtype=torch.complex128)
    assert torch.allclose(
        projected.data.conj().T @ projected.data,
        identity,
        rtol=0,
        atol=1e-11,
    )
    assert all(
        isinstance(label.irrep_of(SpinorIrrepSector), SpinorIrrepSector)
        for label in projected.dims[1].elements()
    )


def test_reoriented_td_keeps_spinor_completeness():
    td = pointgroup("-43m")
    rotation = ImmutableDenseMatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    reoriented = td.reoriented_by(rotation)
    assert reoriented.order == td.order
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    seed = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    projected = Q.point_group_column_symmetrize(
        reoriented, seed, full_sector=True, fixpoint=center
    )
    identity = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(
        projected.data.conj().T @ projected.data,
        identity,
        rtol=0,
        atol=1e-11,
    )


def test_reoriented_d3d_contains_c3_along_111():
    d3d = pointgroup("-3m")
    sqrt2 = sy.sqrt(2)
    sqrt3 = sy.sqrt(3)
    sqrt6 = sy.sqrt(6)
    rotation = ImmutableDenseMatrix(
        [
            [1 / sqrt2, 1 / sqrt6, 1 / sqrt3],
            [-1 / sqrt2, 1 / sqrt6, 1 / sqrt3],
            [0, -2 / sqrt6, 1 / sqrt3],
        ]
    )
    reoriented = d3d.reoriented_by(rotation)
    c3_111 = ImmutableDenseMatrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    c3_111_inv = ImmutableDenseMatrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    generated = {
        tuple(sy.simplify(entry) for entry in element.irrep)
        for element in reoriented.elements()
    }
    assert tuple(c3_111) in generated or tuple(c3_111_inv) in generated
    center = _site()
    space = HilbertSpace.new(
        [U1Basis.new(center, Spin.up), U1Basis.new(center, Spin.down)]
    )
    seed = qten.Tensor(
        data=torch.eye(2, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(2)),
    )
    projected = Q.point_group_column_symmetrize(
        reoriented, seed, full_sector=True, fixpoint=center
    )
    identity = torch.eye(2, dtype=torch.complex128)
    assert torch.allclose(
        projected.data.conj().T @ projected.data,
        identity,
        rtol=0,
        atol=1e-11,
    )
