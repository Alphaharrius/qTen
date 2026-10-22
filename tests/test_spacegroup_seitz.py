"""Seitz operators, space-group little-group characters, and EBR cones.

No notebook import and no network. Geometry is a 1D two-site cell or a 2D
glide, never a named crystal.
"""

import pytest
import torch
import sympy as sy
from sympy import ImmutableDenseMatrix

import qten.ops as Q
from qten.bands import get_band_transform
from qten.geometries.boundary import PeriodicBoundary
from qten.geometries.fourier import fourier_kernel
from qten.geometries.basis_transform import BasisTransform
from qten.geometries.spatials import KPointSet, Lattice, Momentum, Offset
from qten.linalg.tensors import Tensor
from qten.phys import Spin
from qten.pointgroups import (
    PointGroupOpr,
    SpaceGroup,
    ebr_decomposition,
    ebr_indicator,
    ebr_lattice_decomposition,
    hilbert_repr,
    inversion_indicators,
    little_group,
    little_group_irreps,
    pointgroup,
    space_group_character_tables,
    space_group_irreps,
)
from qten.pointgroups.ops import _unwrapped_offset_image
from qten.symbolics.hilbert_space import HilbertSpace, U1Basis
from qten.symbolics.state_space import IndexSpace, brillouin_zone


def _identity_element(group):
    return next(element for element in group.elements() if element.group_order() == 1)


def _1d_two_site_spinful():
    lattice = Lattice(
        basis=ImmutableDenseMatrix([[1]]),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4)),
        unit_cell={
            "A": ImmutableDenseMatrix([0]),
            "B": ImmutableDenseMatrix([sy.Rational(1, 2)]),
        },
    )
    space = HilbertSpace.new(
        [
            U1Basis.new(lattice.at("A"), Spin.up),
            U1Basis.new(lattice.at("A"), Spin.down),
            U1Basis.new(lattice.at("B"), Spin.up),
            U1Basis.new(lattice.at("B"), Spin.down),
        ]
    )
    return lattice, space


def _2d_glide_cell():
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(4, 4)),
        unit_cell={
            "A": ImmutableDenseMatrix([0, 0]),
            "B": ImmutableDenseMatrix([sy.Rational(1, 2), 0]),
        },
    )
    space = HilbertSpace.new(
        U1Basis.new(site, spin)
        for site in (lattice.at("A"), lattice.at("B"))
        for spin in (Spin.up, Spin.down)
    )
    mirror = pointgroup("m", plane="xy").generators[0]
    tau = Offset(ImmutableDenseMatrix([sy.Rational(1, 2), 0]), lattice)
    glide = PointGroupOpr.seitz(mirror, tau)
    return lattice, space, mirror, glide


def test_seitz_half_translation_swaps_sites_linear_identity_does_not():
    lattice, space = _1d_two_site_spinful()
    ident = _identity_element(pointgroup("Cs", plane="x"))
    tau = Offset(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice)
    half = PointGroupOpr.seitz(ident, tau)
    origin = Offset(ImmutableDenseMatrix([0]), lattice)
    linear = PointGroupOpr(ident).fixpoint_at(origin, rebase=True)

    assert (half @ lattice.at("A")).fractional() == lattice.at("B").fractional()
    assert (half @ lattice.at("B")).fractional() == lattice.at("A").fractional()
    assert (linear @ lattice.at("A")).fractional() == lattice.at("A").fractional()
    hilbert_repr(half, space)


def test_seitz_translation_is_not_a_fixpoint_translation():
    _lattice, _space, mirror, glide = _2d_glide_cell()
    # Mirror (x, y) -> (x, -y): (I-R)c never has a nonzero x-component.
    assert sy.simplify(glide.offset.rep[0, 0] - sy.Rational(1, 2)) == 0
    assert sy.simplify(glide.offset.rep[1, 0]) == 0


def test_seitz_glide_closes_and_swaps_sites():
    lattice, space, mirror, glide = _2d_glide_cell()
    origin = Offset(ImmutableDenseMatrix([0, 0]), lattice)
    linear = PointGroupOpr(mirror).fixpoint_at(origin, rebase=True)

    assert (glide @ lattice.at("A")).fractional() == lattice.at("B").fractional()
    assert (glide @ lattice.at("B")).fractional() == lattice.at("A").fractional()
    hilbert_repr(glide, space)
    hilbert_repr(linear, space)
    assert (linear @ lattice.at("A")).fractional() == lattice.at("A").fractional()
    assert (linear @ lattice.at("B")).fractional() == lattice.at("B").fractional()


def test_seitz_product_and_inverse():
    lattice, _space, _mirror, glide = _2d_glide_cell()
    product = glide @ glide
    assert sy.simplify(product.g.irrep - sy.eye(2)).equals(sy.zeros(2))
    # g^2 is a primitive translation, so the factor-group inverse is g itself.
    restored = (glide.inv() @ glide).offset.fractional()
    assert all(sy.simplify(restored.rep[i, 0]) == 0 for i in range(2))
    assert (glide.inv() @ lattice.at("B")).fractional() == lattice.at("A").fractional()


def test_space_group_from_generators_is_order_two():
    _lattice, _space, _mirror, glide = _2d_glide_cell()
    group = SpaceGroup.from_generators([glide])
    assert group.order == 2
    ident = group.elements[0]
    assert all(sy.simplify(ident.offset.fractional().rep[i, 0]) == 0 for i in range(2))
    assert group.elements[1].offset.fractional() == glide.offset.fractional()


def test_from_generators_modulo_drops_primitive_translations_on_a_supercell():
    primitive = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        unit_cell={"A": ImmutableDenseMatrix([0, 0])},
        shape=(6, 6),
    )
    blocked = BasisTransform(ImmutableDenseMatrix.diag(3, 3))(primitive)
    ident = _identity_element(pointgroup("m", plane="xy"))
    tx = PointGroupOpr.seitz(
        ident,
        Offset(ImmutableDenseMatrix([sy.Rational(1, 3), 0]), blocked),
    )
    ty = PointGroupOpr.seitz(
        ident,
        Offset(ImmutableDenseMatrix([0, sy.Rational(1, 3)]), blocked),
    )
    assert SpaceGroup.from_generators([tx, ty]).order == 9
    assert SpaceGroup.from_generators([tx, ty], modulo=primitive).order == 1
    with pytest.raises(ValueError, match="max_order"):
        SpaceGroup.from_generators([tx, ty], max_order=8)


def test_space_group_little_group_keeps_glide_at_zone_edge():
    lattice, _space, _mirror, glide = _2d_glide_cell()
    group = SpaceGroup.from_generators([glide])
    gamma = Momentum(ImmutableDenseMatrix([0, 0]), lattice.dual)
    x_point = Momentum(ImmutableDenseMatrix([sy.Rational(1, 2), 0]), lattice.dual)
    assert len(group.little_group(gamma)) == 2
    assert len(group.little_group(x_point)) == 2


def _z2_tables(k_names):
    return {name: {"A1": (1, 1), "A2": (1, -1)} for name in k_names}


def test_space_group_character_tables_match_little_group_irreps_at_zone_edge():
    c4v = pointgroup("4mm", plane="xy")
    lattice = Lattice(
        basis=ImmutableDenseMatrix.eye(2),
        boundaries=PeriodicBoundary(ImmutableDenseMatrix.diag(2, 2)),
        unit_cell={"r": ImmutableDenseMatrix([0, 0])},
    )
    space = HilbertSpace.new(
        U1Basis.new(lattice.at("r"), spin) for spin in (Spin.up, Spin.down)
    )
    origin = Offset(ImmutableDenseMatrix([0, 0]), lattice)
    seitz = [
        PointGroupOpr(element).fixpoint_at(origin, rebase=True)
        for element in c4v.elements()
    ]
    group = SpaceGroup.from_generators(seitz)
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0, 0), "X": (sy.Rational(1, 2), 0)},
    )
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    tables = space_group_character_tables(c4v, group, kpts)
    assert tables == Q.space_group_character_tables(c4v, group, kpts)
    x_point = next(k for name, k in kpts.points.items() if name == "X")
    g_x = little_group(c4v, x_point)
    assert g_x.order < c4v.order
    assert len(tables["X"][next(iter(tables["X"]))]) == g_x.order
    assert space_group_irreps(seed, group, kpts, tables) == little_group_irreps(
        seed, c4v, kpts, fixpoint=origin
    )


def test_space_group_irreps_full_cell_is_one_copy_of_each_z2_irrep_per_spin():
    lattice, space = _1d_two_site_spinful()
    ident = _identity_element(pointgroup("Cs", plane="x"))
    tau = Offset(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice)
    group = SpaceGroup.from_generators([PointGroupOpr.seitz(ident, tau)])
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0,), "X": (sy.Rational(1, 2),)},
    )
    content = space_group_irreps(seed, group, kpts, _z2_tables(("Gamma", "X")))
    assert content == Q.space_group_irreps(
        seed, group, kpts, _z2_tables(("Gamma", "X"))
    )
    for name in ("Gamma", "X"):
        assert content[name]["A1"] == 2
        assert content[name]["A2"] == 2
    ebrs = {
        "even": {name: {"A1": 1} for name in content},
        "odd": {name: {"A2": 1} for name in content},
    }
    assert ebr_decomposition(content, ebrs) == {"even": 2, "odd": 2}


def test_half_translation_wrap_phase_at_zone_edge():
    lattice, space = _1d_two_site_spinful()
    ident = _identity_element(pointgroup("Cs", plane="x"))
    tau = Offset(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice)
    opr = PointGroupOpr.seitz(ident, tau)
    k_space = brillouin_zone(lattice.dual)
    transform = get_band_transform(opr, k_space, space)
    gamma_repr = hilbert_repr(opr, space)
    images = tuple(
        _unwrapped_offset_image(opr, psi.irrep_of(Offset)) for psi in space.elements()
    )
    mapped = tuple(pair[0] for pair in transform.dims[0].elements())
    phases = fourier_kernel(mapped, images)
    expected = gamma_repr.data.unsqueeze(0) * phases.unsqueeze(1)
    assert torch.allclose(transform.data, expected, rtol=0, atol=1e-12)

    edge = next(
        i
        for i, k in enumerate(k_space.elements())
        if sy.simplify(k.rep[0, 0] - sy.Rational(1, 2)) == 0
    )
    assert not torch.allclose(transform.data[edge], gamma_repr.data, rtol=0, atol=1e-8)


def test_ebr_decomposition_finds_unique_minimal_combination():
    content = {
        "Gamma": {"A1": 1, "A2": 2},
        "X": {"A1": 1, "A2": 2},
    }
    ebrs = {
        "even": {"Gamma": {"A1": 1}, "X": {"A1": 1}},
        "odd": {"Gamma": {"A2": 1}, "X": {"A2": 1}},
    }
    assert ebr_decomposition(content, ebrs) == {"even": 1, "odd": 2}
    assert Q.ebr_decomposition(content, ebrs) == {"even": 1, "odd": 2}


def test_ebr_decomposition_rejects_vector_outside_the_cone():
    content = {"Gamma": {"A1": 1}, "X": {"A2": 1}}
    ebrs = {
        "even": {"Gamma": {"A1": 1}, "X": {"A1": 1}},
        "odd": {"Gamma": {"A2": 1}, "X": {"A2": 1}},
    }
    assert ebr_decomposition(content, ebrs) is None
    report = ebr_indicator(content, ebrs)
    assert report.cone is None
    assert report.in_lattice is False
    assert report.trivial is False
    assert report.lattice is None


def test_ebr_lattice_allows_negative_copies_outside_the_cone():
    content = {"Gamma": {"A2": 2}}
    ebrs = {
        "pair": {"Gamma": {"A1": 2}},
        "mixed": {"Gamma": {"A1": 1, "A2": 1}},
    }
    assert ebr_decomposition(content, ebrs) is None
    assert ebr_lattice_decomposition(content, ebrs) == {"pair": -1, "mixed": 2}
    report = ebr_indicator(content, ebrs)
    assert report.in_lattice is True
    assert report.trivial is True
    assert report.cone is None
    assert report.lattice == {"pair": -1, "mixed": 2}
    assert Q.ebr_indicator(content, ebrs).lattice == report.lattice


def test_ebr_indicator_reports_z4_torsion():
    content = {"Gamma": {"A1": 2}}
    ebrs = {"quad": {"Gamma": {"A1": 4}}}
    report = ebr_indicator(content, ebrs)
    assert report.in_lattice is False
    assert report.trivial is False
    assert report.torsion == ((4, 2),)
    assert report.lattice is None
    assert ebr_decomposition(content, ebrs) is None


def test_inversion_indicators_trivial_and_strong_on_eight_trim():
    half = sy.Rational(1, 2)
    axes = (0, half)
    mesh = [((kx, ky, kz), 2, 2) for kx in axes for ky in axes for kz in axes]
    trivial = inversion_indicators(mesh)
    assert trivial.z4 == 0
    assert trivial.kappa1 == 0
    assert trivial.nu0 == 0
    assert trivial.weak == (0, 0, 0)
    assert trivial.n_trim == 8
    assert trivial.filling == 4

    strong_mesh = [((0, 0, 0), 0, 4)] + mesh[1:]
    strong = inversion_indicators(strong_mesh)
    assert strong.z4 == 1
    assert strong.kappa1 == 3
    assert strong.nu0 == 1
    assert Q.inversion_indicators(strong_mesh).nu0 == 1


def test_inversion_indicators_rejects_odd_counts():
    with pytest.raises(ValueError, match="even"):
        inversion_indicators([((0,), 1, 1)])


def test_inversion_indicators_z4_sums_before_dividing_in_two_dimensions():
    # Each TRIM term is only ≡ 2 (mod 4); the sum over all four is -4.
    half = sy.Rational(1, 2)
    mesh = [((0, 0), 0, 2), ((half, 0), 2, 0), ((0, half), 2, 0), ((half, half), 2, 0)]
    result = inversion_indicators(mesh)
    assert result.z4 == 3
    assert result.kappa1 == 1
    assert result.nu0 == 1


def test_inversion_indicators_requires_every_distinct_trim():
    half = sy.Rational(1, 2)
    with pytest.raises(ValueError, match="Expected all 4 TRIM"):
        inversion_indicators([((0, 0), 2, 2), ((half, 0), 2, 2)])
    with pytest.raises(ValueError, match="distinct"):
        inversion_indicators([((0, 0), 2, 2), ((0, 0), 2, 2)])


def test_space_group_irreps_requires_table_for_each_hsp():
    lattice, space = _1d_two_site_spinful()
    ident = _identity_element(pointgroup("Cs", plane="x"))
    tau = Offset(ImmutableDenseMatrix([sy.Rational(1, 2)]), lattice)
    group = SpaceGroup.from_generators([PointGroupOpr.seitz(ident, tau)])
    seed = Tensor(
        data=torch.eye(space.dim, dtype=torch.complex128),
        dims=(space, IndexSpace.linear(space.dim)),
    )
    kpts = KPointSet.from_points(
        lattice.dual,
        {"Gamma": (0,), "X": (sy.Rational(1, 2),)},
    )
    with pytest.raises(KeyError, match="X"):
        space_group_irreps(seed, group, kpts, {"Gamma": {"A1": (1, 1)}})
